from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from uav_control import (
    DuctedFanController,
    require_mujoco_id,
)


ARDUPILOT_JSON_PORT = 9002

MAGIC_16_CHANNEL = 18458
MAGIC_32_CHANNEL = 29569

SERVO_PACKET_16 = struct.Struct("<HHI16H")
SERVO_PACKET_32 = struct.Struct("<HHI32H")

# MuJoCo：NWU/FLU
# ArduPilot：NED/FRD
AXIS_FLIP = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class ServoPacket:
    magic: int
    frame_rate: int
    frame_count: int
    pwm: tuple[int, ...]


def parse_servo_packet(data: bytes) -> ServoPacket:
    if len(data) == SERVO_PACKET_16.size:
        unpacked = SERVO_PACKET_16.unpack(data)
        expected_magic = MAGIC_16_CHANNEL

    elif len(data) == SERVO_PACKET_32.size:
        unpacked = SERVO_PACKET_32.unpack(data)
        expected_magic = MAGIC_32_CHANNEL

    else:
        raise ValueError(
            f"unexpected servo packet length {len(data)}; "
            f"expected {SERVO_PACKET_16.size} or "
            f"{SERVO_PACKET_32.size} bytes"
        )

    magic, frame_rate, frame_count, *pwm = unpacked

    if magic != expected_magic:
        raise ValueError(
            f"bad JSON SITL magic {magic}; "
            f"expected {expected_magic}"
        )

    return ServoPacket(
        magic=int(magic),
        frame_rate=max(int(frame_rate), 1),
        frame_count=int(frame_count),
        pwm=tuple(int(value) for value in pwm),
    )


def rotation_matrix_to_euler(
    rotation: np.ndarray,
) -> np.ndarray:
    """从机体到地面的旋转矩阵提取roll、pitch、yaw。"""

    sin_pitch = float(
        np.clip(
            -rotation[2, 0],
            -1.0,
            1.0,
        )
    )

    pitch = math.asin(sin_pitch)

    if abs(abs(sin_pitch) - 1.0) < 1e-7:
        roll = 0.0
        yaw = math.atan2(
            -float(rotation[0, 1]),
            float(rotation[1, 1]),
        )
    else:
        roll = math.atan2(
            float(rotation[2, 1]),
            float(rotation[2, 2]),
        )
        yaw = math.atan2(
            float(rotation[1, 0]),
            float(rotation[0, 0]),
        )

    return np.array(
        [roll, pitch, yaw],
        dtype=float,
    )


class MuJoCoArduPilotBridge:
    def __init__(
        self,
        xml_path: Path,
        bind_address: str = "0.0.0.0",
        port: int = ARDUPILOT_JSON_PORT,
        show_viewer: bool = True,
    ) -> None:
        self.xml_path = xml_path
        self.bind_address = bind_address
        self.port = int(port)
        self.show_viewer = bool(show_viewer)

        self.model = mujoco.MjModel.from_xml_path(
            str(xml_path)
        )
        self.data = mujoco.MjData(self.model)

        self.uav_body_id = require_mujoco_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "uav",
        )

        self.root_joint_id = require_mujoco_id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "uav_free",
        )

        self.gyro_sensor_id = require_mujoco_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "imu_gyro",
        )

        self.accel_sensor_id = require_mujoco_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "imu_accelerometer",
        )

        self.rangefinder_site_id = require_mujoco_id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "rangefinder_site",
        )

        self.root_qpos_address = int(
            self.model.jnt_qposadr[
                self.root_joint_id
            ]
        )
        self.root_dof_address = int(
            self.model.jnt_dofadr[
                self.root_joint_id
            ]
        )

        self.gyro_address = int(
            self.model.sensor_adr[
                self.gyro_sensor_id
            ]
        )
        self.accel_address = int(
            self.model.sensor_adr[
                self.accel_sensor_id
            ]
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.controller = DuctedFanController(
            self.model,
            self.data,
            max_thrust_per_duct=140.09,
            max_vane_angle_deg=10.0,
            motor_time_constant=0.08,
            servo_time_constant=0.10,
            throttle_exponent=1.0,
            vane_force_efficiency=0.55,
            vertical_deflection_loss=0.35,
            yaw_differential_sign=-1.0,
            max_blade_speed_rad_s=1250.0,
        )

        self.origin_world = (
            self._world_position().copy()
        )

        self.last_frame_count: int | None = None

        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self.socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        self.socket.bind(
            (self.bind_address, self.port)
        )
        self.socket.settimeout(0.05)

        self.last_status_wall_time = 0.0
        self.last_packet_wall_time = (
            time.monotonic()
        )
        self.wait_message_printed = False

        self.viewer = None
        self.viewer_period = 1.0 / 60.0
        self.last_viewer_sync_time = 0.0

    def _world_position(self) -> np.ndarray:
        start = self.root_qpos_address
        return self.data.qpos[
            start : start + 3
        ].copy()

    def _world_velocity(self) -> np.ndarray:
        start = self.root_dof_address
        return self.data.qvel[
            start : start + 3
        ].copy()

    def _gyro_flu(self) -> np.ndarray:
        return self.data.sensordata[
            self.gyro_address :
            self.gyro_address + 3
        ].copy()

    def _accelerometer_flu(self) -> np.ndarray:
        return self.data.sensordata[
            self.accel_address :
            self.accel_address + 3
        ].copy()

    def _rotation_nwu_flu(self) -> np.ndarray:
        return self.data.xmat[
            self.uav_body_id
        ].reshape(3, 3).copy()

    def reset_physics(self) -> None:
        mujoco.mj_resetData(
            self.model,
            self.data,
        )
        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.controller.reset()
        self.origin_world = (
            self._world_position().copy()
        )
        self.last_frame_count = None

        print(
            "SITL restarted; "
            "MuJoCo state reset"
        )

    def advance_physics(
        self,
        pwm: Sequence[int],
        frame_count: int,
        frame_rate: int,
    ) -> bool:
        if self.last_frame_count is not None:
            if frame_count == self.last_frame_count:
                return False

            wrapped = (
                self.last_frame_count > 0xF0000000
                and frame_count < 0x0FFFFFFF
            )

            if (
                frame_count < self.last_frame_count
                and not wrapped
            ):
                self.reset_physics()

        self.last_frame_count = int(frame_count)

        control_period = (
            1.0
            / float(max(int(frame_rate), 1))
        )

        maximum_physics_step = 0.001

        substep_count = max(
            1,
            int(
                math.ceil(
                    control_period
                    / maximum_physics_step
                )
            ),
        )

        physics_dt = (
            control_period / substep_count
        )

        self.model.opt.timestep = physics_dt

        for _ in range(substep_count):
            self.controller.update(
                pwm,
                physics_dt,
            )
            mujoco.mj_step(
                self.model,
                self.data,
            )

        if not np.all(
            np.isfinite(self.data.qpos)
        ):
            raise RuntimeError(
                "MuJoCo state contains NaN or infinity"
            )

        if not np.all(
            np.isfinite(self.data.qvel)
        ):
            raise RuntimeError(
                "MuJoCo velocity contains NaN or infinity"
            )

        return True

    def start_viewer(self) -> None:
        if self.viewer is not None:
            return

        self.viewer = (
            mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=True,
            )
        )

        with self.viewer.lock():
            self.viewer.cam.type = (
                mujoco.mjtCamera.mjCAMERA_TRACKING
            )
            self.viewer.cam.trackbodyid = (
                self.uav_body_id
            )
            self.viewer.cam.distance = 2.8
            self.viewer.cam.azimuth = 135.0
            self.viewer.cam.elevation = -20.0

            # 不启用BODY坐标系
            # 不启用JOINT显示

        self.viewer.sync()
        self.last_viewer_sync_time = (
            time.monotonic()
        )

    def update_viewer(
        self,
        force: bool = False,
    ) -> bool:
        if self.viewer is None:
            return True

        if not self.viewer.is_running():
            return False

        now = time.monotonic()

        if (
            force
            or now - self.last_viewer_sync_time
            >= self.viewer_period
        ):
            self.viewer.sync()
            self.last_viewer_sync_time = now

        return True

    def build_sensor_message(self) -> dict:
        position_world = self._world_position()
        velocity_world = self._world_velocity()

        position_ned = (
            AXIS_FLIP
            @ (
                position_world
                - self.origin_world
            )
        )

        velocity_ned = (
            AXIS_FLIP @ velocity_world
        )

        rotation_nwu_flu = (
            self._rotation_nwu_flu()
        )

        rotation_ned_frd = (
            AXIS_FLIP
            @ rotation_nwu_flu
            @ AXIS_FLIP
        )

        attitude = rotation_matrix_to_euler(
            rotation_ned_frd
        )

        gyro_frd = (
            AXIS_FLIP @ self._gyro_flu()
        )

        accel_frd = (
            AXIS_FLIP
            @ self._accelerometer_flu()
        )

        velocity_body_flu = (
            rotation_nwu_flu.T
            @ velocity_world
        )

        airspeed = max(
            float(velocity_body_flu[0]),
            0.0,
        )

        height_agl = max(
            float(
                self.data.site_xpos[
                    self.rangefinder_site_id,
                    2,
                ]
            ),
            0.0,
        )

        command = self.controller.last_command

        estimated_current = (
            2.0
            + 70.0
            * (
                command.throttle_left
                + command.throttle_right
            )
        )

        return {
            "timestamp": float(self.data.time),
            "imu": {
                "gyro": gyro_frd.tolist(),
                "accel_body": accel_frd.tolist(),
            },
            "position": position_ned.tolist(),
            "velocity": velocity_ned.tolist(),
            "attitude": attitude.tolist(),
            "airspeed": airspeed,
            "rng_1": height_agl,
            "velocity_wind": [0.0, 0.0, 0.0],
            "battery": {
                "voltage": 22.2,
                "current": estimated_current,
            },
        }

    def send_sensor_message(
        self,
        target: tuple[str, int],
    ) -> None:
        json_bytes = json.dumps(
            self.build_sensor_message(),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")

        self.socket.sendto(
            b"\n" + json_bytes + b"\n",
            target,
        )

    def print_status(
        self,
        packet: ServoPacket,
        sender: tuple[str, int],
    ) -> None:
        now = time.monotonic()

        if (
            now - self.last_status_wall_time
            < 1.0
        ):
            return

        self.last_status_wall_time = now

        position = self._world_position()
        vertical_speed = float(
            self._world_velocity()[2]
        )

        attitude = rotation_matrix_to_euler(
            AXIS_FLIP
            @ self._rotation_nwu_flu()
            @ AXIS_FLIP
        )

        gyro_frd = (
            AXIS_FLIP @ self._gyro_flu()
        )

        yaw_rate_deg_s = float(
            np.rad2deg(gyro_frd[2])
        )

        command = self.controller.last_command

        raw_pwm = tuple(
            int(value)
            for value in packet.pwm[:8]
        )

        print(
            f"SITL={sender[0]}:{sender[1]} | "
            f"frame={packet.frame_count} | "
            f"rate={packet.frame_rate} Hz | "
            f"sim={self.data.time:8.3f} s | "
            f"z={position[2]:6.2f} m | "
            f"RPY={np.rad2deg(attitude).round(1)} deg | "
            f"vz={vertical_speed:6.2f} m/s | "
            f"yaw_rate={yaw_rate_deg_s:7.1f} deg/s | "
            f"pwm={raw_pwm} | "
            f"thr=({command.throttle_left:.2f},"
            f"{command.throttle_right:.2f}) | "
            f"vane=("
            f"{np.rad2deg(command.vane_left_rad):.1f},"
            f"{np.rad2deg(command.vane_right_rad):.1f}"
            f") deg"
        )

    def _handle_timeout(self) -> bool:
        if not self.update_viewer():
            return False

        if (
            time.monotonic()
            - self.last_packet_wall_time
            > 2.0
        ):
            if not self.wait_message_printed:
                print(
                    "Waiting for ArduPilot "
                    "JSON SITL on "
                    f"{self.bind_address}:"
                    f"{self.port}"
                )
                self.wait_message_printed = True

        return True

    def run_loop(self) -> None:
        print(
            f"MuJoCo loaded: {self.xml_path}"
        )
        print(
            "Waiting for ArduPilot JSON SITL on "
            f"{self.bind_address}:{self.port}"
        )

        while True:
            if not self.update_viewer():
                print("MuJoCo viewer closed")
                break

            try:
                raw_packet, sender = (
                    self.socket.recvfrom(2048)
                )

            except socket.timeout:
                if not self._handle_timeout():
                    break
                continue

            try:
                packet = parse_servo_packet(
                    raw_packet
                )

            except ValueError as error:
                print(
                    "Ignoring invalid UDP packet: "
                    f"{error}"
                )
                continue

            self.last_packet_wall_time = (
                time.monotonic()
            )
            self.wait_message_printed = False

            advanced = self.advance_physics(
                pwm=packet.pwm,
                frame_count=packet.frame_count,
                frame_rate=packet.frame_rate,
            )

            if not advanced:
                continue

            self.send_sensor_message(sender)
            self.print_status(packet, sender)

            if not self.update_viewer():
                print("MuJoCo viewer closed")
                break

    def run(self) -> None:
        try:
            if self.show_viewer:
                self.start_viewer()

            self.run_loop()

        except KeyboardInterrupt:
            print(
                "\nCtrl+C received; "
                "simulation closed"
            )

        finally:
            self.controller.stop()

            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None

            self.socket.close()


def parse_arguments() -> argparse.Namespace:
    default_xml = (
        Path(__file__).resolve().parent
        / "uav.xml"
    )

    parser = argparse.ArgumentParser(
        description=(
            "ArduPlane JSON SITL to "
            "SolidWorks MuJoCo bridge"
        )
    )

    parser.add_argument(
        "--xml",
        type=Path,
        default=default_xml,
    )
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=ARDUPILOT_JSON_PORT,
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    xml_path = (
        args.xml.expanduser().resolve()
    )

    if not xml_path.exists():
        raise FileNotFoundError(
            f"MuJoCo XML not found: {xml_path}"
        )

    bridge = MuJoCoArduPilotBridge(
        xml_path=xml_path,
        bind_address=args.bind,
        port=args.port,
        show_viewer=not args.no_viewer,
    )

    bridge.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())