from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np


PWM_MIN = 1000
PWM_TRIM = 1500
PWM_MAX = 2000

# 启动时必须先看到一次左右电机低油门
MOTOR_SAFE_PWM = 1100


@dataclass(frozen=True)
class ChannelMap:
    """ArduPilot JSON舵机输出包中的零基索引。"""

    # SERVO1_FUNCTION=73
    throttle_left: int = 0

    # SERVO2_FUNCTION=74
    throttle_right: int = 1

    # SERVO3_FUNCTION=75
    vane_left: int = 2

    # SERVO4_FUNCTION=76
    vane_right: int = 3


@dataclass
class DuctedFanCommand:
    throttle_left: float
    throttle_right: float
    vane_left_rad: float
    vane_right_rad: float
    thrust_left_n: float
    thrust_right_n: float


def pwm_to_throttle(pwm: int) -> float:
    return float(
        np.clip(
            (float(pwm) - PWM_MIN) / (PWM_MAX - PWM_MIN),
            0.0,
            1.0,
        )
    )


def pwm_to_signed(pwm: int) -> float:
    return float(
        np.clip(
            (float(pwm) - PWM_TRIM) / (PWM_MAX - PWM_TRIM),
            -1.0,
            1.0,
        )
    )


def first_order_filter(
    current: float,
    target: float,
    dt: float,
    tau: float,
) -> float:
    if tau <= 0.0:
        return target

    dt = max(float(dt), 0.0)
    alpha = dt / (float(tau) + dt)
    return current + alpha * (target - current)


def mix_vane_commands(
    left_command: float,
    right_command: float,
    yaw_differential_sign: float = -1.0,
) -> tuple[float, float]:
    """
    保留同向俯仰分量，只反转差动偏航分量。

    如果实际舵面偏航方向相反，只需要把
    yaw_differential_sign从-1改为1。
    """
    common = 0.5 * (left_command + right_command)
    differential = 0.5 * (left_command - right_command)

    physical_left = (
        common + yaw_differential_sign * differential
    )
    physical_right = (
        common - yaw_differential_sign * differential
    )

    return physical_left, physical_right


def require_mujoco_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise ValueError(
            f"MuJoCo model is missing required object: {name}"
        )

    return int(object_id)


class DuctedFanController:
    """把ArduPilot PWM输出转换为新SolidWorks模型的执行器控制量。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        channel_map: ChannelMap | None = None,
        max_thrust_per_duct: float = 140.09,
        max_vane_angle_deg: float = 10.0,
        motor_time_constant: float = 0.08,
        servo_time_constant: float = 0.10,
        throttle_exponent: float = 1.0,
        vane_force_efficiency: float = 0.55,
        vertical_deflection_loss: float = 0.35,
        yaw_differential_sign: float = -1.0,
        max_blade_speed_rad_s: float = 1250.0,
    ) -> None:
        self.model = model
        self.data = data
        self.channels = channel_map or ChannelMap()

        self.max_thrust = float(max_thrust_per_duct)
        self.max_vane_angle = float(
            np.deg2rad(max_vane_angle_deg)
        )

        self.motor_time_constant = float(
            motor_time_constant
        )
        self.servo_time_constant = float(
            servo_time_constant
        )
        self.throttle_exponent = float(
            throttle_exponent
        )
        self.vane_force_efficiency = float(
            vane_force_efficiency
        )
        self.vertical_deflection_loss = float(
            vertical_deflection_loss
        )
        self.yaw_differential_sign = float(
            yaw_differential_sign
        )
        self.max_blade_speed = float(
            max_blade_speed_rad_s
        )

        self.body_id = require_mujoco_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "uav",
        )

        self.actuator_ids = {
            "left_vertical": require_mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "blade_left_thrust",
            ),
            "right_vertical": require_mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "blade_right_thrust",
            ),
            "left_forward": require_mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "rudder_left_forward_force",
            ),
            "right_forward": require_mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "rudder_right_forward_force",
            ),
            "left_vane": require_mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "rudder_left_servo",
            ),
            "right_vane": require_mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "rudder_right_servo",
            ),
            "left_blade": require_mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "blade_left_velocity",
            ),
            "right_blade": require_mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "blade_right_velocity",
            ),
        }

        self.body_velocity_world = np.zeros(
            6,
            dtype=float,
        )

        # 针对约20 kg机体的初始阻尼，可后续通过飞行数据调整
        self.linear_drag = np.array(
            [1.20, 1.60, 2.00],
            dtype=float,
        )
        self.angular_drag_linear = np.array(
            [0.55, 0.70, 0.85],
            dtype=float,
        )
        self.angular_drag_quadratic = np.array(
            [0.06, 0.08, 0.10],
            dtype=float,
        )

        self.reset()

    def reset(self) -> None:
        self.motor_low_seen = False

        self.throttle_left = 0.0
        self.throttle_right = 0.0

        self.commanded_vane_left = 0.0
        self.commanded_vane_right = 0.0

        self.physical_vane_left = 0.0
        self.physical_vane_right = 0.0

        self.last_command = DuctedFanCommand(
            throttle_left=0.0,
            throttle_right=0.0,
            vane_left_rad=0.0,
            vane_right_rad=0.0,
            thrust_left_n=0.0,
            thrust_right_n=0.0,
        )

        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[
            self.body_id, :
        ] = 0.0

    @staticmethod
    def _channel(
        pwm_values: Sequence[int],
        index: int,
        default: int,
    ) -> int:
        if index < 0 or index >= len(pwm_values):
            return default

        value = int(pwm_values[index])

        if 800 <= value <= 2200:
            return value

        return default

    def decode_pwm(
        self,
        pwm_values: Sequence[int],
    ) -> tuple[float, float, float, float]:
        left_motor_pwm = self._channel(
            pwm_values,
            self.channels.throttle_left,
            PWM_MIN,
        )
        right_motor_pwm = self._channel(
            pwm_values,
            self.channels.throttle_right,
            PWM_MIN,
        )

        # 防止错误SERVO_FUNCTION使1500 PWM被当作50%推力
        if not self.motor_low_seen:
            if (
                left_motor_pwm <= MOTOR_SAFE_PWM
                and right_motor_pwm <= MOTOR_SAFE_PWM
            ):
                self.motor_low_seen = True
            else:
                return 0.0, 0.0, 0.0, 0.0

        left_throttle = pwm_to_throttle(
            left_motor_pwm
        )
        right_throttle = pwm_to_throttle(
            right_motor_pwm
        )

        left_vane = (
            pwm_to_signed(
                self._channel(
                    pwm_values,
                    self.channels.vane_left,
                    PWM_TRIM,
                )
            )
            * self.max_vane_angle
        )

        right_vane = (
            pwm_to_signed(
                self._channel(
                    pwm_values,
                    self.channels.vane_right,
                    PWM_TRIM,
                )
            )
            * self.max_vane_angle
        )

        return (
            left_throttle,
            right_throttle,
            left_vane,
            right_vane,
        )

    def update(
        self,
        pwm_values: Sequence[int],
        dt: float,
    ) -> DuctedFanCommand:
        (
            target_left_throttle,
            target_right_throttle,
            target_left_vane,
            target_right_vane,
        ) = self.decode_pwm(pwm_values)

        self.throttle_left = first_order_filter(
            self.throttle_left,
            target_left_throttle,
            dt,
            self.motor_time_constant,
        )
        self.throttle_right = first_order_filter(
            self.throttle_right,
            target_right_throttle,
            dt,
            self.motor_time_constant,
        )

        self.commanded_vane_left = first_order_filter(
            self.commanded_vane_left,
            target_left_vane,
            dt,
            self.servo_time_constant,
        )
        self.commanded_vane_right = first_order_filter(
            self.commanded_vane_right,
            target_right_vane,
            dt,
            self.servo_time_constant,
        )

        (
            self.physical_vane_left,
            self.physical_vane_right,
        ) = mix_vane_commands(
            self.commanded_vane_left,
            self.commanded_vane_right,
            self.yaw_differential_sign,
        )

        thrust_left = (
            self.max_thrust
            * self.throttle_left
            ** self.throttle_exponent
        )
        thrust_right = (
            self.max_thrust
            * self.throttle_right
            ** self.throttle_exponent
        )

        left_loss = (
            self.vertical_deflection_loss
            * (
                1.0
                - np.cos(self.physical_vane_left)
            )
        )
        right_loss = (
            self.vertical_deflection_loss
            * (
                1.0
                - np.cos(self.physical_vane_right)
            )
        )

        left_vertical = (
            thrust_left * (1.0 - left_loss)
        )
        right_vertical = (
            thrust_right * (1.0 - right_loss)
        )

        left_forward = (
            self.vane_force_efficiency
            * thrust_left
            * np.sin(self.physical_vane_left)
        )
        right_forward = (
            self.vane_force_efficiency
            * thrust_right
            * np.sin(self.physical_vane_right)
        )

        # 左右叶片反向旋转，尽量抵消动画带来的反作用力矩
        left_blade_speed = (
            self.max_blade_speed
            * np.sqrt(self.throttle_left)
        )
        right_blade_speed = (
            -self.max_blade_speed
            * np.sqrt(self.throttle_right)
        )

        ctrl = self.data.ctrl

        ctrl[
            self.actuator_ids["left_vertical"]
        ] = left_vertical

        ctrl[
            self.actuator_ids["right_vertical"]
        ] = right_vertical

        ctrl[
            self.actuator_ids["left_forward"]
        ] = left_forward

        ctrl[
            self.actuator_ids["right_forward"]
        ] = right_forward

        ctrl[
            self.actuator_ids["left_vane"]
        ] = self.physical_vane_left

        ctrl[
            self.actuator_ids["right_vane"]
        ] = self.physical_vane_right

        ctrl[
            self.actuator_ids["left_blade"]
        ] = left_blade_speed

        ctrl[
            self.actuator_ids["right_blade"]
        ] = right_blade_speed

        self._apply_aerodynamic_drag()

        self.last_command = DuctedFanCommand(
            throttle_left=self.throttle_left,
            throttle_right=self.throttle_right,
            vane_left_rad=self.physical_vane_left,
            vane_right_rad=self.physical_vane_right,
            thrust_left_n=thrust_left,
            thrust_right_n=thrust_right,
        )

        return self.last_command

    def _apply_aerodynamic_drag(self) -> None:
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.body_id,
            self.body_velocity_world,
            0,
        )

        angular_velocity = (
            self.body_velocity_world[:3]
        )
        linear_velocity = (
            self.body_velocity_world[3:]
        )

        drag_force = (
            -self.linear_drag
            * linear_velocity
            * np.abs(linear_velocity)
        )

        drag_torque = (
            -self.angular_drag_linear
            * angular_velocity
            - self.angular_drag_quadratic
            * angular_velocity
            * np.abs(angular_velocity)
        )

        self.data.xfrc_applied[
            self.body_id, :
        ] = 0.0

        self.data.xfrc_applied[
            self.body_id, :3
        ] = drag_force

        self.data.xfrc_applied[
            self.body_id, 3:
        ] = drag_torque

    def stop(self) -> None:
        self.reset()