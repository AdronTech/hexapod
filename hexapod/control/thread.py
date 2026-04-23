"""
Control thread: reads gamepad state, runs IK + gait, drives servos.
"""

import math
import threading
import time
from dataclasses import replace
from typing import Optional

from hexapod.body_ik import BodyPose, body_pose_ik, neutral_foot_body
from hexapod.control.state import (
    SharedState,
    GAITS,
    STAND_HEIGHT,
    STAND_SPEED,
    DT,
    HEIGHT_MIN,
    HEIGHT_MAX,
    REACH_MIN,
    REACH_MAX,
    REACH_RATE_CMS,
    FREE_STEP_EMERGENCY,
    STORAGE_FEMUR_DEG,
    STORAGE_TIBIA_DEG,
    DPAD_CM_RATE,
    DPAD_DEG_RATE,
)
from hexapod.gait import FreeGait, RippleGait, TripodGait, WaveGait, _NEUTRAL_REACH
from hexapod.kinematics import COXA_LEN, FEMUR_LEN, IKError, angle_to_tick
from hexapod.robot.config import Joint, Leg, servo_id
from hexapod.robot.soft_limits import SoftLimitError, SoftLimits
from hexapod.servo.motion import MotionPlayer
from hexapod.servo.protocol import ProtocolError  # noqa: F401 — may propagate
from hexapod.servo.st3020 import PositionCommand, ST3020Bus
from hexapod.servo.transport import SerialTransport, TransportError

# ---------------------------------------------------------------------------
# Gamepad button / axis indices (Standard Gamepad API)
# ---------------------------------------------------------------------------

BTN_A      = 0
BTN_B      = 1
BTN_X      = 2
BTN_Y      = 3
BTN_LB     = 4
BTN_RB     = 5
BTN_LT     = 6
BTN_RT     = 7
BTN_BACK   = 8
BTN_START  = 9
BTN_DUP    = 12
BTN_DDOWN  = 13
BTN_DLEFT  = 14
BTN_DRIGHT = 15

AX_LSX, AX_LSY = 0, 1
AX_RSX, AX_RSY = 2, 3


def _dead(v: float, deadzone: float = 0.12) -> float:
    """Deadzone + rescale so that 0 stays 0 and ±1 stays ±1."""
    if abs(v) < deadzone:
        return 0.0
    s = 1.0 if v > 0 else -1.0
    return s * (abs(v) - deadzone) / (1.0 - deadzone)


# ---------------------------------------------------------------------------
# Control thread
# ---------------------------------------------------------------------------

class ControlThread(threading.Thread):

    def __init__(self, serial_port: str, shared: SharedState) -> None:
        super().__init__(daemon=True, name="control")
        self._port   = serial_port
        self._shared = shared
        self._stop   = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self._shared.set_status(False, False, {}, "Opening serial port…")
        try:
            with SerialTransport(self._port) as transport:
                bus = ST3020Bus(transport)
                limits = SoftLimits.load()
                self._shared.set_status(False, False, {}, "Ready — press A to stand")
                self._loop(bus, limits)
        except (TransportError, OSError) as e:
            self._shared.set_status(False, False, {}, f"Serial error: {e}")

    def _loop(self, bus: ST3020Bus, limits: Optional[SoftLimits]) -> None:
        pose:  Optional[BodyPose]                              = None
        feet:  Optional[dict[Leg, tuple[float, float, float]]] = None
        gait   = None
        standing   = False
        walk_mode  = False
        free_mode  = False
        prev_btns  = [0.0] * 17
        active_gait_type = "tripod"

        while not self._stop.is_set():
            t0 = time.monotonic()
            axes, buttons, gp_on = self._shared.get_gamepad()
            speed_cm, speed_deg  = self._shared.get_speeds()
            step_height, step_time, step_threshold = self._shared.get_step_params()
            soft_limit_margin_deg = self._shared.get_soft_limit_margin_deg()

            n = max(len(buttons), 17)
            buttons = (buttons + [0.0] * n)[:n]
            pressed = [
                buttons[i] > 0.5 and prev_btns[i] <= 0.5
                for i in range(n)
            ]
            prev_btns = list(buttons)

            pending_cmd = self._shared.pop_command()

            if pending_cmd == "store" and not self._stop.is_set():
                self._shared.set_status(standing, True, self._pose_dict(pose), "Storing…")
                try:
                    self._do_store(bus, limits)
                    self._shared.set_stored()
                except Exception as e:
                    self._shared.set_status(False, False, {}, f"Store failed: {e}")
                pose = None; feet = None; gait = None
                standing = False; walk_mode = False; free_mode = False

            if gp_on and not self._stop.is_set():
                if pressed[BTN_Y]:
                    self._shared.set_status(standing, True, self._pose_dict(pose), "Storing…")
                    try:
                        self._do_store(bus, limits)
                        self._shared.set_stored()
                    except Exception as e:
                        self._shared.set_status(False, False, {}, f"Store failed: {e}")
                    pose = None; feet = None; gait = None
                    standing = False; walk_mode = False; free_mode = False
                    continue

                if buttons[BTN_DUP] > 0.5:
                    self._shared.set_speeds(speed_cm + DPAD_CM_RATE * DT, speed_deg)
                elif buttons[BTN_DDOWN] > 0.5:
                    self._shared.set_speeds(speed_cm - DPAD_CM_RATE * DT, speed_deg)
                if buttons[BTN_DRIGHT] > 0.5:
                    self._shared.set_speeds(speed_cm, speed_deg + DPAD_DEG_RATE * DT)
                elif buttons[BTN_DLEFT] > 0.5:
                    self._shared.set_speeds(speed_cm, speed_deg - DPAD_DEG_RATE * DT)

                if pressed[BTN_A] and not standing:
                    self._shared.set_status(False, True, {}, "Standing up…")
                    try:
                        result = self._do_stand(bus, limits)
                        pose, feet = result
                        standing = True
                        self._shared.set_status(True, False, self._pose_dict(pose), "Standing")
                    except Exception as e:
                        self._shared.set_status(False, False, {}, f"Stand failed: {e}")

                elif pressed[BTN_B] and standing:
                    self._shared.set_status(True, True, self._pose_dict(pose), "Sitting down…")
                    try:
                        self._do_sit(bus)
                    except Exception:
                        pass
                    pose = None; feet = None; gait = None
                    standing = False; walk_mode = False; free_mode = False
                    self._shared.set_status(False, False, {}, "Sitting — press A to stand")

                elif pressed[BTN_BACK] and standing:
                    if walk_mode:
                        idx = GAITS.index(active_gait_type)
                        active_gait_type = GAITS[(idx + 1) % len(GAITS)]
                        self._shared.set_gait_type(active_gait_type)
                        if gait is not None:
                            snapped = {leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()}
                            gait = self._make_gait(active_gait_type, gait.body, snapped, step_height, step_time)
                    elif free_mode:
                        if gait is not None:
                            pose = replace(gait.body, roll=0.0, pitch=0.0)
                            feet = {leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()}
                            try:
                                ticks = self._compute_ticks(pose, feet, limits)
                                self._apply_ticks(bus, ticks)
                            except (IKError, SoftLimitError):
                                pass
                        gait = None
                        free_mode = False
                        self._shared.set_status(True, False, self._pose_dict(pose))
                    elif pose is not None and feet is not None:
                        snapped = {leg: (f[0], f[1], 0.0) for leg, f in feet.items()}
                        gait = FreeGait(
                            replace(pose, roll=0.0, pitch=0.0), snapped,
                            neutral_reach=self._shared.get_reach(),
                            step_height=step_height,
                            step_time=step_time,
                            step_threshold=step_threshold,
                            step_emergency_threshold=FREE_STEP_EMERGENCY,
                            step_reach_max=REACH_MAX,
                            step_reach_min=REACH_MIN,
                            soft_limits=limits,
                            soft_limit_margin_deg=soft_limit_margin_deg,
                        )
                        free_mode = True
                        self._shared.set_status(
                            True, False, self._pose_dict(gait.body), "Free", free_mode=True
                        )

                elif pressed[BTN_X] and standing and not free_mode and pose is not None and feet is not None:
                    walk_mode = not walk_mode
                    if walk_mode:
                        active_gait_type = self._shared.get_gait_type()
                        snapped = {leg: (f[0], f[1], 0.0) for leg, f in feet.items()}
                        gait = self._make_gait(active_gait_type, pose, snapped, step_height, step_time)
                    else:
                        if gait is not None:
                            pose = gait.body
                            feet = {leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()}
                            try:
                                ticks = self._compute_ticks(pose, feet, limits)
                                self._apply_ticks(bus, ticks)
                            except (IKError, SoftLimitError):
                                pass
                        gait = None

                elif pressed[BTN_START] and standing and feet is not None:
                    walk_mode = False; free_mode = False; gait = None
                    neutral = BodyPose(z=STAND_HEIGHT)
                    neutral_feet = self._neutral_feet()
                    try:
                        ticks = self._compute_ticks(neutral, neutral_feet, limits)
                        self._apply_ticks(bus, ticks)
                        pose = neutral
                        feet = neutral_feet
                        self._shared.set_status(True, False, self._pose_dict(pose), "Pose reset")
                    except (IKError, SoftLimitError):
                        pass

                elif standing and free_mode and gait is not None:
                    yaw_rad   = math.radians(gait.body.yaw)
                    body_vx   = -_dead(axes[AX_LSY]) * speed_cm
                    body_vy   = -_dead(axes[AX_LSX]) * speed_cm
                    vx = body_vx * math.cos(yaw_rad) - body_vy * math.sin(yaw_rad)
                    vy = body_vx * math.sin(yaw_rad) + body_vy * math.cos(yaw_rad)

                    lt = _dead(buttons[BTN_LT])
                    rt = _dead(buttons[BTN_RT])
                    dz = (rt - lt) * speed_cm * DT
                    if abs(dz) > 1e-9:
                        gait.body_z = max(HEIGHT_MIN, min(HEIGHT_MAX, gait.body_z + dz))

                    lb = 1.0 if buttons[BTN_LB] > 0.5 else 0.0
                    rb = 1.0 if buttons[BTN_RB] > 0.5 else 0.0
                    omega = (lb - rb) * speed_deg
                    droll = _dead(axes[AX_RSX]) * speed_deg * DT
                    if abs(droll) > 1e-9:
                        gait.body_roll = max(-30.0, min(30.0, gait.body_roll + droll))

                    dpitch = -_dead(axes[AX_RSY]) * speed_deg * DT
                    if abs(dpitch) > 1e-9:
                        gait.body_pitch = max(-30.0, min(30.0, gait.body_pitch + dpitch))

                    gait.neutral_reach  = self._shared.get_reach()
                    gait.step_height    = step_height
                    gait.step_time      = step_time
                    gait.step_threshold = step_threshold

                    new_pose, new_feet = gait.step(vx, vy, omega, DT)
                    try:
                        ticks = self._compute_ticks(new_pose, new_feet, limits)
                        self._apply_ticks(bus, ticks)
                        pose = new_pose
                        feet = new_feet
                    except (IKError, SoftLimitError) as e:
                        self._shared.bump_ik_errors(str(e))
                    self._shared.set_status(
                        True, False, self._pose_dict(pose), "Free", free_mode=True
                    )

                elif standing and walk_mode and gait is not None:
                    desired_gait = self._shared.get_gait_type()
                    if desired_gait != active_gait_type:
                        active_gait_type = desired_gait
                        snapped = {leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()}
                        gait = self._make_gait(active_gait_type, gait.body, snapped, step_height, step_time)

                    yaw_rad  = math.radians(gait.body.yaw)
                    body_vx  = -_dead(axes[AX_LSY]) * speed_cm
                    body_vy  = -_dead(axes[AX_LSX]) * speed_cm
                    omega    = -_dead(axes[AX_RSX]) * speed_deg
                    vx = body_vx * math.cos(yaw_rad) - body_vy * math.sin(yaw_rad)
                    vy = body_vx * math.sin(yaw_rad) + body_vy * math.cos(yaw_rad)

                    lt = _dead(buttons[BTN_LT])
                    rt = _dead(buttons[BTN_RT])
                    dz = (rt - lt) * speed_cm * DT
                    if abs(dz) > 1e-9:
                        gait.body_z = max(HEIGHT_MIN, min(HEIGHT_MAX, gait.body_z + dz))

                    lb = 1.0 if buttons[BTN_LB] > 0.5 else 0.0
                    rb = 1.0 if buttons[BTN_RB] > 0.5 else 0.0
                    if lb or rb:
                        new_reach = self._shared.get_reach() + (rb - lb) * REACH_RATE_CMS * DT
                        self._shared.set_reach(new_reach)
                    gait.neutral_reach = self._shared.get_reach()
                    gait.step_height   = step_height
                    gait.step_time     = step_time

                    new_pose, new_feet = gait.step(vx, vy, omega, DT)
                    try:
                        ticks = self._compute_ticks(new_pose, new_feet, limits)
                        self._apply_ticks(bus, ticks)
                        pose = new_pose
                        feet = new_feet
                    except (IKError, SoftLimitError) as e:
                        self._shared.bump_ik_errors(str(e))
                        gait.body = pose
                    self._shared.set_status(
                        True, False, self._pose_dict(pose), "Walking", walk_mode=True
                    )

                elif standing and not walk_mode and not free_mode and pose is not None and feet is not None:
                    yaw_rad   = math.radians(pose.yaw)
                    body_dx   = -_dead(axes[AX_LSY]) * speed_cm * DT
                    body_dy   = -_dead(axes[AX_LSX]) * speed_cm * DT
                    dx = body_dx * math.cos(yaw_rad) - body_dy * math.sin(yaw_rad)
                    dy = body_dx * math.sin(yaw_rad) + body_dy * math.cos(yaw_rad)
                    lt     =  _dead(buttons[BTN_LT])
                    rt     =  _dead(buttons[BTN_RT])
                    dz     = (rt - lt) * speed_cm * DT
                    lb     = 1.0 if buttons[BTN_LB] > 0.5 else 0.0
                    rb     = 1.0 if buttons[BTN_RB] > 0.5 else 0.0
                    droll  = _dead(axes[AX_RSX]) * speed_deg * DT
                    dpitch = -_dead(axes[AX_RSY]) * speed_deg * DT
                    dyaw   = (lb - rb) * speed_deg * DT

                    if abs(dx)+abs(dy)+abs(dz)+abs(droll)+abs(dpitch)+abs(dyaw) > 1e-9:
                        new_pose = replace(
                            pose,
                            x=pose.x + dx,  y=pose.y + dy,  z=pose.z + dz,
                            roll=pose.roll + droll,
                            pitch=pose.pitch + dpitch,
                            yaw=pose.yaw + dyaw,
                        )
                        try:
                            ticks = self._compute_ticks(new_pose, feet, limits)
                            self._apply_ticks(bus, ticks)
                            pose = new_pose
                        except (IKError, SoftLimitError) as e:
                            self._shared.bump_ik_errors(str(e))
                    self._shared.set_status(True, False, self._pose_dict(pose))

            elapsed = time.monotonic() - t0
            rem = DT - elapsed
            if rem > 0:
                time.sleep(rem)

    # --- helpers ---

    @staticmethod
    def _make_gait(gait_type: str, pose: BodyPose, feet: dict,
                   step_height: float = 4.0, step_time: float = 0.40):
        kw = dict(neutral_reach=_NEUTRAL_REACH, step_height=step_height, step_time=step_time)
        if gait_type == "ripple":
            return RippleGait(pose, feet, **kw)
        if gait_type == "wave":
            return WaveGait(pose, feet, **kw)
        return TripodGait(pose, feet, **kw)

    @staticmethod
    def _neutral_feet() -> dict[Leg, tuple[float, float, float]]:
        return {
            leg: (neutral_foot_body(leg)[0], neutral_foot_body(leg)[1], 0.0)
            for leg in Leg
        }

    @staticmethod
    def _compute_ticks(
        pose:   BodyPose,
        feet:   dict[Leg, tuple[float, float, float]],
        limits: Optional[SoftLimits],
    ) -> dict[Leg, dict[Joint, int]]:
        angles = body_pose_ik(pose, feet)
        ticks: dict[Leg, dict[Joint, int]] = {}
        for leg, (tc, tf, tt) in angles.items():
            if limits:
                try:
                    limits.check(tc, tf, tt)
                except SoftLimitError as e:
                    raise SoftLimitError(f"{leg.name}: {e}") from e
            ticks[leg] = {
                Joint.COXA:  angle_to_tick("coxa",  tc),
                Joint.FEMUR: angle_to_tick("femur", tf),
                Joint.TIBIA: angle_to_tick("tibia", tt),
            }
        return ticks

    @staticmethod
    def _apply_ticks(bus: ST3020Bus, ticks: dict[Leg, dict[Joint, int]]) -> None:
        cmds = [
            PositionCommand(servo_id(leg, joint), ticks[leg][joint], speed=0, acc=0)
            for leg in Leg
            for joint in Joint
        ]
        bus.sync_write_position(cmds)

    def _do_stand(
        self,
        bus:    ST3020Bus,
        limits: Optional[SoftLimits],
    ) -> tuple[BodyPose, dict[Leg, tuple[float, float, float]]]:
        pose = BodyPose(z=STAND_HEIGHT)
        feet = self._neutral_feet()
        ticks = self._compute_ticks(pose, feet, limits)
        targets = [
            (servo_id(leg, joint), ticks[leg][joint], STAND_SPEED)
            for leg in Leg
            for joint in Joint
        ]
        MotionPlayer(bus, acc=0).move(targets)
        return pose, feet

    @staticmethod
    def _do_sit(bus: ST3020Bus) -> None:
        targets = [
            (servo_id(leg, joint), 2048, STAND_SPEED)
            for leg in Leg
            for joint in Joint
        ]
        MotionPlayer(bus, acc=0).move(targets)
        for leg in Leg:
            for joint in Joint:
                bus.torque_enable(servo_id(leg, joint), False)

    @staticmethod
    def _do_store(bus: ST3020Bus, limits: Optional[SoftLimits]) -> None:
        for leg in Leg:
            for joint in Joint:
                bus.torque_enable(servo_id(leg, joint), True)

        femur_up   = limits.femur.max_deg if limits else STORAGE_FEMUR_DEG
        tibia_down = limits.tibia.min_deg if limits else STORAGE_TIBIA_DEG

        player = MotionPlayer(bus, acc=0)
        player.move([
            (servo_id(leg, joint), 2048, STAND_SPEED)
            for leg in Leg
            for joint in Joint
        ])

        femur_tick = angle_to_tick("femur", femur_up)
        tibia_tick = angle_to_tick("tibia", tibia_down)
        player.move([
            (servo_id(leg, joint), tick, STAND_SPEED)
            for leg in Leg
            for joint, tick in ((Joint.FEMUR, femur_tick), (Joint.TIBIA, tibia_tick))
        ])

        for leg in Leg:
            for joint in Joint:
                bus.torque_enable(servo_id(leg, joint), False)

    @staticmethod
    def _pose_dict(pose: Optional[BodyPose]) -> dict:
        if pose is None:
            return {}
        return {
            "x":     round(pose.x,     2),
            "y":     round(pose.y,     2),
            "z":     round(pose.z,     2),
            "roll":  round(pose.roll,  2),
            "pitch": round(pose.pitch, 2),
            "yaw":   round(pose.yaw,   2),
        }
