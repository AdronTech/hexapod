"""
Control thread: reads gamepad state, runs IK + gait, drives servos.
"""

import math
import threading
import time
from dataclasses import replace

from hexapod.body_ik import BodyPose, body_pose_ik, neutral_foot_body
from hexapod.control.recorder import Recorder
from hexapod.control.state import (
    DPAD_CM_RATE,
    DPAD_DEG_RATE,
    DT,
    FREE_STEP_EMERGENCY,
    GAITS,
    HEIGHT_MAX,
    HEIGHT_MIN,
    REACH_MAX,
    REACH_MIN,
    REACH_RATE_CMS,
    STAND_HEIGHT,
    STAND_SPEED,
    STORAGE_FEMUR_DEG,
    STORAGE_TIBIA_DEG,
    SharedState,
)
from hexapod.gait import _NEUTRAL_REACH, FreeGait, RippleGait, TripodGait, WaveGait
from hexapod.kinematics import IKError, angle_to_tick
from hexapod.robot.config import Joint, Leg, servo_id
from hexapod.robot.soft_limits import SoftLimitError, SoftLimits
from hexapod.servo.motion import MotionPlayer
from hexapod.servo.protocol import ProtocolError
from hexapod.servo.st3020 import PositionCommand, ST3020Bus
from hexapod.servo.transport import SerialTransport, TransportError

# ---------------------------------------------------------------------------
# Gamepad button / axis indices (Standard Gamepad API)
# ---------------------------------------------------------------------------

BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5
BTN_LT = 6
BTN_RT = 7
BTN_BACK = 8
BTN_START = 9
BTN_DUP = 12
BTN_DDOWN = 13
BTN_DLEFT = 14
BTN_DRIGHT = 15

AX_LSX, AX_LSY = 0, 1
AX_RSX, AX_RSY = 2, 3

# Failures a servo/IK operation can raise — reported through the status line
# instead of killing the control thread.
MOTION_ERRORS = (TransportError, ProtocolError, IKError, SoftLimitError, OSError)


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
    def __init__(
        self,
        serial_port: str,
        shared: SharedState,
        recorder: Recorder | None = None,
    ) -> None:
        super().__init__(daemon=True, name="control")
        self._port = serial_port
        self._shared = shared
        self._rec = recorder
        self._stop = threading.Event()

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
        finally:
            if self._rec is not None:
                self._rec.close()

    def _loop(self, bus: ST3020Bus, limits: SoftLimits | None) -> None:
        pose: BodyPose | None = None
        feet: dict[Leg, tuple[float, float, float]] | None = None
        gait = None
        standing = False
        walk_mode = False
        free_mode = False
        prev_btns = [0.0] * 17
        active_gait_type = "tripod"

        while not self._stop.is_set():
            t0 = time.monotonic()
            rec_cmd: tuple[float, float, float] | None = None
            rec_ticks: dict[Leg, dict[Joint, int]] | None = None
            rec_err: str | None = None
            axes, buttons, gp_on = self._shared.get_gamepad()
            speed_cm, speed_deg = self._shared.get_speeds()
            step_height, step_time, step_threshold = self._shared.get_step_params()

            n = max(len(buttons), 17)
            buttons = (buttons + [0.0] * n)[:n]
            pressed = [buttons[i] > 0.5 and prev_btns[i] <= 0.5 for i in range(n)]
            prev_btns = list(buttons)

            pending_cmd = self._shared.pop_command()

            if pending_cmd == "mark":
                if self._rec is not None:
                    self._rec.mark()
                pending_cmd = None

            if pending_cmd == "store" and not self._stop.is_set():
                self._shared.set_status(
                    standing, True, self._pose_dict(pose), "Storing…"
                )
                try:
                    self._do_store(bus, limits)
                    self._shared.set_stored()
                except MOTION_ERRORS as e:
                    self._shared.set_status(False, False, {}, f"Store failed: {e}")
                pose = None
                feet = None
                gait = None
                standing = False
                walk_mode = False
                free_mode = False

            if gp_on and not self._stop.is_set():
                if pressed[BTN_Y]:
                    self._shared.set_status(
                        standing, True, self._pose_dict(pose), "Storing…"
                    )
                    try:
                        self._do_store(bus, limits)
                        self._shared.set_stored()
                    except MOTION_ERRORS as e:
                        self._shared.set_status(False, False, {}, f"Store failed: {e}")
                    pose = None
                    feet = None
                    gait = None
                    standing = False
                    walk_mode = False
                    free_mode = False
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
                        self._shared.set_status(
                            True, False, self._pose_dict(pose), "Standing"
                        )
                    except MOTION_ERRORS as e:
                        self._shared.set_status(False, False, {}, f"Stand failed: {e}")

                elif pressed[BTN_B] and standing:
                    self._shared.set_status(
                        True, True, self._pose_dict(pose), "Sitting down…"
                    )
                    try:
                        self._do_sit(bus)
                    except MOTION_ERRORS:
                        pass
                    pose = None
                    feet = None
                    gait = None
                    standing = False
                    walk_mode = False
                    free_mode = False
                    self._shared.set_status(
                        False, False, {}, "Sitting — press A to stand"
                    )

                elif pressed[BTN_BACK] and standing:
                    if walk_mode:
                        idx = GAITS.index(active_gait_type)
                        active_gait_type = GAITS[(idx + 1) % len(GAITS)]
                        self._shared.set_gait_type(active_gait_type)
                        if gait is not None:
                            snapped = {
                                leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()
                            }
                            gait = self._make_gait(
                                active_gait_type,
                                gait.body,
                                snapped,
                                step_height,
                                step_time,
                            )
                    elif free_mode:
                        if gait is not None:
                            pose = replace(gait.body, roll=0.0, pitch=0.0)
                            feet = {
                                leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()
                            }
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
                            replace(pose, roll=0.0, pitch=0.0),
                            snapped,
                            neutral_reach=self._shared.get_reach(),
                            step_height=step_height,
                            step_time=step_time,
                            step_threshold=step_threshold,
                            step_emergency_threshold=FREE_STEP_EMERGENCY,
                            step_reach_max=REACH_MAX,
                            step_reach_min=REACH_MIN,
                        )
                        free_mode = True
                        self._shared.set_status(
                            True,
                            False,
                            self._pose_dict(gait.body),
                            "Free",
                            free_mode=True,
                        )

                elif (
                    pressed[BTN_X]
                    and standing
                    and not free_mode
                    and pose is not None
                    and feet is not None
                ):
                    walk_mode = not walk_mode
                    if walk_mode:
                        active_gait_type = self._shared.get_gait_type()
                        snapped = {leg: (f[0], f[1], 0.0) for leg, f in feet.items()}
                        gait = self._make_gait(
                            active_gait_type, pose, snapped, step_height, step_time
                        )
                    else:
                        if gait is not None:
                            pose = gait.body
                            feet = {
                                leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()
                            }
                            try:
                                ticks = self._compute_ticks(pose, feet, limits)
                                self._apply_ticks(bus, ticks)
                            except (IKError, SoftLimitError):
                                pass
                        gait = None

                elif pressed[BTN_START] and standing and feet is not None:
                    walk_mode = False
                    free_mode = False
                    gait = None
                    neutral = BodyPose(z=STAND_HEIGHT)
                    neutral_feet = self._neutral_feet()
                    try:
                        ticks = self._compute_ticks(neutral, neutral_feet, limits)
                        self._apply_ticks(bus, ticks)
                        pose = neutral
                        feet = neutral_feet
                        self._shared.set_status(
                            True, False, self._pose_dict(pose), "Pose reset"
                        )
                    except (IKError, SoftLimitError):
                        pass

                elif standing and free_mode and gait is not None:
                    yaw_rad = math.radians(gait.body.yaw)
                    body_vx = -_dead(axes[AX_LSY]) * speed_cm
                    body_vy = -_dead(axes[AX_LSX]) * speed_cm
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
                        gait.body_pitch = max(
                            -30.0, min(30.0, gait.body_pitch + dpitch)
                        )

                    gait.neutral_reach = self._shared.get_reach()
                    gait.step_height = step_height
                    gait.step_time = step_time
                    gait.step_threshold = step_threshold

                    rec_cmd = (vx, vy, omega)
                    new_pose, new_feet = gait.step(vx, vy, omega, DT)
                    try:
                        ticks = self._compute_ticks(new_pose, new_feet, limits)
                        self._apply_ticks(bus, ticks)
                        rec_ticks = ticks
                        pose = new_pose
                        feet = new_feet
                    except (IKError, SoftLimitError) as e:
                        rec_err = str(e)
                        self._shared.bump_ik_errors(str(e))
                        if pose is not None:
                            gait.body = pose
                    # The free gait scales the command down to a speed its
                    # legs can actually service; say so rather than let the
                    # robot look like it is ignoring the stick.
                    free_msg = "Free"
                    if gait.command_scale < 0.99:
                        free_msg = f"Free — speed limited to {gait.command_scale:.0%}"
                    self._shared.set_status(
                        True, False, self._pose_dict(pose), free_msg, free_mode=True
                    )

                elif standing and walk_mode and gait is not None:
                    desired_gait = self._shared.get_gait_type()
                    if desired_gait != active_gait_type:
                        active_gait_type = desired_gait
                        snapped = {
                            leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()
                        }
                        gait = self._make_gait(
                            active_gait_type, gait.body, snapped, step_height, step_time
                        )

                    yaw_rad = math.radians(gait.body.yaw)
                    body_vx = -_dead(axes[AX_LSY]) * speed_cm
                    body_vy = -_dead(axes[AX_LSX]) * speed_cm
                    omega = -_dead(axes[AX_RSX]) * speed_deg
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
                        new_reach = (
                            self._shared.get_reach() + (rb - lb) * REACH_RATE_CMS * DT
                        )
                        self._shared.set_reach(new_reach)
                    gait.neutral_reach = self._shared.get_reach()
                    gait.step_height = step_height
                    gait.step_time = step_time

                    rec_cmd = (vx, vy, omega)
                    new_pose, new_feet = gait.step(vx, vy, omega, DT)
                    try:
                        ticks = self._compute_ticks(new_pose, new_feet, limits)
                        self._apply_ticks(bus, ticks)
                        rec_ticks = ticks
                        pose = new_pose
                        feet = new_feet
                    except (IKError, SoftLimitError) as e:
                        rec_err = str(e)
                        self._shared.bump_ik_errors(str(e))
                        if pose is not None:
                            gait.body = pose
                    self._shared.set_status(
                        True, False, self._pose_dict(pose), "Walking", walk_mode=True
                    )

                elif (
                    standing
                    and not walk_mode
                    and not free_mode
                    and pose is not None
                    and feet is not None
                ):
                    yaw_rad = math.radians(pose.yaw)
                    body_dx = -_dead(axes[AX_LSY]) * speed_cm * DT
                    body_dy = -_dead(axes[AX_LSX]) * speed_cm * DT
                    dx = body_dx * math.cos(yaw_rad) - body_dy * math.sin(yaw_rad)
                    dy = body_dx * math.sin(yaw_rad) + body_dy * math.cos(yaw_rad)
                    lt = _dead(buttons[BTN_LT])
                    rt = _dead(buttons[BTN_RT])
                    dz = (rt - lt) * speed_cm * DT
                    lb = 1.0 if buttons[BTN_LB] > 0.5 else 0.0
                    rb = 1.0 if buttons[BTN_RB] > 0.5 else 0.0
                    droll = _dead(axes[AX_RSX]) * speed_deg * DT
                    dpitch = -_dead(axes[AX_RSY]) * speed_deg * DT
                    dyaw = (lb - rb) * speed_deg * DT

                    if (
                        abs(dx)
                        + abs(dy)
                        + abs(dz)
                        + abs(droll)
                        + abs(dpitch)
                        + abs(dyaw)
                        > 1e-9
                    ):
                        new_pose = replace(
                            pose,
                            x=pose.x + dx,
                            y=pose.y + dy,
                            z=pose.z + dz,
                            roll=pose.roll + droll,
                            pitch=pose.pitch + dpitch,
                            yaw=pose.yaw + dyaw,
                        )
                        try:
                            ticks = self._compute_ticks(new_pose, feet, limits)
                            self._apply_ticks(bus, ticks)
                            rec_ticks = ticks
                            pose = new_pose
                        except (IKError, SoftLimitError) as e:
                            rec_err = str(e)
                            self._shared.bump_ik_errors(str(e))
                    self._shared.set_status(True, False, self._pose_dict(pose))

            if self._rec is not None:
                self._rec.tick(
                    mode=self._mode_name(standing, walk_mode, free_mode),
                    axes=axes,
                    buttons=buttons,
                    speeds=(speed_cm, speed_deg),
                    step_params=(step_height, step_time, step_threshold),
                    reach=self._shared.get_reach(),
                    cmd=rec_cmd,
                    pose=pose,
                    gait=gait,
                    gait_type="free" if free_mode else active_gait_type,
                    ticks=rec_ticks,
                    error=rec_err,
                )

            elapsed = time.monotonic() - t0
            rem = DT - elapsed
            if rem > 0:
                time.sleep(rem)

    # --- helpers ---

    @staticmethod
    def _mode_name(standing: bool, walk_mode: bool, free_mode: bool) -> str:
        if not standing:
            return "idle"
        if free_mode:
            return "free"
        if walk_mode:
            return "walk"
        return "pose"

    @staticmethod
    def _make_gait(
        gait_type: str,
        pose: BodyPose,
        feet: dict,
        step_height: float = 4.0,
        step_time: float = 0.40,
    ):
        kw = {
            "neutral_reach": _NEUTRAL_REACH,
            "step_height": step_height,
            "step_time": step_time,
        }
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
        pose: BodyPose,
        feet: dict[Leg, tuple[float, float, float]],
        limits: SoftLimits | None,
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
                Joint.COXA: angle_to_tick("coxa", tc),
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
        bus: ST3020Bus,
        limits: SoftLimits | None,
    ) -> tuple[BodyPose, dict[Leg, tuple[float, float, float]]]:
        pose = BodyPose(z=STAND_HEIGHT)
        feet = self._neutral_feet()
        ticks = self._compute_ticks(pose, feet, limits)
        # Sit and store leave the servos limp — without torque they ignore
        # every goal position and the robot never gets up again.
        for leg in Leg:
            for joint in Joint:
                bus.torque_enable(servo_id(leg, joint), True)
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
            (servo_id(leg, joint), 2048, STAND_SPEED) for leg in Leg for joint in Joint
        ]
        MotionPlayer(bus, acc=0).move(targets)
        for leg in Leg:
            for joint in Joint:
                bus.torque_enable(servo_id(leg, joint), False)

    @staticmethod
    def _do_store(bus: ST3020Bus, limits: SoftLimits | None) -> None:
        for leg in Leg:
            for joint in Joint:
                bus.torque_enable(servo_id(leg, joint), True)

        femur_up = limits.femur.max_deg if limits else STORAGE_FEMUR_DEG
        tibia_down = limits.tibia.min_deg if limits else STORAGE_TIBIA_DEG

        player = MotionPlayer(bus, acc=0)
        player.move(
            [
                (servo_id(leg, joint), 2048, STAND_SPEED)
                for leg in Leg
                for joint in Joint
            ]
        )

        femur_tick = angle_to_tick("femur", femur_up)
        tibia_tick = angle_to_tick("tibia", tibia_down)
        player.move(
            [
                (servo_id(leg, joint), tick, STAND_SPEED)
                for leg in Leg
                for joint, tick in (
                    (Joint.FEMUR, femur_tick),
                    (Joint.TIBIA, tibia_tick),
                )
            ]
        )

        for leg in Leg:
            for joint in Joint:
                bus.torque_enable(servo_id(leg, joint), False)

    @staticmethod
    def _pose_dict(pose: BodyPose | None) -> dict:
        if pose is None:
            return {}
        return {
            "x": round(pose.x, 2),
            "y": round(pose.y, 2),
            "z": round(pose.z, 2),
            "roll": round(pose.roll, 2),
            "pitch": round(pose.pitch, 2),
            "yaw": round(pose.yaw, 2),
        }
