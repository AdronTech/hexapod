#!/usr/bin/env python3
"""
Web-based hexapod body controller.

Usage:
    uv run scripts/web_control.py [--port /dev/ttyACM0] [--bind 0.0.0.0] [--http-port 8080]

Opens http://<bind>:<http-port> in the browser (Steam Deck, laptop, phone …).
The browser reads the connected gamepad via the HTML5 Gamepad API and streams
control data to this server over a WebSocket.

Controller mapping (Xbox / Steam Deck layout):
  A                — stand
  B                — sit
  X                — toggle walk / pose mode
  Y                — storage mode (fold legs up, disable motors)
  Start            — reset to neutral pose

  Pose mode (body sway, feet planted):
  Left  stick X/Y  — body strafe / forward-back
  Right stick X/Y  — roll / pitch
  LT / RT          — body down / up  (analog)
  LB / RB          — yaw left / right  (digital)

  Walk mode (tripod / ripple / wave gait):
  Left  stick X/Y  — walk direction (body-relative)
  Right stick X    — turn left / right
  LT / RT          — body height
  LB / RB          — foot reach in / out
  Back             — cycle gait (tripod → ripple → wave)

  Free mode (reactive stepping + full body pose):
  Back (standing)  — enter free mode
  Back (free)      — exit free mode
  Left  stick X/Y  — walk direction (steps only when needed)
  Right stick X/Y  — roll / pitch
  LT / RT          — body height
  LB / RB          — turn left / right  (reach via web UI)

  D-pad ↑/↓        — translate speed ±0.5 cm/s
  D-pad ←/→        — rotate speed ±2 °/s
"""

import argparse
import asyncio
import json
import math
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexapod.body_ik import BodyPose, body_pose_ik, neutral_foot_body
from hexapod.gait import FreeGait, RippleGait, TripodGait, WaveGait, _NEUTRAL_REACH

GAITS = ["tripod", "ripple", "wave"]
from hexapod.kinematics import COXA_LEN, FEMUR_LEN
from hexapod.kinematics import IKError, angle_to_tick
from hexapod.robot.config import Joint, Leg, servo_id
from hexapod.robot.soft_limits import SoftLimitError, SoftLimits
from hexapod.servo.motion import MotionPlayer
from hexapod.servo.protocol import ProtocolError
from hexapod.servo.st3020 import PositionCommand, ST3020Bus
from hexapod.servo.transport import SerialTransport, TransportError

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8080

STAND_HEIGHT = 15.0   # cm
STAND_SPEED  = 300    # ticks/s for stand/sit motion

CONTROL_HZ  = 20
DT          = 1.0 / CONTROL_HZ
DEADZONE    = 0.12

# Default speeds — adjustable at runtime via UI or D-pad
DEFAULT_RATE_CM  = 15.0   # cm/s max translation speed
DEFAULT_RATE_DEG = 60.0   # deg/s max rotation / pitch speed
# Roll rate is always derived as RATE_DEG / 2

SPEED_CM_MIN,  SPEED_CM_MAX,  SPEED_CM_STEP  = 0.5, 30.0, 0.5
SPEED_DEG_MIN, SPEED_DEG_MAX, SPEED_DEG_STEP = 2.0, 120.0, 2.0

HEIGHT_MIN, HEIGHT_MAX = 8.0, 20.0   # cm
REACH_MIN,  REACH_MAX  = 12.0, 26.0  # cm, range for neutral foot radius
REACH_RATE_CMS         = 3.0         # cm/s change rate when LB/RB held in walk mode
FREE_STEP_THRESHOLD    = 8.0         # cm from neutral before free-gait triggers a step

# Storage pose — fallback angles when soft_limits.json is not present
STORAGE_FEMUR_DEG = 90.0    # raise femur this many degrees above horizontal
STORAGE_TIBIA_DEG = -80.0   # fold tibia this many degrees inward

# Standard Gamepad API button indices
BTN_A   = 0
BTN_B   = 1
BTN_X   = 2   # toggle walk / pose mode
BTN_Y   = 3   # storage mode
BTN_LB  = 4
BTN_RB  = 5
BTN_LT  = 6   # analog trigger, value 0..1
BTN_RT  = 7   # analog trigger, value 0..1
BTN_BACK   = 8   # Back / Select — cycle gait
BTN_START  = 9   # Start / Menu — reset pose to neutral
BTN_DUP    = 12  # D-pad up    — translate speed +
BTN_DDOWN  = 13  # D-pad down  — translate speed −
BTN_DLEFT  = 14  # D-pad left  — rotation speed −
BTN_DRIGHT = 15  # D-pad right — rotation speed +

# Axis indices
AX_LSX, AX_LSY = 0, 1
AX_RSX, AX_RSY = 2, 3


def _dead(v: float) -> float:
    """Deadzone + rescale to keep 0 at centre and ±1 at full deflection."""
    if abs(v) < DEADZONE:
        return 0.0
    s = 1.0 if v > 0 else -1.0
    return s * (abs(v) - DEADZONE) / (1.0 - DEADZONE)


# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------

class SharedState:
    """All cross-thread communication lives here."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Written by WebSocket handler, read by control thread
        self._axes: list[float]    = [0.0] * 8
        self._buttons: list[float] = [0.0] * 17
        self._gp_connected: bool   = False
        # Speeds — written by either side, clamped on write
        self._speed_cm:  float = DEFAULT_RATE_CM
        self._speed_deg: float = DEFAULT_RATE_DEG
        self._reach:     float = _NEUTRAL_REACH
        # Written by control thread, read by WebSocket handler
        self._standing: bool   = False
        self._busy: bool       = False
        self._walk_mode: bool  = False
        self._free_mode: bool  = False
        self._stored: bool     = False
        self._pose: dict       = {}
        self._message: str     = "Waiting for serial connection…"
        # Pending command from the web UI (one-shot; cleared after reading)
        self._pending_cmd: Optional[str] = None
        self._gait_type: str = "tripod"

    # --- input side ---

    def set_gamepad(self, axes: list[float], buttons: list[float], connected: bool) -> None:
        with self._lock:
            pad = lambda lst, n: (lst + [0.0] * n)[:n]
            self._axes    = pad(axes, 8)
            self._buttons = pad(buttons, 17)
            self._gp_connected = connected

    def get_gamepad(self) -> tuple[list[float], list[float], bool]:
        with self._lock:
            return list(self._axes), list(self._buttons), self._gp_connected

    def set_speeds(self, speed_cm: float, speed_deg: float) -> None:
        with self._lock:
            self._speed_cm  = max(SPEED_CM_MIN,  min(SPEED_CM_MAX,  speed_cm))
            self._speed_deg = max(SPEED_DEG_MIN, min(SPEED_DEG_MAX, speed_deg))

    def get_speeds(self) -> tuple[float, float]:
        with self._lock:
            return self._speed_cm, self._speed_deg

    def set_reach(self, reach: float) -> None:
        with self._lock:
            self._reach = max(REACH_MIN, min(REACH_MAX, reach))

    def get_reach(self) -> float:
        with self._lock:
            return self._reach

    # --- output side ---

    def set_status(
        self,
        standing: bool,
        busy: bool,
        pose: dict,
        message: str = "",
        walk_mode: bool = False,
        free_mode: bool = False,
    ) -> None:
        with self._lock:
            self._standing  = standing
            self._busy      = busy
            self._walk_mode = walk_mode
            self._free_mode = free_mode
            self._stored    = False   # any normal status update clears stored state
            self._pose      = dict(pose)
            self._message   = message

    def set_stored(self) -> None:
        with self._lock:
            self._standing  = False
            self._busy      = False
            self._walk_mode = False
            self._free_mode = False
            self._stored    = True
            self._pose      = {}
            self._message   = "Stored — press A to stand"

    def request_command(self, cmd: str) -> None:
        with self._lock:
            self._pending_cmd = cmd

    def pop_command(self) -> Optional[str]:
        with self._lock:
            cmd = self._pending_cmd
            self._pending_cmd = None
            return cmd

    def set_gait_type(self, gait: str) -> None:
        with self._lock:
            if gait in GAITS:
                self._gait_type = gait

    def get_gait_type(self) -> str:
        with self._lock:
            return self._gait_type

    def get_status(self) -> dict:
        with self._lock:
            return {
                "standing":  self._standing,
                "busy":      self._busy,
                "walk_mode": self._walk_mode,
                "free_mode": self._free_mode,
                "stored":    self._stored,
                "pose":      dict(self._pose),
                "message":   self._message,
                "speed_cm":  self._speed_cm,
                "speed_deg": self._speed_deg,
                "reach":     self._reach,
                "gait_type": self._gait_type,
            }


# ---------------------------------------------------------------------------
# Control thread
# ---------------------------------------------------------------------------

class ControlThread(threading.Thread):

    def __init__(self, serial_port: str, shared: SharedState) -> None:
        super().__init__(daemon=True, name="control")
        self._port   = serial_port
        self._shared = shared
        self._stop   = threading.Event()

    # --- public ---

    def stop(self) -> None:
        self._stop.set()

    # --- private ---

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

            # Edge detection: pressed = low → high this tick
            n = max(len(buttons), 17)
            buttons = (buttons + [0.0] * n)[:n]
            pressed = [
                buttons[i] > 0.5 and prev_btns[i] <= 0.5
                for i in range(n)
            ]
            prev_btns = list(buttons)

            pending_cmd = self._shared.pop_command()

            # Web UI "Store" command works even without a gamepad connected
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
                # Y button — storage mode (web UI path handled above)
                if pressed[BTN_Y]:
                    self._shared.set_status(standing, True, self._pose_dict(pose), "Storing…")
                    try:
                        self._do_store(bus, limits)
                        self._shared.set_stored()
                    except Exception as e:
                        self._shared.set_status(False, False, {}, f"Store failed: {e}")
                    pose = None; feet = None; gait = None
                    standing = False; walk_mode = False; free_mode = False
                    continue   # skip remaining button logic this tick

                # D-pad: speed adjustment (always active, even while sitting)
                if pressed[BTN_DUP]:
                    self._shared.set_speeds(speed_cm + SPEED_CM_STEP, speed_deg)
                elif pressed[BTN_DDOWN]:
                    self._shared.set_speeds(speed_cm - SPEED_CM_STEP, speed_deg)
                if pressed[BTN_DRIGHT]:
                    self._shared.set_speeds(speed_cm, speed_deg + SPEED_DEG_STEP)
                elif pressed[BTN_DLEFT]:
                    self._shared.set_speeds(speed_cm, speed_deg - SPEED_DEG_STEP)

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
                        # Cycle phase gait (tripod → ripple → wave)
                        idx = GAITS.index(active_gait_type)
                        active_gait_type = GAITS[(idx + 1) % len(GAITS)]
                        self._shared.set_gait_type(active_gait_type)
                        if gait is not None:
                            snapped = {leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()}
                            gait = self._make_gait(active_gait_type, gait.body, snapped)
                    elif free_mode:
                        # Exit free mode — snap feet to ground, return to pose mode
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
                        # Enter free mode from pose — snap feet to ground first
                        snapped = {leg: (f[0], f[1], 0.0) for leg, f in feet.items()}
                        gait = FreeGait(
                            replace(pose, roll=0.0, pitch=0.0), snapped,
                            neutral_reach=self._shared.get_reach(),
                            step_threshold=FREE_STEP_THRESHOLD,
                        )
                        free_mode = True
                        self._shared.set_status(
                            True, False, self._pose_dict(gait.body), "Free", free_mode=True
                        )

                elif pressed[BTN_X] and standing and not free_mode and pose is not None and feet is not None:
                    # Toggle walk / pose mode (phase gaits only)
                    walk_mode = not walk_mode
                    if walk_mode:
                        active_gait_type = self._shared.get_gait_type()
                        snapped = {leg: (f[0], f[1], 0.0) for leg, f in feet.items()}
                        gait = self._make_gait(active_gait_type, pose, snapped)
                    else:
                        # Snap swing feet to ground when returning to pose mode
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
                    # Reset to neutral pose; exits walk / free mode
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
                    # --- FREE MODE: reactive stepping + full body pose control ---
                    # Left stick: walk direction (triggers steps when feet drift)
                    # Right stick X/Y: roll / pitch  |  LT/RT: height  |  LB/RB: yaw
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

                    gait.neutral_reach = self._shared.get_reach()

                    new_pose, new_feet = gait.step(vx, vy, omega, DT)
                    try:
                        ticks = self._compute_ticks(new_pose, new_feet, limits)
                        self._apply_ticks(bus, ticks)
                        pose = new_pose
                        feet = new_feet
                    except (IKError, SoftLimitError):
                        pass
                    self._shared.set_status(
                        True, False, self._pose_dict(pose), "Free", free_mode=True
                    )

                elif standing and walk_mode and gait is not None:
                    # --- WALK MODE: phase gait (tripod / ripple / wave) ---
                    desired_gait = self._shared.get_gait_type()
                    if desired_gait != active_gait_type:
                        active_gait_type = desired_gait
                        snapped = {leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()}
                        gait = self._make_gait(active_gait_type, gait.body, snapped)

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

                    new_pose, new_feet = gait.step(vx, vy, omega, DT)
                    try:
                        ticks = self._compute_ticks(new_pose, new_feet, limits)
                        self._apply_ticks(bus, ticks)
                        pose = new_pose
                        feet = new_feet
                    except (IKError, SoftLimitError):
                        pass
                    self._shared.set_status(
                        True, False, self._pose_dict(pose), "Walking", walk_mode=True
                    )

                elif standing and not walk_mode and not free_mode and pose is not None and feet is not None:
                    # --- POSE MODE: body sway, feet stay planted ---
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
                        except (IKError, SoftLimitError):
                            pass
                    self._shared.set_status(True, False, self._pose_dict(pose))

            elapsed = time.monotonic() - t0
            rem = DT - elapsed
            if rem > 0:
                time.sleep(rem)

    # --- helpers ---

    @staticmethod
    def _make_gait(gait_type: str, pose: BodyPose, feet: dict):
        kw = dict(neutral_reach=_NEUTRAL_REACH)
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
                limits.check(tc, tf, tt)
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
        # Re-enable torque (safe to call even if already enabled)
        for leg in Leg:
            for joint in Joint:
                bus.torque_enable(servo_id(leg, joint), True)

        femur_up   = limits.femur.max_deg if limits else STORAGE_FEMUR_DEG
        tibia_down = limits.tibia.min_deg if limits else STORAGE_TIBIA_DEG

        player = MotionPlayer(bus, acc=0)

        # Step 1: all joints to neutral — safely lowers robot from any state
        player.move([
            (servo_id(leg, joint), 2048, STAND_SPEED)
            for leg in Leg
            for joint in Joint
        ])

        # Step 2: raise femurs and fold tibias simultaneously
        femur_tick = angle_to_tick("femur", femur_up)
        tibia_tick = angle_to_tick("tibia", tibia_down)
        player.move([
            (servo_id(leg, joint), tick, STAND_SPEED)
            for leg in Leg
            for joint, tick in ((Joint.FEMUR, femur_tick), (Joint.TIBIA, tibia_tick))
        ])

        # Disable all motors
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


# ---------------------------------------------------------------------------
# HTML (embedded so the script is self-contained)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hexapod</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117; color: #c9d1d9;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 1rem; font-size: 15px;
  }
  h1 { color: #58a6ff; margin-bottom: 0.75rem; font-size: 1.4rem; letter-spacing: 0.05em; }
  h2 { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }

  .row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
  .badge {
    padding: 0.25rem 0.7rem; border-radius: 99px; font-size: 0.8rem; font-weight: 600;
    transition: background 0.2s, color 0.2s;
  }
  .off  { background: #21262d; color: #8b949e; }
  .ok   { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }
  .warn { background: #9e6a0333; color: #d29922; border: 1px solid #9e6a03; }
  .good { background: #1a7f3733; color: #3fb950; border: 1px solid #238636; }

  .panel {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 0.75rem 1rem; margin-bottom: 0.75rem;
  }

  .pose-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem;
    font-family: 'Consolas', 'Courier New', monospace;
  }
  .pose-item { text-align: center; }
  .pose-label { font-size: 0.7rem; color: #8b949e; margin-bottom: 0.1rem; }
  .pose-val   { font-size: 1.1rem; color: #79c0ff; }

  .sticks { display: flex; gap: 1.5rem; align-items: center; margin-bottom: 0.6rem; }
  .stick-wrap { text-align: center; }
  .stick-label { font-size: 0.7rem; color: #8b949e; margin-bottom: 0.3rem; }
  canvas { display: block; border-radius: 50%; }

  .btns { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.4rem; }
  .btn {
    padding: 0.2rem 0.55rem; border-radius: 4px; font-size: 0.72rem;
    background: #21262d; color: #8b949e; transition: background 0.08s, color 0.08s;
  }
  .btn.on { background: #1a7f37; color: #aff3c8; }

  .message { font-size: 0.85rem; color: #d29922; min-height: 1.2em; margin-bottom: 0.5rem; }

  table.controls { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  table.controls td { padding: 0.2rem 0.5rem 0.2rem 0; color: #8b949e; }
  table.controls td:first-child { color: #c9d1d9; font-weight: 600; white-space: nowrap; }

  .spdbtn {
    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 4px; width: 2rem; height: 2rem; font-size: 1.1rem;
    cursor: pointer; line-height: 1;
  }
  .spdbtn:active { background: #1f6feb; }

  .speedbar-track {
    height: 6px; background: #21262d; border-radius: 3px;
    margin-top: 0.4rem; overflow: hidden;
  }
  .speedbar-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #1f6feb 0%, #58a6ff 100%);
    transition: width 0.15s ease;
  }

  .gait-btn {
    padding: 0.3rem 0.9rem; border-radius: 4px; font-size: 0.82rem;
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    cursor: pointer; transition: background 0.1s, color 0.1s, border-color 0.1s;
  }
  .gait-btn.active {
    background: #1f6feb33; color: #58a6ff; border-color: #1f6feb;
  }
</style>
</head>
<body>
<h1>&#129264; Hexapod</h1>

<div class="row">
  <span class="badge off" id="b-ws">WS: …</span>
  <span class="badge off" id="b-gp">Controller: none</span>
  <span class="badge off" id="b-robot">Sitting</span>
  <button class="badge warn" id="btn-store" onclick="sendCommand('store')" style="cursor:pointer;border:none">&#9660; Store</button>
</div>

<p class="message" id="msg"></p>

<div class="panel">
  <h2>Body Pose</h2>
  <div class="pose-grid">
    <div class="pose-item"><div class="pose-label">X forward</div><div class="pose-val" id="px">—</div></div>
    <div class="pose-item"><div class="pose-label">Y left</div><div class="pose-val" id="py">—</div></div>
    <div class="pose-item"><div class="pose-label">Z up</div><div class="pose-val" id="pz">—</div></div>
    <div class="pose-item"><div class="pose-label">Roll</div><div class="pose-val" id="pr">—</div></div>
    <div class="pose-item"><div class="pose-label">Pitch</div><div class="pose-val" id="pp">—</div></div>
    <div class="pose-item"><div class="pose-label">Yaw</div><div class="pose-val" id="pw">—</div></div>
  </div>
</div>

<div class="panel">
  <h2>Controller</h2>
  <p id="hint" style="color:#d29922;font-size:0.85rem;margin-bottom:0.6rem">
    &#128269; Press any button on your controller to activate it.
  </p>
  <div class="sticks">
    <div class="stick-wrap">
      <div class="stick-label">Left stick</div>
      <canvas id="ls" width="72" height="72"></canvas>
    </div>
    <div class="stick-wrap">
      <div class="stick-label">Right stick</div>
      <canvas id="rs" width="72" height="72"></canvas>
    </div>
    <div style="flex:1">
      <div class="stick-label" style="margin-bottom:0.4rem">Triggers</div>
      <div style="font-family:monospace;font-size:0.85rem;color:#79c0ff">
        LT <span id="lt-val">0.00</span> &nbsp; RT <span id="rt-val">0.00</span>
      </div>
    </div>
  </div>
  <div class="btns" id="btns"></div>
</div>

<div class="panel">
  <h2>Speed</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
    <div>
      <div style="font-size:0.75rem;color:#8b949e;margin-bottom:0.3rem">Translate (cm/s) &nbsp;<span style="color:#8b949e;font-size:0.7rem">D-pad ↑↓</span></div>
      <div style="display:flex;align-items:center;gap:0.4rem">
        <button class="spdbtn" onclick="adjustSpeed('cm',-1)">−</button>
        <span id="spd-cm" style="font-family:monospace;font-size:1.1rem;color:#79c0ff;min-width:3rem;text-align:center">15.0</span>
        <button class="spdbtn" onclick="adjustSpeed('cm',+1)">+</button>
      </div>
      <div class="speedbar-track"><div class="speedbar-fill" id="bar-cm" style="width:100%"></div></div>
    </div>
    <div>
      <div style="font-size:0.75rem;color:#8b949e;margin-bottom:0.3rem">Rotate (°/s) &nbsp;<span style="color:#8b949e;font-size:0.7rem">D-pad ←→</span></div>
      <div style="display:flex;align-items:center;gap:0.4rem">
        <button class="spdbtn" onclick="adjustSpeed('deg',-1)">−</button>
        <span id="spd-deg" style="font-family:monospace;font-size:1.1rem;color:#79c0ff;min-width:3rem;text-align:center">60.0</span>
        <button class="spdbtn" onclick="adjustSpeed('deg',+1)">+</button>
      </div>
      <div class="speedbar-track"><div class="speedbar-fill" id="bar-deg" style="width:100%"></div></div>
    </div>
  </div>
</div>

<div class="panel">
  <h2>Walk Settings</h2>
  <div style="margin-bottom:0.75rem">
    <div style="font-size:0.75rem;color:#8b949e;margin-bottom:0.4rem">
      Gait &nbsp;<span style="color:#8b949e;font-size:0.7rem">Walk: Back button cycles</span>
    </div>
    <div style="display:flex;gap:0.4rem">
      <button class="gait-btn" id="gait-tripod" onclick="selectGait('tripod')">Tripod</button>
      <button class="gait-btn" id="gait-ripple" onclick="selectGait('ripple')">Ripple</button>
      <button class="gait-btn" id="gait-wave"   onclick="selectGait('wave')">Wave</button>
    </div>
    <div style="font-size:0.7rem;color:#8b949e;margin-top:0.35rem">
      Tripod: 3 legs · fast &nbsp;|&nbsp; Ripple: 2 legs · medium &nbsp;|&nbsp; Wave: 1 leg · stable
    </div>
  </div>
  <div>
    <div style="font-size:0.75rem;color:#8b949e;margin-bottom:0.3rem">
      Foot Reach (cm) &nbsp;<span style="color:#8b949e;font-size:0.7rem">Walk: LB/RB</span>
    </div>
    <div style="display:flex;align-items:center;gap:0.4rem">
      <button class="spdbtn" onclick="adjustReach(-1)">−</button>
      <span id="reach-val" style="font-family:monospace;font-size:1.1rem;color:#79c0ff;min-width:3rem;text-align:center">17.4</span>
      <button class="spdbtn" onclick="adjustReach(+1)">+</button>
    </div>
    <div class="speedbar-track"><div class="speedbar-fill" id="bar-reach" style="width:39%"></div></div>
  </div>
</div>

<div class="panel">
  <h2>Controls</h2>
  <table class="controls">
    <tr><td>A</td><td>Stand</td><td>B</td><td>Sit</td></tr>
    <tr><td>X</td><td>Toggle walk / pose</td><td>Y</td><td>Storage mode</td></tr>
    <tr><td>Back (standing)</td><td>Enter free mode</td><td>Back (free)</td><td>Exit free mode</td></tr>
    <tr><td>Start</td><td>Reset to neutral</td><td></td><td></td></tr>
    <tr><td colspan="4" style="color:#58a6ff;padding-top:0.5rem;font-size:0.75rem">POSE MODE</td></tr>
    <tr><td>Left stick</td><td>Translate body X/Y</td><td>Right stick</td><td>Roll / Pitch</td></tr>
    <tr><td>LT / RT</td><td>Body down / up</td><td>LB / RB</td><td>Yaw left / right</td></tr>
    <tr><td colspan="4" style="color:#58a6ff;padding-top:0.5rem;font-size:0.75rem">WALK MODE (tripod / ripple / wave)</td></tr>
    <tr><td>Left stick</td><td>Walk direction</td><td>Right stick X</td><td>Turn left / right</td></tr>
    <tr><td>LT / RT</td><td>Body height</td><td>LB / RB</td><td>Foot reach in / out</td></tr>
    <tr><td>Back</td><td>Cycle gait (tripod→ripple→wave)</td><td></td><td></td></tr>
    <tr><td colspan="4" style="color:#58a6ff;padding-top:0.5rem;font-size:0.75rem">FREE MODE</td></tr>
    <tr><td>Left stick</td><td>Walk direction (steps when needed)</td><td>Right stick X</td><td>Roll</td></tr>
    <tr><td>Right stick Y</td><td>Pitch</td><td>LB / RB</td><td>Turn left / right</td></tr>
    <tr><td>LT / RT</td><td>Body height</td><td></td><td>Reach via web UI</td></tr>
    <tr><td>D-pad ↑/↓</td><td>Speed ±0.5 cm/s</td><td>D-pad ←/→</td><td>Turn rate ±2 °/s</td></tr>
  </table>
</div>

<script>
const BTN_NAMES = ['A','B','X','Y','LB','RB','LT','RT','Back','Start','L3','R3','↑','↓','←','→','Home'];

// --- WebSocket ---
let ws, wsOk = false;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen  = () => { wsOk = true;  badge('b-ws', 'WS: Connected', 'ok'); };
  ws.onclose = () => { wsOk = false; badge('b-ws', 'WS: Disconnected', 'warn'); setTimeout(connect, 2000); };
  ws.onmessage = ev => updateStatus(JSON.parse(ev.data));
}

function badge(id, text, cls) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = 'badge ' + cls;
}

// Speed / reach state (kept locally to avoid needing a round-trip before the first +/− click)
let localSpeedCm  = 15.0;
let localSpeedDeg = 60.0;
let localReach    = 17.4;
const STEP_CM = 0.5, STEP_DEG = 2.0, STEP_REACH = 0.5;
const MIN_CM = 0.5, MAX_CM = 30.0, MIN_DEG = 2.0, MAX_DEG = 120.0;
const MIN_REACH = 12.0, MAX_REACH = 26.0;

function sendCommand(cmd) {
  if (wsOk) ws.send(JSON.stringify({ type: 'command', cmd: cmd }));
}

function updateStatus(d) {
  if (d.busy)               badge('b-robot', 'Busy…',    'warn');
  else if (d.stored)        badge('b-robot', 'Stored',   'warn');
  else if (d.free_mode)     badge('b-robot', 'Free',     'ok');
  else if (d.walk_mode)     badge('b-robot', 'Walking',  'ok');
  else if (d.standing)      badge('b-robot', 'Standing', 'good');
  else                      badge('b-robot', 'Sitting',  'off');
  document.getElementById('msg').textContent = d.message || '';
  const p = d.pose;
  if (p && 'x' in p) {
    setText('px', p.x.toFixed(1) + ' cm');
    setText('py', p.y.toFixed(1) + ' cm');
    setText('pz', p.z.toFixed(1) + ' cm');
    setText('pr', p.roll.toFixed(1)  + '°');
    setText('pp', p.pitch.toFixed(1) + '°');
    setText('pw', p.yaw.toFixed(1)   + '°');
  } else {
    ['px','py','pz','pr','pp','pw'].forEach(id => setText(id, '—'));
  }
  // Sync speed / reach / gait display from server
  if (d.speed_cm  !== undefined) { localSpeedCm  = d.speed_cm;  setSpeed('cm',  d.speed_cm);  }
  if (d.speed_deg !== undefined) { localSpeedDeg = d.speed_deg; setSpeed('deg', d.speed_deg); }
  if (d.reach     !== undefined) { localReach    = d.reach;     setReach(d.reach); }
  if (d.gait_type !== undefined && d.gait_type !== localGait) { localGait = d.gait_type; setGait(d.gait_type); }
}

function setText(id, v) { document.getElementById(id).textContent = v; }

function setSpeed(axis, val) {
  if (axis === 'cm') {
    setText('spd-cm', val.toFixed(1));
    document.getElementById('bar-cm').style.width =
      ((val - MIN_CM) / (MAX_CM - MIN_CM) * 100).toFixed(1) + '%';
  } else {
    setText('spd-deg', val.toFixed(1));
    document.getElementById('bar-deg').style.width =
      ((val - MIN_DEG) / (MAX_DEG - MIN_DEG) * 100).toFixed(1) + '%';
  }
}

function setReach(val) {
  setText('reach-val', val.toFixed(1));
  document.getElementById('bar-reach').style.width =
    ((val - MIN_REACH) / (MAX_REACH - MIN_REACH) * 100).toFixed(1) + '%';
}

function adjustSpeed(axis, dir) {
  if (axis === 'cm') {
    localSpeedCm = Math.max(MIN_CM,  Math.min(MAX_CM,  +(localSpeedCm  + dir * STEP_CM).toFixed(1)));
    setSpeed('cm', localSpeedCm);
  } else {
    localSpeedDeg = Math.max(MIN_DEG, Math.min(MAX_DEG, +(localSpeedDeg + dir * STEP_DEG).toFixed(1)));
    setSpeed('deg', localSpeedDeg);
  }
  if (wsOk) ws.send(JSON.stringify({ type: 'speed', speed_cm: localSpeedCm, speed_deg: localSpeedDeg }));
}

function adjustReach(dir) {
  localReach = Math.max(MIN_REACH, Math.min(MAX_REACH, +(localReach + dir * STEP_REACH).toFixed(1)));
  setReach(localReach);
  if (wsOk) ws.send(JSON.stringify({ type: 'reach', reach: localReach }));
}

let localGait = 'tripod';
function selectGait(g) {
  localGait = g;
  setGait(g);
  if (wsOk) ws.send(JSON.stringify({ type: 'gait', gait: g }));
}
function setGait(g) {
  ['tripod','ripple','wave'].forEach(name => {
    const el = document.getElementById('gait-' + name);
    if (el) el.className = 'gait-btn' + (name === g ? ' active' : '');
  });
}

// --- Gamepad ---
// seenIdx tracks which gamepad indices have had their button UI created
const seenIdx = new Set();

function activateGamepad(gp) {
  if (seenIdx.has(gp.index)) return;
  seenIdx.add(gp.index);
  badge('b-gp', gp.id.slice(0, 28), 'ok');
  document.getElementById('hint').style.display = 'none';
  const wrap = document.getElementById('btns');
  wrap.innerHTML = '';
  BTN_NAMES.forEach((n, i) => {
    const d = document.createElement('span');
    d.className = 'btn'; d.id = `bn${i}`; d.textContent = n;
    wrap.appendChild(d);
  });
}

// gamepadconnected only fires on first button press — also catch already-active pads in the loop
window.addEventListener('gamepadconnected',    e => activateGamepad(e.gamepad));
window.addEventListener('gamepaddisconnected', e => {
  seenIdx.delete(e.gamepad.index);
  if (!seenIdx.size) {
    badge('b-gp', 'Controller: none', 'off');
    document.getElementById('hint').style.display = '';
  }
});

function drawStick(id, x, y) {
  const c = document.getElementById(id), ctx = c.getContext('2d');
  const cx = c.width/2, cy = c.height/2, r = cx - 3;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2*Math.PI); ctx.stroke();
  ctx.strokeStyle = '#30363d';
  [[cx-r,cy,cx+r,cy],[cx,cy-r,cx,cy+r]].forEach(([x1,y1,x2,y2]) => {
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  });
  ctx.fillStyle = '#58a6ff';
  ctx.beginPath(); ctx.arc(cx + x*r*0.88, cy + y*r*0.88, 5, 0, 2*Math.PI); ctx.fill();
}

let lastSend = 0;
function loop() {
  requestAnimationFrame(loop);
  const gps = navigator.getGamepads();
  let gp = null;
  for (const g of gps) { if (g) { gp = g; break; } }

  if (!gp) return;

  // Catch pads that were already active before the page loaded
  activateGamepad(gp);

  drawStick('ls', gp.axes[0] || 0, gp.axes[1] || 0);
  drawStick('rs', gp.axes[2] || 0, gp.axes[3] || 0);
  setText('lt-val', (gp.buttons[6]?.value || 0).toFixed(2));
  setText('rt-val', (gp.buttons[7]?.value || 0).toFixed(2));

  BTN_NAMES.forEach((_, i) => {
    const el = document.getElementById(`bn${i}`);
    if (el) el.className = 'btn' + (gp.buttons[i]?.pressed ? ' on' : '');
  });

  const now = performance.now();
  if (now - lastSend < 1000/30 || !wsOk) return;
  lastSend = now;
  ws.send(JSON.stringify({
    axes:      Array.from(gp.axes),
    buttons:   Array.from(gp.buttons, b => b.value),
    connected: true,
  }));
}

connect();
setGait('tripod');
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(shared: SharedState) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        shared.set_gamepad([], [], False)

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        async def send_loop() -> None:
            try:
                while True:
                    await ws.send_text(json.dumps(shared.get_status()))
                    await asyncio.sleep(0.1)
            except Exception:
                pass  # connection closed — exit silently

        send_task = asyncio.create_task(send_loop())
        try:
            async for raw in ws.iter_text():
                try:
                    data = json.loads(raw)
                    if data.get("type") == "speed":
                        sc, sd = shared.get_speeds()
                        shared.set_speeds(
                            data.get("speed_cm",  sc),
                            data.get("speed_deg", sd),
                        )
                    elif data.get("type") == "reach":
                        shared.set_reach(data.get("reach", _NEUTRAL_REACH))
                    elif data.get("type") == "gait":
                        shared.set_gait_type(data.get("gait", "tripod"))
                    elif data.get("type") == "command":
                        shared.request_command(data.get("cmd", ""))
                    else:
                        shared.set_gamepad(
                            data.get("axes", []),
                            data.get("buttons", []),
                            data.get("connected", False),
                        )
                except (json.JSONDecodeError, KeyError):
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            shared.set_gamepad([], [], False)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hexapod web controller")
    parser.add_argument("--port",      default=DEFAULT_SERIAL_PORT, help="Serial port")
    parser.add_argument("--bind",      default=DEFAULT_HTTP_HOST,   help="HTTP bind address")
    parser.add_argument("--http-port", default=DEFAULT_HTTP_PORT, type=int, help="HTTP port")
    args = parser.parse_args()

    shared  = SharedState()
    ctrl    = ControlThread(args.port, shared)
    ctrl.start()

    app = build_app(shared)
    print(f"Open http://{args.bind}:{args.http_port} in your browser.")
    try:
        uvicorn.run(app, host=args.bind, port=args.http_port, log_level="warning")
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
