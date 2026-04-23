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

CONFIG_PATH = Path(__file__).parent / "hexapod_config.json"

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
DPAD_CM_RATE  = 3.0   # cm/s per second while D-pad held
DPAD_DEG_RATE = 12.0  # °/s per second while D-pad held

HEIGHT_MIN, HEIGHT_MAX = 8.0, 20.0   # cm
REACH_MIN,  REACH_MAX  = 12.0, 26.0  # cm, range for neutral foot radius
STEP_H_MIN, STEP_H_MAX = 1.0, 12.0   # cm, swing arc height
STEP_T_MIN, STEP_T_MAX = 0.15, 1.0   # s,  per-leg swing duration
REACH_RATE_CMS         = 3.0         # cm/s change rate when LB/RB held in walk mode
FREE_STEP_THRESHOLD    = 3.0         # cm from neutral before free-gait triggers a step
FREE_STEP_EMERGENCY    = 6.0         # cm — overrides adjacency constraint to prevent going out of reach
STEP_THRESHOLD_MIN, STEP_THRESHOLD_MAX = 0.5, 8.0

DEFAULT_CONFIG: dict = {
    "speed_cm":       DEFAULT_RATE_CM,
    "speed_deg":      DEFAULT_RATE_DEG,
    "reach":          _NEUTRAL_REACH,
    "step_height":    4.0,
    "step_time":      0.40,
    "gait_type":      "tripod",
    "step_threshold": FREE_STEP_THRESHOLD,
}


def load_config() -> dict:
    try:
        return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(shared: "SharedState") -> None:
    status = shared.get_status()
    cfg = {k: status[k] for k in DEFAULT_CONFIG}
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def apply_config(cfg: dict, shared: "SharedState") -> None:
    shared.set_speeds(cfg.get("speed_cm", DEFAULT_RATE_CM), cfg.get("speed_deg", DEFAULT_RATE_DEG))
    shared.set_reach(cfg.get("reach", _NEUTRAL_REACH))
    shared.set_step_height(cfg.get("step_height", 4.0))
    shared.set_step_time(cfg.get("step_time", 0.40))
    shared.set_gait_type(cfg.get("gait_type", "tripod"))
    shared.set_step_threshold(cfg.get("step_threshold", FREE_STEP_THRESHOLD))

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
        self._reach:        float = _NEUTRAL_REACH
        self._step_height:  float = 4.0
        self._step_time:    float = 0.40
        self._step_threshold: float = FREE_STEP_THRESHOLD
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
        self._ik_errors: int = 0
        self._last_ik_error: str = ""

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

    def set_step_height(self, h: float) -> None:
        with self._lock:
            self._step_height = max(STEP_H_MIN, min(STEP_H_MAX, h))

    def set_step_time(self, t: float) -> None:
        with self._lock:
            self._step_time = max(STEP_T_MIN, min(STEP_T_MAX, t))

    def set_step_threshold(self, t: float) -> None:
        with self._lock:
            self._step_threshold = max(STEP_THRESHOLD_MIN, min(STEP_THRESHOLD_MAX, t))

    def get_step_params(self) -> tuple[float, float, float]:
        with self._lock:
            return self._step_height, self._step_time, self._step_threshold

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

    def bump_ik_errors(self, msg: str = "") -> None:
        with self._lock:
            self._ik_errors += 1
            if msg:
                self._last_ik_error = msg

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
                "gait_type":      self._gait_type,
                "step_height":    self._step_height,
                "step_time":      self._step_time,
                "step_threshold": self._step_threshold,
                "ik_errors":      self._ik_errors,
                "last_ik_error":  self._last_ik_error,
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
            step_height, step_time, step_threshold = self._shared.get_step_params()

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

                # D-pad: speed adjustment (continuous while held)
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
                        # Cycle phase gait (tripod → ripple → wave)
                        idx = GAITS.index(active_gait_type)
                        active_gait_type = GAITS[(idx + 1) % len(GAITS)]
                        self._shared.set_gait_type(active_gait_type)
                        if gait is not None:
                            snapped = {leg: (f[0], f[1], 0.0) for leg, f in gait.feet.items()}
                            gait = self._make_gait(active_gait_type, gait.body, snapped, step_height, step_time)
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
                            step_height=step_height,
                            step_time=step_time,
                            step_threshold=step_threshold,
                            step_emergency_threshold=FREE_STEP_EMERGENCY,
                            step_reach_max=REACH_MAX,
                            step_reach_min=REACH_MIN,
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
                        gait = self._make_gait(active_gait_type, pose, snapped, step_height, step_time)
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
                    # --- WALK MODE: phase gait (tripod / ripple / wave) ---
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
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Hexapod</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117; color: #c9d1d9;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px; padding: 0.4rem;
    height: 100vh; display: flex; flex-direction: column; gap: 0.25rem;
    overflow: hidden;
  }
  h2 { color: #8b949e; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }

  .badge {
    padding: 0.2rem 0.55rem; border-radius: 99px; font-size: 0.74rem; font-weight: 600;
    transition: background 0.2s, color 0.2s;
  }
  .off  { background: #21262d; color: #8b949e; }
  .ok   { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }
  .warn { background: #9e6a0333; color: #d29922; border: 1px solid #9e6a03; }
  .good { background: #1a7f3733; color: #3fb950; border: 1px solid #238636; }

  .panel {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 0.45rem 0.65rem;
  }

  /* Header */
  #hdr { flex: 0 0 auto; }
  #hdr-top { display: flex; align-items: center; flex-wrap: wrap; gap: 0.3rem; }
  #hdr-top h1 { color: #58a6ff; font-size: 1.05rem; letter-spacing: 0.04em; margin-right: 0.15rem; }
  #msg-area { font-size: 0.76rem; color: #d29922; min-height: 1em; margin-top: 0.1rem; }

  /* 2-column main layout */
  #main {
    flex: 1 1 0; display: grid;
    grid-template-columns: 1fr 1.45fr;
    gap: 0.35rem; min-height: 0;
  }
  #col-left, #col-right {
    display: flex; flex-direction: column; gap: 0.35rem; min-height: 0;
  }

  /* Body pose */
  .pose-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.25rem;
    font-family: 'Consolas', 'Courier New', monospace;
  }
  .pose-item { text-align: center; }
  .pose-label { font-size: 0.63rem; color: #8b949e; }
  .pose-val   { font-size: 0.95rem; color: #79c0ff; }

  /* Controller */
  .sticks { display: flex; gap: 0.9rem; align-items: center; margin-bottom: 0.35rem; }
  .stick-wrap { text-align: center; }
  .stick-label { font-size: 0.63rem; color: #8b949e; margin-bottom: 0.15rem; }
  canvas { display: block; border-radius: 50%; }
  .btns { display: flex; flex-wrap: wrap; gap: 0.2rem; }
  .btn {
    padding: 0.12rem 0.4rem; border-radius: 4px; font-size: 0.66rem;
    background: #21262d; color: #8b949e;
  }
  .btn.on { background: #1a7f37; color: #aff3c8; }

  /* Slider controls */
  .ctrl-label { font-size: 0.69rem; color: #8b949e; margin-bottom: 0.1rem; }
  .ctrl-row { display: flex; align-items: center; gap: 0.3rem; margin-bottom: 0.2rem; }
  .ctrl-val { font-family: monospace; font-size: 0.95rem; color: #79c0ff; min-width: 3.2rem; text-align: center; }
  .spdbtn {
    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 4px; width: 1.7rem; height: 1.7rem; font-size: 0.95rem;
    cursor: pointer; line-height: 1; flex-shrink: 0;
  }
  .spdbtn:active { background: #1f6feb; }
  .bar-track {
    flex: 1; height: 7px; background: #21262d; border-radius: 4px;
    overflow: hidden; cursor: pointer; touch-action: none;
  }
  .bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    transition: width 0.1s ease;
  }

  /* Speed 2-col sub-grid */
  .speed-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }

  /* Gait buttons */
  .gait-btns { display: flex; gap: 0.3rem; margin-bottom: 0.3rem; }
  .gait-btn {
    flex: 1; padding: 0.22rem 0; border-radius: 4px; font-size: 0.76rem;
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    cursor: pointer; text-align: center;
  }
  .gait-btn.active { background: #1f6feb33; color: #58a6ff; border-color: #1f6feb; }

  /* Config buttons */
  .cfg-btns { display: flex; gap: 0.35rem; margin-top: 0.35rem; }
  .cfg-btn {
    flex: 1; padding: 0.28rem; border-radius: 4px; font-size: 0.76rem; font-weight: 600;
    cursor: pointer; border: 1px solid #30363d;
  }
  .cfg-save  { background: #1a7f3733; color: #3fb950; border-color: #238636; }
  .cfg-save:active  { background: #1a7f37; }
  .cfg-reset { background: #9e6a0333; color: #d29922; border-color: #9e6a03; }
  .cfg-reset:active { background: #9e6a03; color: #fff; }

  /* Collapsible controls reference */
  details.panel summary {
    cursor: pointer; color: #58a6ff; font-size: 0.69rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; list-style: none;
  }
  details.panel summary::before { content: '▶  '; }
  details[open].panel summary::before { content: '▼  '; }
  details.panel .ref-wrap { margin-top: 0.35rem; overflow-y: auto; max-height: 180px; }
  table.controls { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  table.controls td { padding: 0.12rem 0.35rem 0.12rem 0; color: #8b949e; }
  table.controls td:first-child { color: #c9d1d9; font-weight: 600; white-space: nowrap; }
  table.controls .sh td { color: #58a6ff; font-size: 0.65rem; padding-top: 0.35rem; }
</style>
</head>
<body>

<div id="hdr">
  <div id="hdr-top">
    <h1>&#129264; Hexapod</h1>
    <span class="badge off" id="b-ws">WS: …</span>
    <span class="badge off" id="b-gp">Controller: none</span>
    <span class="badge off" id="b-robot">Sitting</span>
    <span class="badge off" id="b-ik">IK 0</span>
    <button class="badge warn" onclick="sendCommand('store')" style="cursor:pointer;border:1px solid #9e6a03">&#9660; Store</button>
  </div>
  <div id="msg-area">
    <span id="msg"></span>
    <span id="msg-ik" style="font-size:0.68rem;color:#8b949e;margin-left:0.4rem"></span>
  </div>
</div>

<div id="main">

  <!-- LEFT: pose + controller -->
  <div id="col-left">

    <div class="panel">
      <h2>Body Pose</h2>
      <div class="pose-grid">
        <div class="pose-item"><div class="pose-label">X fwd</div><div class="pose-val" id="px">—</div></div>
        <div class="pose-item"><div class="pose-label">Y left</div><div class="pose-val" id="py">—</div></div>
        <div class="pose-item"><div class="pose-label">Z up</div><div class="pose-val" id="pz">—</div></div>
        <div class="pose-item"><div class="pose-label">Roll</div><div class="pose-val" id="pr">—</div></div>
        <div class="pose-item"><div class="pose-label">Pitch</div><div class="pose-val" id="pp">—</div></div>
        <div class="pose-item"><div class="pose-label">Yaw</div><div class="pose-val" id="pw">—</div></div>
      </div>
    </div>

    <div class="panel" style="flex:1 1 0;min-height:0">
      <h2>Controller</h2>
      <p id="hint" style="color:#d29922;font-size:0.76rem;margin-bottom:0.3rem">&#128269; Press any button to activate.</p>
      <div class="sticks">
        <div class="stick-wrap">
          <div class="stick-label">Left</div>
          <canvas id="ls" width="56" height="56"></canvas>
        </div>
        <div class="stick-wrap">
          <div class="stick-label">Right</div>
          <canvas id="rs" width="56" height="56"></canvas>
        </div>
        <div>
          <div class="stick-label" style="margin-bottom:0.25rem">Triggers</div>
          <div style="font-family:monospace;font-size:0.78rem;color:#79c0ff">
            LT <span id="lt-val">0.00</span><br>RT <span id="rt-val">0.00</span>
          </div>
        </div>
      </div>
      <div class="btns" id="btns"></div>
    </div>

  </div>

  <!-- RIGHT: speed + walk settings + controls ref -->
  <div id="col-right">

    <div class="panel">
      <h2>Speed</h2>
      <div class="speed-grid">
        <div>
          <div class="ctrl-label">Translate (cm/s) <span style="font-size:0.62rem">D-pad ↑↓</span></div>
          <div class="ctrl-row">
            <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustSpeed('cm',-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
            <span class="ctrl-val" id="spd-cm">15.0</span>
            <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustSpeed('cm',+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
            <div class="bar-track" id="track-cm"><div class="bar-fill" id="bar-cm" style="width:100%"></div></div>
          </div>
        </div>
        <div>
          <div class="ctrl-label">Rotate (°/s) <span style="font-size:0.62rem">D-pad ←→</span></div>
          <div class="ctrl-row">
            <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustSpeed('deg',-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
            <span class="ctrl-val" id="spd-deg">60.0</span>
            <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustSpeed('deg',+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
            <div class="bar-track" id="track-deg"><div class="bar-fill" id="bar-deg" style="width:100%"></div></div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel" style="flex:1 1 0;min-height:0">
      <h2>Walk Settings</h2>

      <div class="ctrl-label">Gait <span style="font-size:0.62rem">Walk: Back cycles</span></div>
      <div class="gait-btns">
        <button class="gait-btn" id="gait-tripod" onclick="selectGait('tripod')">Tripod</button>
        <button class="gait-btn" id="gait-ripple" onclick="selectGait('ripple')">Ripple</button>
        <button class="gait-btn" id="gait-wave"   onclick="selectGait('wave')">Wave</button>
      </div>

      <div class="ctrl-label">Foot Reach (cm) <span style="font-size:0.62rem">Walk: LB/RB</span></div>
      <div class="ctrl-row">
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustReach(-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
        <span class="ctrl-val" id="reach-val">17.4</span>
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustReach(+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
        <div class="bar-track" id="track-reach"><div class="bar-fill" id="bar-reach" style="width:39%"></div></div>
      </div>

      <div class="ctrl-label">Step Height (cm)</div>
      <div class="ctrl-row">
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepH(-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
        <span class="ctrl-val" id="step-h-val">4.0</span>
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepH(+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
        <div class="bar-track" id="track-step-h"><div class="bar-fill" id="bar-step-h" style="width:27%"></div></div>
      </div>

      <div class="ctrl-label">Step Duration (s)</div>
      <div class="ctrl-row">
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepT(-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
        <span class="ctrl-val" id="step-t-val">0.40</span>
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepT(+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
        <div class="bar-track" id="track-step-t"><div class="bar-fill" id="bar-step-t" style="width:29%"></div></div>
      </div>

      <div class="ctrl-label">Free Step Threshold (cm)</div>
      <div class="ctrl-row">
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepThr(-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
        <span class="ctrl-val" id="step-thr-val">3.0</span>
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepThr(+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
        <div class="bar-track" id="track-step-thr"><div class="bar-fill" id="bar-step-thr" style="width:34%"></div></div>
      </div>

      <div class="cfg-btns">
        <button class="cfg-btn cfg-save"  onclick="sendCommand('save_config')">&#128190; Save Config</button>
        <button class="cfg-btn cfg-reset" onclick="sendCommand('reset_config')">&#8635; Reset Defaults</button>
      </div>
    </div>

    <details class="panel">
      <summary>Controls Reference</summary>
      <div class="ref-wrap">
        <table class="controls">
          <tr><td>A</td><td>Stand</td><td>B</td><td>Sit</td></tr>
          <tr><td>X</td><td>Toggle walk/pose</td><td>Y</td><td>Storage mode</td></tr>
          <tr><td>Back (standing)</td><td>Enter free mode</td><td>Back (free)</td><td>Exit free mode</td></tr>
          <tr><td>Start</td><td>Reset neutral</td><td></td><td></td></tr>
          <tr class="sh"><td colspan="4">POSE MODE</td></tr>
          <tr><td>Left stick</td><td>Translate X/Y</td><td>Right stick</td><td>Roll / Pitch</td></tr>
          <tr><td>LT / RT</td><td>Height</td><td>LB / RB</td><td>Yaw</td></tr>
          <tr class="sh"><td colspan="4">WALK MODE</td></tr>
          <tr><td>Left stick</td><td>Walk direction</td><td>Right stick X</td><td>Turn</td></tr>
          <tr><td>LT / RT</td><td>Height</td><td>LB / RB</td><td>Foot reach</td></tr>
          <tr><td>Back</td><td>Cycle gait</td><td>D-pad ↑↓</td><td>Speed ±cm/s</td></tr>
          <tr class="sh"><td colspan="4">FREE MODE</td></tr>
          <tr><td>Left stick</td><td>Walk (reactive)</td><td>Right stick</td><td>Roll / Pitch</td></tr>
          <tr><td>LT / RT</td><td>Height</td><td>LB / RB</td><td>Turn</td></tr>
        </table>
      </div>
    </details>

  </div>
</div>

<script>
const BTN_NAMES = ['A','B','X','Y','LB','RB','LT','RT','Back','Start','L3','R3','↑','↓','←','→','Home'];

let ws, wsOk = false;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen  = () => { wsOk = true;  badge('b-ws', 'WS: OK',  'ok'); };
  ws.onclose = () => { wsOk = false; badge('b-ws', 'WS: …',   'warn'); setTimeout(connect, 2000); };
  ws.onmessage = ev => updateStatus(JSON.parse(ev.data));
}

function badge(id, text, cls) {
  const el = document.getElementById(id);
  el.textContent = text; el.className = 'badge ' + cls;
}

let localSpeedCm = 15.0, localSpeedDeg = 60.0, localReach = 17.4;
let localStepH = 4.0, localStepT = 0.40, localStepThr = 3.0;
const STEP_CM = 0.5, STEP_DEG = 2.0, STEP_REACH = 0.5;
const MIN_CM = 0.5, MAX_CM = 30.0, MIN_DEG = 2.0, MAX_DEG = 120.0;
const MIN_REACH = 12.0, MAX_REACH = 26.0;
const STEP_H_STEP = 0.5, STEP_T_STEP = 0.05, STEP_THR_STEP = 0.25;
const STEP_H_MIN = 1.0, STEP_H_MAX = 12.0, STEP_T_MIN = 0.15, STEP_T_MAX = 1.0;
const STEP_THR_MIN = 0.5, STEP_THR_MAX = 8.0;

function sendCommand(cmd) {
  if (wsOk) ws.send(JSON.stringify({type:'command', cmd}));
}

function updateStatus(d) {
  if (d.busy)           badge('b-robot', 'Busy…',    'warn');
  else if (d.stored)    badge('b-robot', 'Stored',   'warn');
  else if (d.free_mode) badge('b-robot', 'Free',     'ok');
  else if (d.walk_mode) badge('b-robot', 'Walking',  'ok');
  else if (d.standing)  badge('b-robot', 'Standing', 'good');
  else                  badge('b-robot', 'Sitting',  'off');
  document.getElementById('msg').textContent = d.message || '';
  const p = d.pose;
  if (p && 'x' in p) {
    setText('px', p.x.toFixed(1));
    setText('py', p.y.toFixed(1));
    setText('pz', p.z.toFixed(1));
    setText('pr', p.roll.toFixed(1)  + '°');
    setText('pp', p.pitch.toFixed(1) + '°');
    setText('pw', p.yaw.toFixed(1)   + '°');
  } else { ['px','py','pz','pr','pp','pw'].forEach(id => setText(id, '—')); }
  if (d.speed_cm       !== undefined) { localSpeedCm  = d.speed_cm;       setSpeed('cm',  d.speed_cm); }
  if (d.speed_deg      !== undefined) { localSpeedDeg = d.speed_deg;      setSpeed('deg', d.speed_deg); }
  if (d.reach          !== undefined) { localReach    = d.reach;          setReach(d.reach); }
  if (d.step_height    !== undefined) { localStepH    = d.step_height;    setStepH(d.step_height); }
  if (d.step_time      !== undefined) { localStepT    = d.step_time;      setStepT(d.step_time); }
  if (d.step_threshold !== undefined) { localStepThr  = d.step_threshold; setStepThr(d.step_threshold); }
  if (d.gait_type !== undefined && d.gait_type !== localGait) { localGait = d.gait_type; setGait(d.gait_type); }
  if (d.ik_errors !== undefined) {
    const el = document.getElementById('b-ik');
    el.textContent = 'IK ' + d.ik_errors;
    el.className = 'badge ' + (d.ik_errors > 0 ? 'warn' : 'off');
    document.getElementById('msg-ik').textContent = d.last_ik_error || '';
  }
}

function setText(id, v) { document.getElementById(id).textContent = v; }

function setSpeed(axis, val) {
  if (axis === 'cm') {
    setText('spd-cm', val.toFixed(1));
    document.getElementById('bar-cm').style.width = ((val-MIN_CM)/(MAX_CM-MIN_CM)*100).toFixed(1)+'%';
  } else {
    setText('spd-deg', val.toFixed(1));
    document.getElementById('bar-deg').style.width = ((val-MIN_DEG)/(MAX_DEG-MIN_DEG)*100).toFixed(1)+'%';
  }
}
function setReach(v)   { setText('reach-val',    v.toFixed(1));  document.getElementById('bar-reach').style.width   = ((v-MIN_REACH)/(MAX_REACH-MIN_REACH)*100).toFixed(1)+'%'; }
function setStepH(v)   { setText('step-h-val',   v.toFixed(1));  document.getElementById('bar-step-h').style.width  = ((v-STEP_H_MIN)/(STEP_H_MAX-STEP_H_MIN)*100).toFixed(1)+'%'; }
function setStepT(v)   { setText('step-t-val',   v.toFixed(2));  document.getElementById('bar-step-t').style.width  = ((v-STEP_T_MIN)/(STEP_T_MAX-STEP_T_MIN)*100).toFixed(1)+'%'; }
function setStepThr(v) { setText('step-thr-val', v.toFixed(2));  document.getElementById('bar-step-thr').style.width = ((v-STEP_THR_MIN)/(STEP_THR_MAX-STEP_THR_MIN)*100).toFixed(1)+'%'; }

function adjustSpeed(axis, dir) {
  if (axis === 'cm') { localSpeedCm  = Math.max(MIN_CM,  Math.min(MAX_CM,  +(localSpeedCm  + dir*STEP_CM).toFixed(1)));  setSpeed('cm',  localSpeedCm); }
  else               { localSpeedDeg = Math.max(MIN_DEG, Math.min(MAX_DEG, +(localSpeedDeg + dir*STEP_DEG).toFixed(1))); setSpeed('deg', localSpeedDeg); }
  if (wsOk) ws.send(JSON.stringify({type:'speed', speed_cm:localSpeedCm, speed_deg:localSpeedDeg}));
}
function adjustReach(dir)    { localReach   = Math.max(MIN_REACH,  Math.min(MAX_REACH,  +(localReach   + dir*STEP_REACH).toFixed(1)));   setReach(localReach);    if (wsOk) ws.send(JSON.stringify({type:'reach',         reach:localReach})); }
function adjustStepH(dir)    { localStepH   = Math.max(STEP_H_MIN, Math.min(STEP_H_MAX, +(localStepH   + dir*STEP_H_STEP).toFixed(1)));  setStepH(localStepH);    if (wsOk) ws.send(JSON.stringify({type:'step_height',   value:localStepH})); }
function adjustStepT(dir)    { localStepT   = Math.max(STEP_T_MIN, Math.min(STEP_T_MAX, +(localStepT   + dir*STEP_T_STEP).toFixed(2)));  setStepT(localStepT);    if (wsOk) ws.send(JSON.stringify({type:'step_time',     value:localStepT})); }
function adjustStepThr(dir)  { localStepThr = Math.max(STEP_THR_MIN, Math.min(STEP_THR_MAX, +(localStepThr + dir*STEP_THR_STEP).toFixed(2))); setStepThr(localStepThr); if (wsOk) ws.send(JSON.stringify({type:'step_threshold', value:localStepThr})); }

let _pressTimer = null, _pressInterval = null;
function _pressStart(fn) { fn(); _pressTimer = setTimeout(() => { _pressInterval = setInterval(fn, 80); }, 450); }
function _pressStop()    { clearTimeout(_pressTimer); clearInterval(_pressInterval); _pressTimer = _pressInterval = null; }

function _makeDraggable(trackId, min, max, decimals, sender) {
  const track = document.getElementById(trackId);
  if (!track) return;
  let active = false;
  function fromPointer(e) {
    const rect = track.getBoundingClientRect();
    sender(+(min + Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width))*(max-min)).toFixed(decimals));
  }
  track.addEventListener('pointerdown', e => { e.preventDefault(); active=true; track.setPointerCapture(e.pointerId); fromPointer(e); });
  track.addEventListener('pointermove', e => { if (active) fromPointer(e); });
  track.addEventListener('pointerup',     () => { active=false; });
  track.addEventListener('pointercancel', () => { active=false; });
}

let localGait = 'tripod';
function selectGait(g) { localGait=g; setGait(g); if (wsOk) ws.send(JSON.stringify({type:'gait', gait:g})); }
function setGait(g) {
  ['tripod','ripple','wave'].forEach(n => {
    const el = document.getElementById('gait-'+n);
    if (el) el.className = 'gait-btn' + (n===g?' active':'');
  });
}

const seenIdx = new Set();
function activateGamepad(gp) {
  if (seenIdx.has(gp.index)) return;
  seenIdx.add(gp.index);
  badge('b-gp', gp.id.slice(0,26), 'ok');
  document.getElementById('hint').style.display = 'none';
  const wrap = document.getElementById('btns');
  wrap.innerHTML = '';
  BTN_NAMES.forEach((n,i) => {
    const d = document.createElement('span');
    d.className='btn'; d.id=`bn${i}`; d.textContent=n; wrap.appendChild(d);
  });
}
window.addEventListener('gamepadconnected',    e => activateGamepad(e.gamepad));
window.addEventListener('gamepaddisconnected', e => {
  seenIdx.delete(e.gamepad.index);
  if (!seenIdx.size) { badge('b-gp','Controller: none','off'); document.getElementById('hint').style.display=''; }
});

function drawStick(id, x, y) {
  const c=document.getElementById(id), ctx=c.getContext('2d');
  const cx=c.width/2, cy=c.height/2, r=cx-2;
  ctx.clearRect(0,0,c.width,c.height);
  ctx.strokeStyle='#21262d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI); ctx.stroke();
  ctx.strokeStyle='#30363d';
  [[cx-r,cy,cx+r,cy],[cx,cy-r,cx,cy+r]].forEach(([x1,y1,x2,y2])=>{ ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); });
  ctx.fillStyle='#58a6ff';
  ctx.beginPath(); ctx.arc(cx+x*r*0.88, cy+y*r*0.88, 4, 0, 2*Math.PI); ctx.fill();
}

let lastSend = 0;
function loop() {
  requestAnimationFrame(loop);
  const gps = navigator.getGamepads();
  let gp = null;
  for (const g of gps) { if (g) { gp=g; break; } }
  if (!gp) return;
  activateGamepad(gp);
  drawStick('ls', gp.axes[0]||0, gp.axes[1]||0);
  drawStick('rs', gp.axes[2]||0, gp.axes[3]||0);
  setText('lt-val', (gp.buttons[6]?.value||0).toFixed(2));
  setText('rt-val', (gp.buttons[7]?.value||0).toFixed(2));
  BTN_NAMES.forEach((_,i) => {
    const el=document.getElementById(`bn${i}`);
    if (el) el.className='btn'+(gp.buttons[i]?.pressed?' on':'');
  });
  const now=performance.now();
  if (now-lastSend < 1000/30 || !wsOk) return;
  lastSend=now;
  ws.send(JSON.stringify({axes:Array.from(gp.axes), buttons:Array.from(gp.buttons,b=>b.value), connected:true}));
}

connect();
setGait('tripod');
function _sendSpeeds() { if (wsOk) ws.send(JSON.stringify({type:'speed', speed_cm:localSpeedCm, speed_deg:localSpeedDeg})); }
_makeDraggable('track-cm',       MIN_CM,       MAX_CM,       1, v=>{localSpeedCm=v;  setSpeed('cm',v);   _sendSpeeds();});
_makeDraggable('track-deg',      MIN_DEG,      MAX_DEG,      1, v=>{localSpeedDeg=v; setSpeed('deg',v);  _sendSpeeds();});
_makeDraggable('track-reach',    MIN_REACH,    MAX_REACH,    1, v=>{localReach=v;    setReach(v);    if(wsOk)ws.send(JSON.stringify({type:'reach',         reach:v}));});
_makeDraggable('track-step-h',   STEP_H_MIN,   STEP_H_MAX,   1, v=>{localStepH=v;   setStepH(v);   if(wsOk)ws.send(JSON.stringify({type:'step_height',   value:v}));});
_makeDraggable('track-step-t',   STEP_T_MIN,   STEP_T_MAX,   2, v=>{localStepT=v;   setStepT(v);   if(wsOk)ws.send(JSON.stringify({type:'step_time',     value:v}));});
_makeDraggable('track-step-thr', STEP_THR_MIN, STEP_THR_MAX, 2, v=>{localStepThr=v; setStepThr(v); if(wsOk)ws.send(JSON.stringify({type:'step_threshold', value:v}));});
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
                    elif data.get("type") == "step_height":
                        shared.set_step_height(data.get("value", 4.0))
                    elif data.get("type") == "step_time":
                        shared.set_step_time(data.get("value", 0.40))
                    elif data.get("type") == "step_threshold":
                        shared.set_step_threshold(data.get("value", FREE_STEP_THRESHOLD))
                    elif data.get("type") == "command":
                        cmd = data.get("cmd", "")
                        if cmd == "save_config":
                            save_config(shared)
                        elif cmd == "reset_config":
                            apply_config(DEFAULT_CONFIG, shared)
                        else:
                            shared.request_command(cmd)
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
    apply_config(load_config(), shared)
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
