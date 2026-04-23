"""
Thread-safe shared state and configuration for the web controller.
"""

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from hexapod.gait import _NEUTRAL_REACH

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

STAND_HEIGHT = 15.0   # cm
STAND_SPEED  = 300    # ticks/s for stand/sit motion

CONTROL_HZ = 20
DT         = 1.0 / CONTROL_HZ
DEADZONE   = 0.12

DEFAULT_RATE_CM  = 15.0   # cm/s max translation speed
DEFAULT_RATE_DEG = 60.0   # deg/s max rotation / pitch speed

SPEED_CM_MIN,  SPEED_CM_MAX,  SPEED_CM_STEP  = 0.5, 30.0, 0.5
SPEED_DEG_MIN, SPEED_DEG_MAX, SPEED_DEG_STEP = 2.0, 120.0, 2.0
DPAD_CM_RATE  = 3.0   # cm/s per second while D-pad held
DPAD_DEG_RATE = 12.0  # °/s per second while D-pad held

HEIGHT_MIN, HEIGHT_MAX = 8.0, 20.0   # cm
REACH_MIN,  REACH_MAX  = 12.0, 26.0  # cm
STEP_H_MIN, STEP_H_MAX = 1.0, 12.0   # cm
STEP_T_MIN, STEP_T_MAX = 0.15, 1.0   # s
REACH_RATE_CMS         = 3.0         # cm/s change rate when LB/RB held in walk mode
FREE_STEP_THRESHOLD    = 3.0         # cm from neutral before free-gait triggers a step
FREE_STEP_EMERGENCY    = 6.0         # cm — overrides adjacency constraint
STEP_THRESHOLD_MIN, STEP_THRESHOLD_MAX = 0.5, 8.0
SOFT_LIMIT_MARGIN_DEG_DEFAULT = 15.0
SOFT_LIMIT_MARGIN_DEG_MIN     =  5.0
SOFT_LIMIT_MARGIN_DEG_MAX     = 45.0

GAITS = ["tripod", "ripple", "wave"]

STORAGE_FEMUR_DEG = 90.0    # raise femur this many degrees above horizontal
STORAGE_TIBIA_DEG = -80.0   # fold tibia this many degrees inward

DEFAULT_CONFIG: dict = {
    "speed_cm":              DEFAULT_RATE_CM,
    "speed_deg":             DEFAULT_RATE_DEG,
    "reach":                 _NEUTRAL_REACH,
    "step_height":           4.0,
    "step_time":             0.40,
    "gait_type":             "tripod",
    "step_threshold":        FREE_STEP_THRESHOLD,
    "soft_limit_margin_deg": SOFT_LIMIT_MARGIN_DEG_DEFAULT,
}


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    try:
        return {**DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(shared: "SharedState", config_path: Path) -> None:
    status = shared.get_status()
    cfg = {k: status[k] for k in DEFAULT_CONFIG}
    config_path.write_text(json.dumps(cfg, indent=2))


def apply_config(cfg: dict, shared: "SharedState") -> None:
    shared.set_speeds(cfg.get("speed_cm", DEFAULT_RATE_CM), cfg.get("speed_deg", DEFAULT_RATE_DEG))
    shared.set_reach(cfg.get("reach", _NEUTRAL_REACH))
    shared.set_step_height(cfg.get("step_height", 4.0))
    shared.set_step_time(cfg.get("step_time", 0.40))
    shared.set_gait_type(cfg.get("gait_type", "tripod"))
    shared.set_step_threshold(cfg.get("step_threshold", FREE_STEP_THRESHOLD))
    shared.set_soft_limit_margin_deg(cfg.get("soft_limit_margin_deg", SOFT_LIMIT_MARGIN_DEG_DEFAULT))


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
        self._soft_limit_margin_deg: float = SOFT_LIMIT_MARGIN_DEG_DEFAULT
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

    def set_soft_limit_margin_deg(self, v: float) -> None:
        with self._lock:
            self._soft_limit_margin_deg = max(SOFT_LIMIT_MARGIN_DEG_MIN, min(SOFT_LIMIT_MARGIN_DEG_MAX, v))

    def get_soft_limit_margin_deg(self) -> float:
        with self._lock:
            return self._soft_limit_margin_deg

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
            self._stored    = False
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
                "step_height":           self._step_height,
                "step_time":             self._step_time,
                "step_threshold":        self._step_threshold,
                "soft_limit_margin_deg": self._soft_limit_margin_deg,
                "ik_errors":             self._ik_errors,
                "last_ik_error":  self._last_ik_error,
            }
