"""
Control-loop recorder: one JSON object per control tick, newline delimited.

The recorder is a passive observer — it never touches the robot.  Every tick
of ControlThread._loop is written out with the gamepad input that produced it,
the resolved command velocities, the body pose, the per-leg gait state and the
servo ticks that went on the wire.  A recording therefore replays the whole
decision chain offline, which is what `scripts/analyze_recording.py` reads.

File layout:

    {"type": "header", "v": 1, ...}     # once, first line
    {"type": "tick",  "t": 0.05, ...}   # one per control tick
    {"type": "mark",  "t": 3.20, ...}   # operator-flagged moment

Use it as a context manager, or call close() when the control thread exits.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Self

from hexapod.body_ik import BodyPose
from hexapod.control.state import (
    CONTROL_HZ,
    DT,
    FREE_STEP_EMERGENCY,
    HEIGHT_MAX,
    HEIGHT_MIN,
    REACH_MAX,
    REACH_MIN,
)
from hexapod.robot.config import Joint, Leg, servo_id

SCHEMA_VERSION = 1

# Ticks buffered before the file is flushed.  One second of data, so a crash
# during a glitch costs at most the last second.
_FLUSH_EVERY = CONTROL_HZ


def _r(v: float | None, nd: int = 3) -> float | None:
    """Round for compactness; NaN/inf become None so the JSON stays valid."""
    if v is None:
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


class Recorder:
    """Appends one JSONL record per control tick to *path*."""

    def __init__(self, path: str | Path, meta: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", buffering=1024 * 64)
        self._t0 = time.monotonic()
        self._n = 0
        self._marks = 0
        self._since_flush = 0
        self._write(
            {
                "type": "header",
                "v": SCHEMA_VERSION,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "control_hz": CONTROL_HZ,
                "dt": DT,
                "limits": {
                    "height": [HEIGHT_MIN, HEIGHT_MAX],
                    "reach": [REACH_MIN, REACH_MAX],
                    "free_emergency": FREE_STEP_EMERGENCY,
                },
                "legs": [leg.name for leg in Leg],
                "servo_ids": {
                    leg.name: {j.name: servo_id(leg, j) for j in Joint} for leg in Leg
                },
                **(meta or {}),
            }
        )

    # --- writing ---

    def _write(self, obj: dict[str, Any]) -> None:
        if self._fh.closed:
            return
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._since_flush += 1
        if self._since_flush >= _FLUSH_EVERY:
            self._fh.flush()
            self._since_flush = 0

    def mark(self, label: str = "") -> None:
        """Flag the current moment — the operator saw something go wrong."""
        self._marks += 1
        self._write(
            {
                "type": "mark",
                "t": _r(time.monotonic() - self._t0),
                "n": self._marks,
                "label": label,
                "tick": self._n,
            }
        )
        self._fh.flush()
        self._since_flush = 0

    def tick(
        self,
        *,
        mode: str,
        axes: list[float],
        buttons: list[float],
        speeds: tuple[float, float],
        step_params: tuple[float, float, float],
        reach: float,
        cmd: tuple[float, float, float] | None = None,
        pose: BodyPose | None = None,
        gait: Any = None,
        gait_type: str = "",
        ticks: dict[Leg, dict[Joint, int]] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Record one control tick.

        Parameters mirror what the control loop already has in hand:
        *cmd* is the world-frame (vx, vy, omega) actually fed to the gait,
        *ticks* the servo goals that were written (None if nothing was sent),
        *error* the IK/soft-limit message if this tick failed.
        """
        rec: dict[str, Any] = {
            "type": "tick",
            "t": _r(time.monotonic() - self._t0),
            "i": self._n,
            "mode": mode,
            "ax": [_r(a, 4) for a in axes[:4]],
            "btn": [i for i, b in enumerate(buttons) if b > 0.5],
            "trig": [
                _r(buttons[6] if len(buttons) > 6 else 0.0, 3),
                _r(buttons[7] if len(buttons) > 7 else 0.0, 3),
            ],
            "cfg": {
                "speed_cm": _r(speeds[0], 2),
                "speed_deg": _r(speeds[1], 2),
                "reach": _r(reach, 2),
                "step_height": _r(step_params[0], 2),
                "step_time": _r(step_params[1], 3),
                "step_threshold": _r(step_params[2], 2),
            },
        }
        self._n += 1

        if gait_type:
            rec["gait"] = gait_type
        if cmd is not None:
            rec["cmd"] = [_r(cmd[0], 3), _r(cmd[1], 3), _r(cmd[2], 3)]
        if pose is not None:
            rec["pose"] = [
                _r(pose.x),
                _r(pose.y),
                _r(pose.z),
                _r(pose.roll, 3),
                _r(pose.pitch, 3),
                _r(pose.yaw, 3),
            ]
        if gait is not None:
            rec["legs"] = {
                name: self._leg_record(d) for name, d in gait.diagnostics().items()
            }
        if ticks is not None:
            rec["ticks"] = {
                str(servo_id(leg, j)): ticks[leg][j] for leg in ticks for j in Joint
            }
        if error:
            rec["err"] = error

        self._write(rec)

    @staticmethod
    def _leg_record(d: dict) -> dict:
        out: dict[str, Any] = {
            "foot": [_r(v) for v in d["foot"]],
            "neutral": [_r(v) for v in d["neutral"]],
            "err": _r(d["err"]),
            "swing": bool(d["swing"]),
            "t": _r(d["t"], 3),
        }
        if d.get("target") is not None:
            out["target"] = [_r(v) for v in d["target"]]
        for k in ("due", "emergency"):
            if k in d:
                out[k] = bool(d[k])
        return out

    # --- lifecycle ---

    def close(self) -> None:
        if not self._fh.closed:
            self._write(
                {
                    "type": "footer",
                    "t": _r(time.monotonic() - self._t0),
                    "ticks": self._n,
                    "marks": self._marks,
                }
            )
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
