#!/usr/bin/env python3
"""Interactive servo calibration: write 128 to torque switch so firmware sets
current physical position as center (tick 2048).

Workflow per servo:
  Phase 1 – FREE:      Torque OFF. Physically move joint to its neutral angle.
                       SPACE = lock and proceed to fine-tune.
  Phase 2 – FINE-TUNE: Torque ON, servo holds position.
                       ← → = ±1 tick (fine)
                       ↑ ↓ = ±10 ticks (coarse)
                       ENTER = write center to firmware and move on.
  B = go back to previous servo.
  S = skip (no write).
  Q = quit.

Neutral angle reference (robot on flat surface):
  Coxa  — pointing straight out, perpendicular to the body side.
  Femur — horizontal (use a bubble level on the link).
  Tibia — vertical, hanging straight down (gravity reference).
"""

import os
import select
import sys
import termios
import tty
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexapod.servo.transport import SerialTransport, TransportError
from hexapod.servo.protocol import ProtocolError
from hexapod.servo.st3020 import ST3020Bus
from hexapod.robot.config import Leg, Joint, servo_id

PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000

LEG_ORDER = [
    Leg.FRONT_RIGHT,
    Leg.MID_RIGHT,
    Leg.REAR_RIGHT,
    Leg.REAR_LEFT,
    Leg.MID_LEFT,
    Leg.FRONT_LEFT,
]
JOINT_ORDER = [Joint.COXA, Joint.FEMUR, Joint.TIBIA]
SERVOS = [(leg, joint) for leg in LEG_ORDER for joint in JOINT_ORDER]


@contextmanager
def raw_mode():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key() -> str | None:
    """Non-blocking read via os.read() to avoid Python stdio buffering issues
    with multi-byte escape sequences (arrow keys)."""
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], 0.08)[0]:
        return None
    data = os.read(fd, 32)
    if not data:
        return None
    if data[0:1] == b"\x1b" and len(data) >= 3 and data[1:2] == b"[":
        code = chr(data[2])
        return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(code, f"ESC[{code}")
    if data[0:1] == b"\x1b":
        return "ESC"
    return data[0:1].decode("utf-8", errors="replace")


def safe_read_pos(bus: ST3020Bus, sid: int) -> int | None:
    try:
        return bus.read_position(sid)
    except (ProtocolError, TransportError):
        return None


def calibrate_servo(bus: ST3020Bus, leg: Leg, joint: Joint, index: int, total: int) -> str:
    """Calibrate one servo. Returns 'done', 'skip', 'back', or 'quit'."""
    sid = servo_id(leg, joint)

    print(f"\n{'='*58}")
    print(f"  [{index+1}/{total}]  Leg {leg.value} {leg.name}  /  {joint.name}  (ID {sid})")
    print(f"{'='*58}")

    # ── Phase 1: free position ──────────────────────────────────
    bus.torque_enable(sid, False)
    print("PHASE 1 – FREE  (torque OFF)")
    print("  Move joint to neutral.  SPACE=lock  B=back  S=skip  Q=quit")

    goal = 2048
    with raw_mode():
        while True:
            pos = safe_read_pos(bus, sid)
            label = f"{pos:4d}" if pos is not None else " ERR"
            sys.stdout.write(f"\r  Position: {label}   ")
            sys.stdout.flush()

            key = read_key()
            if key == " ":
                goal = pos if pos is not None else 2048
                break
            elif key in ("b", "B"):
                sys.stdout.write("\r  [back]\n")
                sys.stdout.flush()
                bus.torque_enable(sid, True)
                return "back"
            elif key in ("s", "S"):
                sys.stdout.write("\r  [skipped]\n")
                sys.stdout.flush()
                bus.torque_enable(sid, True)
                return "skip"
            elif key in ("q", "Q"):
                sys.stdout.write("\r  [quit]\n")
                sys.stdout.flush()
                bus.torque_enable(sid, True)
                return "quit"

    sys.stdout.write("\n")
    sys.stdout.flush()

    # ── Phase 2: fine-tune ──────────────────────────────────────
    bus.torque_enable(sid, True)
    bus.write_position(sid, goal, speed=80, acc=5)
    time.sleep(0.3)

    print(f"PHASE 2 – FINE-TUNE  (holding at {goal})")
    print("  ← →=±1 tick   ↑ ↓=±10 ticks   ENTER=set center   B=back  S=skip  Q=quit")

    with raw_mode():
        while True:
            pos = safe_read_pos(bus, sid)
            pos_label = f"{pos:4d}" if pos is not None else " ERR"
            sys.stdout.write(
                f"\r  goal={goal:4d}  pos={pos_label}   "
            )
            sys.stdout.flush()

            key = read_key()
            if key is None:
                continue

            nudge = {"LEFT": -1, "RIGHT": +1, "UP": +10, "DOWN": -10}.get(key)
            if nudge is not None:
                goal = max(0, min(4095, goal + nudge))
                bus.write_position(sid, goal, speed=80, acc=5)
            elif key in ("\r", "\n"):
                break
            elif key in ("b", "B"):
                sys.stdout.write("\r  [back]\n")
                sys.stdout.flush()
                return "back"
            elif key in ("s", "S"):
                sys.stdout.write("\r  [skipped]\n")
                sys.stdout.flush()
                return "skip"
            elif key in ("q", "Q"):
                sys.stdout.write("\r  [quit]\n")
                sys.stdout.flush()
                return "quit"

    sys.stdout.write("\n")
    sys.stdout.flush()

    # ── Write center position ────────────────────────────────────
    try:
        bus.set_middle_position(sid)
        print("  Center written. This position will now read as 2048.")
    except (ProtocolError, TransportError) as exc:
        print(f"  ERROR writing center: {exc}")
        return "skip"

    return "done"


def main() -> None:
    written = 0
    skipped = 0
    total = len(SERVOS)

    print("Hexapod servo calibration")
    print(f"Port: {PORT}  Baudrate: {BAUDRATE}")
    print("Writes torque-switch=128 so firmware adopts current position as 2048.\n")
    print("Neutral reference:")
    print("  Coxa  — perpendicular to body, pointing straight out")
    print("  Femur — horizontal (use a bubble level)")
    print("  Tibia — vertical, hanging straight down")

    with SerialTransport(PORT, BAUDRATE) as transport:
        bus = ST3020Bus(transport)

        i = 0
        while i < total:
            leg, joint = SERVOS[i]
            result = calibrate_servo(bus, leg, joint, i, total)

            if result == "done":
                written += 1
                i += 1
            elif result == "skip":
                skipped += 1
                i += 1
            elif result == "back":
                i = max(0, i - 1)
            elif result == "quit":
                break

            time.sleep(0.1)

    print(f"\n{'='*58}")
    print(f"Done.  Written: {written}  Skipped: {skipped}")


if __name__ == "__main__":
    main()
