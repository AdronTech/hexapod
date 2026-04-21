#!/usr/bin/env python3
"""Continuously display current positions of all 18 servos."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexapod.servo.transport import SerialTransport, TransportError
from hexapod.servo.protocol import ProtocolError
from hexapod.servo.st3020 import ST3020Bus
from hexapod.robot.config import Leg, Joint, servo_id

PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000

JOINT_ORDER = [Joint.COXA, Joint.FEMUR, Joint.TIBIA]
LEG_ORDER = [
    Leg.FRONT_RIGHT,
    Leg.MID_RIGHT,
    Leg.REAR_RIGHT,
    Leg.REAR_LEFT,
    Leg.MID_LEFT,
    Leg.FRONT_LEFT,
]

HEADER = f"{'Leg':<14} {'Coxa':>6} {'Femur':>6} {'Tibia':>6}"
SEPARATOR = "-" * len(HEADER)


def read_all(bus: ST3020Bus) -> dict[int, int | None]:
    positions: dict[int, int | None] = {}
    for leg in LEG_ORDER:
        for joint in JOINT_ORDER:
            sid = servo_id(leg, joint)
            try:
                positions[sid] = bus.read_position(sid)
            except (ProtocolError, TransportError):
                positions[sid] = None
    return positions


def render(positions: dict[int, int | None], elapsed: float) -> list[str]:
    lines = [HEADER, SEPARATOR]
    for leg in LEG_ORDER:
        vals = []
        for joint in JOINT_ORDER:
            sid = servo_id(leg, joint)
            pos = positions.get(sid)
            vals.append(f"{pos:>6d}" if pos is not None else "  FAIL")
        lines.append(f"{leg}: {leg.name:<14} {'  '.join(vals)}")
    lines.append(SEPARATOR)
    lines.append(f"Refresh: {elapsed * 1000:.0f} ms   (Ctrl-C to quit)")
    return lines


def main() -> None:
    print(f"Connecting to {PORT} at {BAUDRATE} baud …")
    num_lines = len(LEG_ORDER) + 4  # header + sep + rows + sep + status

    with SerialTransport(PORT, BAUDRATE) as transport:
        bus = ST3020Bus(transport)
        first = True
        while True:
            t0 = time.monotonic()
            positions = read_all(bus)
            elapsed = time.monotonic() - t0

            lines = render(positions, elapsed)

            if not first:
                # Move cursor up to overwrite previous output
                sys.stdout.write(f"\033[{num_lines}A")
            first = False

            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
