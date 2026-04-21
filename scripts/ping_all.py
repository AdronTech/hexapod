#!/usr/bin/env python3
"""Ping all 18 servos and report which ones respond."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexapod.servo.transport import SerialTransport
from hexapod.servo.st3020 import ST3020Bus
from hexapod.robot.config import Leg, Joint, LEGS, servo_id

PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000


def main() -> None:
    found = 0
    missing: list[int] = []

    with SerialTransport(PORT, BAUDRATE) as transport:
        bus = ST3020Bus(transport)

        for leg in Leg:
            print(f"Leg {leg.value} ({leg.name}):")
            for joint in Joint:
                sid = servo_id(leg, joint)
                ok = bus.ping(sid)
                status = "OK" if ok else "NO RESPONSE"
                print(f"  ID {sid:3d} ({joint.name:<5}): {status}")
                if ok:
                    found += 1
                else:
                    missing.append(sid)

    print(f"\nFound {found}/18 servos.")
    if missing:
        print(f"Missing IDs: {missing}")


if __name__ == "__main__":
    main()
