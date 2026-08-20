"""
Virtual hexapod: an emulated ST3020 servo bus behind a virtual serial port.

Lets the whole control stack (web controller, leg tests, calibration scripts)
run without hardware attached — see scripts/simulator.py.
"""

from .bus import VirtualBus
from .port import VirtualSerialPort
from .runtime import SimulatorThread
from .servo import VirtualServo

__all__ = ["SimulatorThread", "VirtualBus", "VirtualSerialPort", "VirtualServo"]
