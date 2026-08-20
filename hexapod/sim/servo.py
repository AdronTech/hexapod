"""
Virtual ST3020 servo: a register file plus a simple motion model.

The register map and the semantics of the special registers mirror
hexapod/servo/registers.py, so the real bus driver talks to this model
without knowing the difference.

Motion model — trapezoidal velocity profile:
  * GOAL_SPEED (0 = unlimited) caps the velocity, in ticks/s
  * ACC (register unit = 100 ticks/s², 0 = unlimited) caps acceleration
  * the servo decelerates in time to stop on the goal position
  * with torque disabled the joint holds still and reports no motion
"""

from hexapod.servo import registers as R
from hexapod.servo.protocol import decode_signed, decode_word, encode_signed

TICK_MIN = 0
TICK_MAX = 4095
CENTER_TICK = 2048

# ST3020: ~0.18 s / 60° at 7.4 V ≈ 333 °/s ≈ 3800 ticks/s
DEFAULT_MAX_SPEED = 3800.0  # ticks/s
ACC_UNIT = 100.0  # ACC register unit, ticks/s²
UNLIMITED_ACC = 1.0e6  # ticks/s², stand-in for "no acceleration limit"

NOMINAL_VOLTAGE = 7.4  # V
AMBIENT_TEMP = 25.0  # °C
TEMP_RISE_PER_LOAD = 30.0  # °C at sustained full load
TEMP_TIME_CONSTANT = 45.0  # s


class VirtualServo:
    """One emulated servo on the virtual bus."""

    def __init__(
        self,
        servo_id: int,
        position: int = CENTER_TICK,
        max_speed: float = DEFAULT_MAX_SPEED,
    ) -> None:
        self.regs = bytearray(256)
        self.pos = float(position)  # reported (post-offset) position, ticks
        self.vel = 0.0  # ticks/s, signed
        self.ofs = 0  # calibration offset, ticks
        self.temperature = AMBIENT_TEMP
        self._max_speed = max_speed

        self.regs[R.MODEL_L], self.regs[R.MODEL_H] = 0x09, 0x03
        self.regs[R.ID] = servo_id
        self.regs[R.BAUD_RATE] = R.BAUD_1M
        self.regs[R.MODE] = 0  # position servo mode
        self.regs[R.TORQUE_ENABLE] = 1
        self._set_word(R.GOAL_POS_L, position)
        self._refresh_feedback()

    # --- properties backed by the register file ---

    @property
    def servo_id(self) -> int:
        return self.regs[R.ID]

    @property
    def torque_on(self) -> bool:
        return self.regs[R.TORQUE_ENABLE] == 1

    @property
    def goal(self) -> float:
        return float(self._get_word(R.GOAL_POS_L))

    @property
    def speed_limit(self) -> float:
        """Velocity cap in ticks/s; the register value 0 means unlimited."""
        limit = self._get_word(R.GOAL_SPEED_L)
        return self._max_speed if limit == 0 else min(float(limit), self._max_speed)

    @property
    def acceleration(self) -> float:
        """Acceleration cap in ticks/s²; the register value 0 means unlimited."""
        acc = self.regs[R.ACC]
        return UNLIMITED_ACC if acc == 0 else acc * ACC_UNIT

    @property
    def moving(self) -> bool:
        return abs(self.goal - self.pos) > 1.0 or abs(self.vel) > 1.0

    # --- motion model ---

    def step(self, dt: float) -> None:
        if not self.torque_on:
            self.vel = 0.0
            self._update_thermal(dt, load_frac=0.0)
            return

        remaining = self.goal - self.pos
        a_max = self.acceleration
        v_max = self.speed_limit

        # Distance needed to bleed the current speed off to zero
        stopping = self.vel * self.vel / (2.0 * a_max)
        direction = 1.0 if remaining > 0 else -1.0

        if abs(remaining) <= stopping and abs(self.vel) > 0.0:
            accel = -a_max * (1.0 if self.vel > 0 else -1.0)  # brake
        else:
            accel = a_max * direction

        self.vel += accel * dt
        self.vel = max(-v_max, min(v_max, self.vel))

        step = self.vel * dt
        if abs(step) >= abs(remaining):  # would overshoot: land exactly
            self.pos = self.goal
            self.vel = 0.0
        else:
            self.pos += step

        if self.pos < TICK_MIN or self.pos > TICK_MAX:  # hit the encoder wrap
            self.pos = max(float(TICK_MIN), min(float(TICK_MAX), self.pos))
            self.vel = 0.0

        self._update_thermal(dt, load_frac=abs(self.vel) / self._max_speed)

    def _update_thermal(self, dt: float, load_frac: float) -> None:
        target = AMBIENT_TEMP + TEMP_RISE_PER_LOAD * load_frac
        alpha = min(1.0, dt / TEMP_TIME_CONSTANT)
        self.temperature += (target - self.temperature) * alpha

    # --- register access ---

    def read(self, addr: int, count: int) -> bytes:
        self._refresh_feedback()
        end = min(addr + count, len(self.regs))
        data = bytes(self.regs[addr:end])
        return data.ljust(count, b"\x00")

    def write(self, addr: int, data: bytes) -> None:
        if addr == R.TORQUE_ENABLE and data and data[0] == 128:
            self._calibrate_middle()
            return

        for i, byte in enumerate(data):
            if addr + i < len(self.regs):
                self.regs[addr + i] = byte

        if addr <= R.OFS_H < addr + len(data) or addr <= R.OFS_L < addr + len(data):
            self._apply_offset()
        if addr <= R.GOAL_POS_L < addr + len(data):
            goal = max(TICK_MIN, min(TICK_MAX, self._get_word(R.GOAL_POS_L)))
            self._set_word(R.GOAL_POS_L, goal)

    def _calibrate_middle(self) -> None:
        """Writing 128 to TORQUE_ENABLE: make the current position read 2048."""
        self.ofs += round(self.pos) - CENTER_TICK
        self._set_word(R.OFS_L, encode_signed(self.ofs))
        self.pos = float(CENTER_TICK)
        self.vel = 0.0
        self._set_word(R.GOAL_POS_L, CENTER_TICK)
        self.regs[R.TORQUE_ENABLE] = 1

    def _apply_offset(self) -> None:
        """An OFS write shifts the reported position by the same amount."""
        new_ofs = decode_signed(self._get_word(R.OFS_L))
        self.pos -= new_ofs - self.ofs
        self.ofs = new_ofs

    def _refresh_feedback(self) -> None:
        self._set_word(R.PRESENT_POS_L, round(self.pos) & 0xFFFF)
        self._set_word(R.PRESENT_SPD_L, encode_signed(round(self.vel)))

        load_frac = abs(self.vel) / self._max_speed if self.torque_on else 0.0
        load = round(min(1.0, load_frac) * 1000)
        self._set_word(
            R.PRESENT_LOAD_L, encode_signed(load if self.vel >= 0 else -load)
        )

        self.regs[R.PRESENT_VOLTAGE] = round((NOMINAL_VOLTAGE - 0.3 * load_frac) * 10)
        self.regs[R.PRESENT_TEMP] = round(self.temperature)
        self.regs[R.MOVING] = int(self.moving)

    def _get_word(self, addr: int) -> int:
        return decode_word(self.regs[addr], self.regs[addr + 1])

    def _set_word(self, addr: int, value: int) -> None:
        self.regs[addr] = value & 0xFF
        self.regs[addr + 1] = (value >> 8) & 0xFF
