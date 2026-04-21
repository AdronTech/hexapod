from dataclasses import dataclass

from .transport import SerialTransport, TransportError
from .protocol import (
    ProtocolError,
    StatusPacket,
    build_packet,
    build_sync_write,
    parse_status_packet,
    decode_word,
    decode_signed,
    encode_signed,
    encode_word,
)
from . import registers as R


@dataclass
class ServoFeedback:
    servo_id: int
    position: int    # 0–4095 raw ticks
    speed: int       # signed ticks/s, negative = reverse direction
    load: int        # signed, 0–1000 (1000 = 100% max load)
    voltage: float   # volts
    temperature: int # °C
    moving: bool


@dataclass
class PositionCommand:
    servo_id: int
    position: int    # 0–4095
    speed: int = 300
    acc: int = 50


class ST3020Bus:
    def __init__(self, transport: SerialTransport):
        self._t = transport

    def ping(self, servo_id: int) -> bool:
        packet = build_packet(servo_id, R.INST_PING, [])
        self._t.send(packet)
        try:
            raw = self._t.receive(6)  # FF FF ID 02 ERROR CHECKSUM
            status = parse_status_packet(raw)
            return status.servo_id == servo_id
        except (ProtocolError, TransportError):
            return False

    def read_position(self, servo_id: int) -> int:
        packet = build_packet(servo_id, R.INST_READ, [R.PRESENT_POS_L, 2])
        self._t.send(packet)
        raw = self._t.receive(8)  # 6 header bytes + 2 data bytes
        status = parse_status_packet(raw)
        return decode_word(status.data[0], status.data[1])

    def read_feedback(self, servo_id: int) -> ServoFeedback:
        # Read 11 bytes starting at PRESENT_POS_L (56):
        #   [0,1]  present position  (56,57)
        #   [2,3]  present speed     (58,59)
        #   [4,5]  present load      (60,61)
        #   [6]    voltage           (62)
        #   [7]    temperature       (63)
        #   [8,9]  reserved          (64,65)
        #   [10]   moving            (66)
        packet = build_packet(servo_id, R.INST_READ, [R.PRESENT_POS_L, 11])
        self._t.send(packet)
        raw = self._t.receive(17)  # 6 header bytes + 11 data bytes
        status = parse_status_packet(raw)
        d = status.data
        return ServoFeedback(
            servo_id=servo_id,
            position=decode_word(d[0], d[1]),
            speed=decode_signed(decode_word(d[2], d[3])),
            load=decode_signed(decode_word(d[4], d[5])),
            voltage=d[6] / 10.0,
            temperature=d[7],
            moving=bool(d[10]),
        )

    def write_position(
        self,
        servo_id: int,
        position: int,
        speed: int = 300,
        acc: int = 50,
    ) -> None:
        # Writes 7 bytes starting at ACC (41):
        #   acc, pos_l, pos_h, time_l(0), time_h(0), spd_l, spd_h
        pos_l, pos_h = encode_word(position)
        spd_l, spd_h = encode_word(speed)
        params = [R.ACC, acc, pos_l, pos_h, 0, 0, spd_l, spd_h]
        packet = build_packet(servo_id, R.INST_WRITE, params)
        self._t.send(packet)
        self._t.receive(6)  # consume ACK

    def torque_enable(self, servo_id: int, enable: bool) -> None:
        self._write_byte(servo_id, R.TORQUE_ENABLE, int(enable))

    def set_middle_position(self, servo_id: int) -> None:
        """Mark the current physical position as the center (tick 2048).

        Writing 128 to the torque switch register instructs the firmware to
        compute and store the OFS so the current position reads as 2048.
        Goal position is then updated to 2048 so the servo does not drift
        after the position feedback shifts.
        """
        self._write_byte(servo_id, R.TORQUE_ENABLE, 128)
        self.write_position(servo_id, 2048, speed=50, acc=10)
        self._write_byte(servo_id, R.TORQUE_ENABLE, 1)

    def _write_byte(self, servo_id: int, reg: int, value: int) -> None:
        packet = build_packet(servo_id, R.INST_WRITE, [reg, value & 0xFF])
        self._t.send(packet)
        self._t.receive(6)

    def sync_write_position(self, commands: list[PositionCommand]) -> None:
        servo_data = []
        for cmd in commands:
            pos_l, pos_h = encode_word(cmd.position)
            spd_l, spd_h = encode_word(cmd.speed)
            servo_data.append(
                (cmd.servo_id, [cmd.acc, pos_l, pos_h, 0, 0, spd_l, spd_h])
            )
        packet = build_sync_write(R.ACC, 7, servo_data)
        self._t.send(packet)
        # Sync write is a broadcast — no response
