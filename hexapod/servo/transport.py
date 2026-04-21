import serial


class TransportError(Exception):
    pass


class SerialTransport:
    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.1):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial: serial.Serial | None = None

    def open(self) -> None:
        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self._timeout,
        )
        self._serial.reset_input_buffer()

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()

    def __enter__(self) -> "SerialTransport":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def send(self, data: bytes) -> None:
        if not self._serial or not self._serial.is_open:
            raise TransportError("Port is not open")
        self._serial.reset_input_buffer()
        self._serial.write(data)
        self._serial.flush()

    def receive(self, n: int) -> bytes:
        if not self._serial or not self._serial.is_open:
            raise TransportError("Port is not open")
        data = self._serial.read(n)
        if len(data) != n:
            raise TransportError(f"Timeout: expected {n} bytes, received {len(data)}")
        return data

    @property
    def port(self) -> str:
        return self._port

    @property
    def baudrate(self) -> int:
        return self._baudrate
