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
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hexapod.control import (
    DEFAULT_CONFIG,
    ControlThread,
    SharedState,
    apply_config,
    load_config,
    save_config,
)
from hexapod.gait import _NEUTRAL_REACH

DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8080

CONFIG_PATH = Path(__file__).parent / "hexapod_config.json"
WEB_DIR = Path(__file__).parent / "web"

FREE_STEP_THRESHOLD = 3.0  # must match state.py default

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def build_app(shared: SharedState) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        shared.set_gamepad([], [], False)

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        async def send_loop() -> None:
            try:
                while True:
                    await ws.send_text(json.dumps(shared.get_status()))
                    await asyncio.sleep(0.1)
            except (WebSocketDisconnect, ConnectionError, RuntimeError):
                pass

        send_task = asyncio.create_task(send_loop())
        try:
            async for raw in ws.iter_text():
                try:
                    data = json.loads(raw)
                    if data.get("type") == "speed":
                        sc, sd = shared.get_speeds()
                        shared.set_speeds(
                            data.get("speed_cm", sc),
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
                        shared.set_step_threshold(
                            data.get("value", FREE_STEP_THRESHOLD)
                        )
                    elif data.get("type") == "command":
                        cmd = data.get("cmd", "")
                        if cmd == "save_config":
                            save_config(shared, CONFIG_PATH)
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
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT, help="Serial port")
    parser.add_argument("--bind", default=DEFAULT_HTTP_HOST, help="HTTP bind address")
    parser.add_argument(
        "--http-port", default=DEFAULT_HTTP_PORT, type=int, help="HTTP port"
    )
    args = parser.parse_args()

    shared = SharedState()
    apply_config(load_config(CONFIG_PATH), shared)
    ctrl = ControlThread(args.port, shared)
    ctrl.start()

    app = build_app(shared)
    print(f"Open http://{args.bind}:{args.http_port} in your browser.")
    try:
        uvicorn.run(app, host=args.bind, port=args.http_port, log_level="warning")
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
