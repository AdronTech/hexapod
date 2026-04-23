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

from hexapod.control import (
    ControlThread,
    SharedState,
    apply_config,
    load_config,
    save_config,
    DEFAULT_CONFIG,
)
from hexapod.gait import _NEUTRAL_REACH

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_HTTP_HOST   = "0.0.0.0"
DEFAULT_HTTP_PORT   = 8080

CONFIG_PATH = Path(__file__).parent / "hexapod_config.json"

FREE_STEP_THRESHOLD = 3.0  # must match state.py default

# ---------------------------------------------------------------------------
# HTML (embedded so the script is self-contained)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Hexapod</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d1117; color: #c9d1d9;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px; padding: 0.4rem;
    height: 100vh; display: flex; flex-direction: column; gap: 0.25rem;
    overflow: hidden;
  }
  h2 { color: #8b949e; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }

  .badge {
    padding: 0.2rem 0.55rem; border-radius: 99px; font-size: 0.74rem; font-weight: 600;
    transition: background 0.2s, color 0.2s;
  }
  .off  { background: #21262d; color: #8b949e; }
  .ok   { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }
  .warn { background: #9e6a0333; color: #d29922; border: 1px solid #9e6a03; }
  .good { background: #1a7f3733; color: #3fb950; border: 1px solid #238636; }

  .panel {
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 0.45rem 0.65rem;
  }

  /* Header */
  #hdr { flex: 0 0 auto; }
  #hdr-top { display: flex; align-items: center; flex-wrap: wrap; gap: 0.3rem; }
  #hdr-top h1 { color: #58a6ff; font-size: 1.05rem; letter-spacing: 0.04em; margin-right: 0.15rem; }
  #msg-area { font-size: 0.76rem; color: #d29922; min-height: 1em; margin-top: 0.1rem; }

  /* 2-column main layout */
  #main {
    flex: 1 1 0; display: grid;
    grid-template-columns: 1fr 1.45fr;
    gap: 0.35rem; min-height: 0;
  }
  #col-left, #col-right {
    display: flex; flex-direction: column; gap: 0.35rem; min-height: 0;
  }

  /* Body pose */
  .pose-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.25rem;
    font-family: 'Consolas', 'Courier New', monospace;
  }
  .pose-item { text-align: center; }
  .pose-label { font-size: 0.63rem; color: #8b949e; }
  .pose-val   { font-size: 0.95rem; color: #79c0ff; }

  /* Controller */
  .sticks { display: flex; gap: 0.9rem; align-items: center; margin-bottom: 0.35rem; }
  .stick-wrap { text-align: center; }
  .stick-label { font-size: 0.63rem; color: #8b949e; margin-bottom: 0.15rem; }
  canvas { display: block; border-radius: 50%; }
  .btns { display: flex; flex-wrap: wrap; gap: 0.2rem; }
  .btn {
    padding: 0.12rem 0.4rem; border-radius: 4px; font-size: 0.66rem;
    background: #21262d; color: #8b949e;
  }
  .btn.on { background: #1a7f37; color: #aff3c8; }

  /* Slider controls */
  .ctrl-label { font-size: 0.69rem; color: #8b949e; margin-bottom: 0.1rem; }
  .ctrl-row { display: flex; align-items: center; gap: 0.3rem; margin-bottom: 0.2rem; }
  .ctrl-val { font-family: monospace; font-size: 0.95rem; color: #79c0ff; min-width: 3.2rem; text-align: center; }
  .spdbtn {
    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 4px; width: 1.7rem; height: 1.7rem; font-size: 0.95rem;
    cursor: pointer; line-height: 1; flex-shrink: 0;
  }
  .spdbtn:active { background: #1f6feb; }
  .bar-track {
    flex: 1; height: 7px; background: #21262d; border-radius: 4px;
    overflow: hidden; cursor: pointer; touch-action: none;
  }
  .bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    transition: width 0.1s ease;
  }

  /* Speed 2-col sub-grid */
  .speed-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }

  /* Gait buttons */
  .gait-btns { display: flex; gap: 0.3rem; margin-bottom: 0.3rem; }
  .gait-btn {
    flex: 1; padding: 0.22rem 0; border-radius: 4px; font-size: 0.76rem;
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    cursor: pointer; text-align: center;
  }
  .gait-btn.active { background: #1f6feb33; color: #58a6ff; border-color: #1f6feb; }

  /* Config buttons */
  .cfg-btns { display: flex; gap: 0.35rem; margin-top: 0.35rem; }
  .cfg-btn {
    flex: 1; padding: 0.28rem; border-radius: 4px; font-size: 0.76rem; font-weight: 600;
    cursor: pointer; border: 1px solid #30363d;
  }
  .cfg-save  { background: #1a7f3733; color: #3fb950; border-color: #238636; }
  .cfg-save:active  { background: #1a7f37; }
  .cfg-reset { background: #9e6a0333; color: #d29922; border-color: #9e6a03; }
  .cfg-reset:active { background: #9e6a03; color: #fff; }

  /* Collapsible controls reference */
  details.panel summary {
    cursor: pointer; color: #58a6ff; font-size: 0.69rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; list-style: none;
  }
  details.panel summary::before { content: '▶  '; }
  details[open].panel summary::before { content: '▼  '; }
  details.panel .ref-wrap { margin-top: 0.35rem; overflow-y: auto; max-height: 180px; }
  table.controls { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  table.controls td { padding: 0.12rem 0.35rem 0.12rem 0; color: #8b949e; }
  table.controls td:first-child { color: #c9d1d9; font-weight: 600; white-space: nowrap; }
  table.controls .sh td { color: #58a6ff; font-size: 0.65rem; padding-top: 0.35rem; }
</style>
</head>
<body>

<div id="hdr">
  <div id="hdr-top">
    <h1>&#129264; Hexapod</h1>
    <span class="badge off" id="b-ws">WS: …</span>
    <span class="badge off" id="b-gp">Controller: none</span>
    <span class="badge off" id="b-robot">Sitting</span>
    <span class="badge off" id="b-ik">IK 0</span>
    <button class="badge warn" onclick="sendCommand('store')" style="cursor:pointer;border:1px solid #9e6a03">&#9660; Store</button>
  </div>
  <div id="msg-area">
    <span id="msg"></span>
    <span id="msg-ik" style="font-size:0.68rem;color:#8b949e;margin-left:0.4rem"></span>
  </div>
</div>

<div id="main">

  <!-- LEFT: pose + controller -->
  <div id="col-left">

    <div class="panel">
      <h2>Body Pose</h2>
      <div class="pose-grid">
        <div class="pose-item"><div class="pose-label">X fwd</div><div class="pose-val" id="px">—</div></div>
        <div class="pose-item"><div class="pose-label">Y left</div><div class="pose-val" id="py">—</div></div>
        <div class="pose-item"><div class="pose-label">Z up</div><div class="pose-val" id="pz">—</div></div>
        <div class="pose-item"><div class="pose-label">Roll</div><div class="pose-val" id="pr">—</div></div>
        <div class="pose-item"><div class="pose-label">Pitch</div><div class="pose-val" id="pp">—</div></div>
        <div class="pose-item"><div class="pose-label">Yaw</div><div class="pose-val" id="pw">—</div></div>
      </div>
    </div>

    <div class="panel" style="flex:1 1 0;min-height:0">
      <h2>Controller</h2>
      <p id="hint" style="color:#d29922;font-size:0.76rem;margin-bottom:0.3rem">&#128269; Press any button to activate.</p>
      <div class="sticks">
        <div class="stick-wrap">
          <div class="stick-label">Left</div>
          <canvas id="ls" width="56" height="56"></canvas>
        </div>
        <div class="stick-wrap">
          <div class="stick-label">Right</div>
          <canvas id="rs" width="56" height="56"></canvas>
        </div>
        <div>
          <div class="stick-label" style="margin-bottom:0.25rem">Triggers</div>
          <div style="font-family:monospace;font-size:0.78rem;color:#79c0ff">
            LT <span id="lt-val">0.00</span><br>RT <span id="rt-val">0.00</span>
          </div>
        </div>
      </div>
      <div class="btns" id="btns"></div>
    </div>

  </div>

  <!-- RIGHT: speed + walk settings + controls ref -->
  <div id="col-right">

    <div class="panel">
      <h2>Speed</h2>
      <div class="speed-grid">
        <div>
          <div class="ctrl-label">Translate (cm/s) <span style="font-size:0.62rem">D-pad ↑↓</span></div>
          <div class="ctrl-row">
            <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustSpeed('cm',-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
            <span class="ctrl-val" id="spd-cm">15.0</span>
            <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustSpeed('cm',+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
            <div class="bar-track" id="track-cm"><div class="bar-fill" id="bar-cm" style="width:100%"></div></div>
          </div>
        </div>
        <div>
          <div class="ctrl-label">Rotate (°/s) <span style="font-size:0.62rem">D-pad ←→</span></div>
          <div class="ctrl-row">
            <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustSpeed('deg',-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
            <span class="ctrl-val" id="spd-deg">60.0</span>
            <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustSpeed('deg',+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
            <div class="bar-track" id="track-deg"><div class="bar-fill" id="bar-deg" style="width:100%"></div></div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel" style="flex:1 1 0;min-height:0">
      <h2>Walk Settings</h2>

      <div class="ctrl-label">Gait <span style="font-size:0.62rem">Walk: Back cycles</span></div>
      <div class="gait-btns">
        <button class="gait-btn" id="gait-tripod" onclick="selectGait('tripod')">Tripod</button>
        <button class="gait-btn" id="gait-ripple" onclick="selectGait('ripple')">Ripple</button>
        <button class="gait-btn" id="gait-wave"   onclick="selectGait('wave')">Wave</button>
      </div>

      <div class="ctrl-label">Foot Reach (cm) <span style="font-size:0.62rem">Walk: LB/RB</span></div>
      <div class="ctrl-row">
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustReach(-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
        <span class="ctrl-val" id="reach-val">17.4</span>
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustReach(+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
        <div class="bar-track" id="track-reach"><div class="bar-fill" id="bar-reach" style="width:39%"></div></div>
      </div>

      <div class="ctrl-label">Step Height (cm)</div>
      <div class="ctrl-row">
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepH(-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
        <span class="ctrl-val" id="step-h-val">4.0</span>
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepH(+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
        <div class="bar-track" id="track-step-h"><div class="bar-fill" id="bar-step-h" style="width:27%"></div></div>
      </div>

      <div class="ctrl-label">Step Duration (s)</div>
      <div class="ctrl-row">
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepT(-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
        <span class="ctrl-val" id="step-t-val">0.40</span>
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepT(+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
        <div class="bar-track" id="track-step-t"><div class="bar-fill" id="bar-step-t" style="width:29%"></div></div>
      </div>

      <div class="ctrl-label">Free Step Threshold (cm)</div>
      <div class="ctrl-row">
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepThr(-1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">−</button>
        <span class="ctrl-val" id="step-thr-val">3.0</span>
        <button class="spdbtn" onpointerdown="event.preventDefault();_pressStart(()=>adjustStepThr(+1))" onpointerup="_pressStop()" onpointerleave="_pressStop()">+</button>
        <div class="bar-track" id="track-step-thr"><div class="bar-fill" id="bar-step-thr" style="width:34%"></div></div>
      </div>

      <div class="cfg-btns">
        <button class="cfg-btn cfg-save"  onclick="sendCommand('save_config')">&#128190; Save Config</button>
        <button class="cfg-btn cfg-reset" onclick="sendCommand('reset_config')">&#8635; Reset Defaults</button>
      </div>
    </div>

    <details class="panel">
      <summary>Controls Reference</summary>
      <div class="ref-wrap">
        <table class="controls">
          <tr><td>A</td><td>Stand</td><td>B</td><td>Sit</td></tr>
          <tr><td>X</td><td>Toggle walk/pose</td><td>Y</td><td>Storage mode</td></tr>
          <tr><td>Back (standing)</td><td>Enter free mode</td><td>Back (free)</td><td>Exit free mode</td></tr>
          <tr><td>Start</td><td>Reset neutral</td><td></td><td></td></tr>
          <tr class="sh"><td colspan="4">POSE MODE</td></tr>
          <tr><td>Left stick</td><td>Translate X/Y</td><td>Right stick</td><td>Roll / Pitch</td></tr>
          <tr><td>LT / RT</td><td>Height</td><td>LB / RB</td><td>Yaw</td></tr>
          <tr class="sh"><td colspan="4">WALK MODE</td></tr>
          <tr><td>Left stick</td><td>Walk direction</td><td>Right stick X</td><td>Turn</td></tr>
          <tr><td>LT / RT</td><td>Height</td><td>LB / RB</td><td>Foot reach</td></tr>
          <tr><td>Back</td><td>Cycle gait</td><td>D-pad ↑↓</td><td>Speed ±cm/s</td></tr>
          <tr class="sh"><td colspan="4">FREE MODE</td></tr>
          <tr><td>Left stick</td><td>Walk (reactive)</td><td>Right stick</td><td>Roll / Pitch</td></tr>
          <tr><td>LT / RT</td><td>Height</td><td>LB / RB</td><td>Turn</td></tr>
        </table>
      </div>
    </details>

  </div>
</div>

<script>
const BTN_NAMES = ['A','B','X','Y','LB','RB','LT','RT','Back','Start','L3','R3','↑','↓','←','→','Home'];

let ws, wsOk = false;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen  = () => { wsOk = true;  badge('b-ws', 'WS: OK',  'ok'); };
  ws.onclose = () => { wsOk = false; badge('b-ws', 'WS: …',   'warn'); setTimeout(connect, 2000); };
  ws.onmessage = ev => updateStatus(JSON.parse(ev.data));
}

function badge(id, text, cls) {
  const el = document.getElementById(id);
  el.textContent = text; el.className = 'badge ' + cls;
}

let localSpeedCm = 15.0, localSpeedDeg = 60.0, localReach = 17.4;
let localStepH = 4.0, localStepT = 0.40, localStepThr = 3.0;
const STEP_CM = 0.5, STEP_DEG = 2.0, STEP_REACH = 0.5;
const MIN_CM = 0.5, MAX_CM = 30.0, MIN_DEG = 2.0, MAX_DEG = 120.0;
const MIN_REACH = 12.0, MAX_REACH = 26.0;
const STEP_H_STEP = 0.5, STEP_T_STEP = 0.05, STEP_THR_STEP = 0.25;
const STEP_H_MIN = 1.0, STEP_H_MAX = 12.0, STEP_T_MIN = 0.15, STEP_T_MAX = 1.0;
const STEP_THR_MIN = 0.5, STEP_THR_MAX = 8.0;

function sendCommand(cmd) {
  if (wsOk) ws.send(JSON.stringify({type:'command', cmd}));
}

function updateStatus(d) {
  if (d.busy)           badge('b-robot', 'Busy…',    'warn');
  else if (d.stored)    badge('b-robot', 'Stored',   'warn');
  else if (d.free_mode) badge('b-robot', 'Free',     'ok');
  else if (d.walk_mode) badge('b-robot', 'Walking',  'ok');
  else if (d.standing)  badge('b-robot', 'Standing', 'good');
  else                  badge('b-robot', 'Sitting',  'off');
  document.getElementById('msg').textContent = d.message || '';
  const p = d.pose;
  if (p && 'x' in p) {
    setText('px', p.x.toFixed(1));
    setText('py', p.y.toFixed(1));
    setText('pz', p.z.toFixed(1));
    setText('pr', p.roll.toFixed(1)  + '°');
    setText('pp', p.pitch.toFixed(1) + '°');
    setText('pw', p.yaw.toFixed(1)   + '°');
  } else { ['px','py','pz','pr','pp','pw'].forEach(id => setText(id, '—')); }
  if (d.speed_cm       !== undefined) { localSpeedCm  = d.speed_cm;       setSpeed('cm',  d.speed_cm); }
  if (d.speed_deg      !== undefined) { localSpeedDeg = d.speed_deg;      setSpeed('deg', d.speed_deg); }
  if (d.reach          !== undefined) { localReach    = d.reach;          setReach(d.reach); }
  if (d.step_height    !== undefined) { localStepH    = d.step_height;    setStepH(d.step_height); }
  if (d.step_time      !== undefined) { localStepT    = d.step_time;      setStepT(d.step_time); }
  if (d.step_threshold !== undefined) { localStepThr  = d.step_threshold; setStepThr(d.step_threshold); }
  if (d.gait_type !== undefined && d.gait_type !== localGait) { localGait = d.gait_type; setGait(d.gait_type); }
  if (d.ik_errors !== undefined) {
    const el = document.getElementById('b-ik');
    el.textContent = 'IK ' + d.ik_errors;
    el.className = 'badge ' + (d.ik_errors > 0 ? 'warn' : 'off');
    document.getElementById('msg-ik').textContent = d.last_ik_error || '';
  }
}

function setText(id, v) { document.getElementById(id).textContent = v; }

function setSpeed(axis, val) {
  if (axis === 'cm') {
    setText('spd-cm', val.toFixed(1));
    document.getElementById('bar-cm').style.width = ((val-MIN_CM)/(MAX_CM-MIN_CM)*100).toFixed(1)+'%';
  } else {
    setText('spd-deg', val.toFixed(1));
    document.getElementById('bar-deg').style.width = ((val-MIN_DEG)/(MAX_DEG-MIN_DEG)*100).toFixed(1)+'%';
  }
}
function setReach(v)   { setText('reach-val',    v.toFixed(1));  document.getElementById('bar-reach').style.width   = ((v-MIN_REACH)/(MAX_REACH-MIN_REACH)*100).toFixed(1)+'%'; }
function setStepH(v)   { setText('step-h-val',   v.toFixed(1));  document.getElementById('bar-step-h').style.width  = ((v-STEP_H_MIN)/(STEP_H_MAX-STEP_H_MIN)*100).toFixed(1)+'%'; }
function setStepT(v)   { setText('step-t-val',   v.toFixed(2));  document.getElementById('bar-step-t').style.width  = ((v-STEP_T_MIN)/(STEP_T_MAX-STEP_T_MIN)*100).toFixed(1)+'%'; }
function setStepThr(v) { setText('step-thr-val', v.toFixed(2));  document.getElementById('bar-step-thr').style.width = ((v-STEP_THR_MIN)/(STEP_THR_MAX-STEP_THR_MIN)*100).toFixed(1)+'%'; }

function adjustSpeed(axis, dir) {
  if (axis === 'cm') { localSpeedCm  = Math.max(MIN_CM,  Math.min(MAX_CM,  +(localSpeedCm  + dir*STEP_CM).toFixed(1)));  setSpeed('cm',  localSpeedCm); }
  else               { localSpeedDeg = Math.max(MIN_DEG, Math.min(MAX_DEG, +(localSpeedDeg + dir*STEP_DEG).toFixed(1))); setSpeed('deg', localSpeedDeg); }
  if (wsOk) ws.send(JSON.stringify({type:'speed', speed_cm:localSpeedCm, speed_deg:localSpeedDeg}));
}
function adjustReach(dir)    { localReach   = Math.max(MIN_REACH,  Math.min(MAX_REACH,  +(localReach   + dir*STEP_REACH).toFixed(1)));   setReach(localReach);    if (wsOk) ws.send(JSON.stringify({type:'reach',         reach:localReach})); }
function adjustStepH(dir)    { localStepH   = Math.max(STEP_H_MIN, Math.min(STEP_H_MAX, +(localStepH   + dir*STEP_H_STEP).toFixed(1)));  setStepH(localStepH);    if (wsOk) ws.send(JSON.stringify({type:'step_height',   value:localStepH})); }
function adjustStepT(dir)    { localStepT   = Math.max(STEP_T_MIN, Math.min(STEP_T_MAX, +(localStepT   + dir*STEP_T_STEP).toFixed(2)));  setStepT(localStepT);    if (wsOk) ws.send(JSON.stringify({type:'step_time',     value:localStepT})); }
function adjustStepThr(dir)  { localStepThr = Math.max(STEP_THR_MIN, Math.min(STEP_THR_MAX, +(localStepThr + dir*STEP_THR_STEP).toFixed(2))); setStepThr(localStepThr); if (wsOk) ws.send(JSON.stringify({type:'step_threshold', value:localStepThr})); }

let _pressTimer = null, _pressInterval = null;
function _pressStart(fn) { fn(); _pressTimer = setTimeout(() => { _pressInterval = setInterval(fn, 80); }, 450); }
function _pressStop()    { clearTimeout(_pressTimer); clearInterval(_pressInterval); _pressTimer = _pressInterval = null; }

function _makeDraggable(trackId, min, max, decimals, sender) {
  const track = document.getElementById(trackId);
  if (!track) return;
  let active = false;
  function fromPointer(e) {
    const rect = track.getBoundingClientRect();
    sender(+(min + Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width))*(max-min)).toFixed(decimals));
  }
  track.addEventListener('pointerdown', e => { e.preventDefault(); active=true; track.setPointerCapture(e.pointerId); fromPointer(e); });
  track.addEventListener('pointermove', e => { if (active) fromPointer(e); });
  track.addEventListener('pointerup',     () => { active=false; });
  track.addEventListener('pointercancel', () => { active=false; });
}

let localGait = 'tripod';
function selectGait(g) { localGait=g; setGait(g); if (wsOk) ws.send(JSON.stringify({type:'gait', gait:g})); }
function setGait(g) {
  ['tripod','ripple','wave'].forEach(n => {
    const el = document.getElementById('gait-'+n);
    if (el) el.className = 'gait-btn' + (n===g?' active':'');
  });
}

const seenIdx = new Set();
function activateGamepad(gp) {
  if (seenIdx.has(gp.index)) return;
  seenIdx.add(gp.index);
  badge('b-gp', gp.id.slice(0,26), 'ok');
  document.getElementById('hint').style.display = 'none';
  const wrap = document.getElementById('btns');
  wrap.innerHTML = '';
  BTN_NAMES.forEach((n,i) => {
    const d = document.createElement('span');
    d.className='btn'; d.id=`bn${i}`; d.textContent=n; wrap.appendChild(d);
  });
}
window.addEventListener('gamepadconnected',    e => activateGamepad(e.gamepad));
window.addEventListener('gamepaddisconnected', e => {
  seenIdx.delete(e.gamepad.index);
  if (!seenIdx.size) { badge('b-gp','Controller: none','off'); document.getElementById('hint').style.display=''; }
});

function drawStick(id, x, y) {
  const c=document.getElementById(id), ctx=c.getContext('2d');
  const cx=c.width/2, cy=c.height/2, r=cx-2;
  ctx.clearRect(0,0,c.width,c.height);
  ctx.strokeStyle='#21262d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI); ctx.stroke();
  ctx.strokeStyle='#30363d';
  [[cx-r,cy,cx+r,cy],[cx,cy-r,cx,cy+r]].forEach(([x1,y1,x2,y2])=>{ ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); });
  ctx.fillStyle='#58a6ff';
  ctx.beginPath(); ctx.arc(cx+x*r*0.88, cy+y*r*0.88, 4, 0, 2*Math.PI); ctx.fill();
}

let lastSend = 0;
function loop() {
  requestAnimationFrame(loop);
  const gps = navigator.getGamepads();
  let gp = null;
  for (const g of gps) { if (g) { gp=g; break; } }
  if (!gp) return;
  activateGamepad(gp);
  drawStick('ls', gp.axes[0]||0, gp.axes[1]||0);
  drawStick('rs', gp.axes[2]||0, gp.axes[3]||0);
  setText('lt-val', (gp.buttons[6]?.value||0).toFixed(2));
  setText('rt-val', (gp.buttons[7]?.value||0).toFixed(2));
  BTN_NAMES.forEach((_,i) => {
    const el=document.getElementById(`bn${i}`);
    if (el) el.className='btn'+(gp.buttons[i]?.pressed?' on':'');
  });
  const now=performance.now();
  if (now-lastSend < 1000/30 || !wsOk) return;
  lastSend=now;
  ws.send(JSON.stringify({axes:Array.from(gp.axes), buttons:Array.from(gp.buttons,b=>b.value), connected:true}));
}

connect();
setGait('tripod');
function _sendSpeeds() { if (wsOk) ws.send(JSON.stringify({type:'speed', speed_cm:localSpeedCm, speed_deg:localSpeedDeg})); }
_makeDraggable('track-cm',       MIN_CM,       MAX_CM,       1, v=>{localSpeedCm=v;  setSpeed('cm',v);   _sendSpeeds();});
_makeDraggable('track-deg',      MIN_DEG,      MAX_DEG,      1, v=>{localSpeedDeg=v; setSpeed('deg',v);  _sendSpeeds();});
_makeDraggable('track-reach',    MIN_REACH,    MAX_REACH,    1, v=>{localReach=v;    setReach(v);    if(wsOk)ws.send(JSON.stringify({type:'reach',         reach:v}));});
_makeDraggable('track-step-h',   STEP_H_MIN,   STEP_H_MAX,   1, v=>{localStepH=v;   setStepH(v);   if(wsOk)ws.send(JSON.stringify({type:'step_height',   value:v}));});
_makeDraggable('track-step-t',   STEP_T_MIN,   STEP_T_MAX,   2, v=>{localStepT=v;   setStepT(v);   if(wsOk)ws.send(JSON.stringify({type:'step_time',     value:v}));});
_makeDraggable('track-step-thr', STEP_THR_MIN, STEP_THR_MAX, 2, v=>{localStepThr=v; setStepThr(v); if(wsOk)ws.send(JSON.stringify({type:'step_threshold', value:v}));});
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(shared: SharedState) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        shared.set_gamepad([], [], False)

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        async def send_loop() -> None:
            try:
                while True:
                    await ws.send_text(json.dumps(shared.get_status()))
                    await asyncio.sleep(0.1)
            except Exception:
                pass

        send_task = asyncio.create_task(send_loop())
        try:
            async for raw in ws.iter_text():
                try:
                    data = json.loads(raw)
                    if data.get("type") == "speed":
                        sc, sd = shared.get_speeds()
                        shared.set_speeds(
                            data.get("speed_cm",  sc),
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
                        shared.set_step_threshold(data.get("value", FREE_STEP_THRESHOLD))
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
    parser.add_argument("--port",      default=DEFAULT_SERIAL_PORT, help="Serial port")
    parser.add_argument("--bind",      default=DEFAULT_HTTP_HOST,   help="HTTP bind address")
    parser.add_argument("--http-port", default=DEFAULT_HTTP_PORT, type=int, help="HTTP port")
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
