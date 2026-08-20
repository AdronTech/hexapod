'use strict';

const BTN_NAMES = ['A','B','X','Y','LB','RB','LT','RT','Back','Start','L3','R3','↑','↓','←','→','Home'];

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

let ws, wsOk = false;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen  = () => { wsOk = true;  badge('b-ws', 'WS: OK',  'ok'); };
  ws.onclose = () => { wsOk = false; badge('b-ws', 'WS: …',   'warn'); setTimeout(connect, 2000); };
  ws.onmessage = ev => updateStatus(JSON.parse(ev.data));
}

function send(msg) { if (wsOk) ws.send(JSON.stringify(msg)); }
function sendCommand(cmd) { send({type:'command', cmd}); }

function badge(id, text, cls) {
  const el = document.getElementById(id);
  el.textContent = text; el.className = 'badge ' + cls;
}

function setText(id, v) { document.getElementById(id).textContent = v; }

// ---------------------------------------------------------------------------
// Numeric controls (± buttons + draggable bar)
//
// Each entry maps a status field to a row in the DOM marked with
// data-ctrl="<key>"; the row supplies the ± buttons, the value readout and the
// bar. `send` turns a new value into the websocket message for that control.
// ---------------------------------------------------------------------------

function sendSpeeds() {
  send({type:'speed', speed_cm:CONTROLS.speed_cm.value, speed_deg:CONTROLS.speed_deg.value});
}

const CONTROLS = {
  speed_cm:       {value: 15.0, min:  0.5, max:  30.0, step: 0.5,  decimals: 1, send: sendSpeeds},
  speed_deg:      {value: 60.0, min:  2.0, max: 120.0, step: 2.0,  decimals: 1, send: sendSpeeds},
  reach:          {value: 17.4, min: 12.0, max:  26.0, step: 0.5,  decimals: 1, send: v => send({type:'reach', reach:v})},
  step_height:    {value:  4.0, min:  1.0, max:  12.0, step: 0.5,  decimals: 1, send: v => send({type:'step_height', value:v})},
  step_time:      {value: 0.40, min: 0.15, max:   1.0, step: 0.05, decimals: 2, send: v => send({type:'step_time', value:v})},
  step_threshold: {value:  3.0, min:  0.5, max:   8.0, step: 0.25, decimals: 2, send: v => send({type:'step_threshold', value:v})},
};

function renderControl(key) {
  const c = CONTROLS[key];
  if (!c.el) return;
  c.el.value.textContent = c.value.toFixed(c.decimals);
  c.el.bar.style.width = ((c.value - c.min) / (c.max - c.min) * 100).toFixed(1) + '%';
}

/** Clamp, store and render — without notifying the server. */
function setControl(key, value) {
  const c = CONTROLS[key];
  c.value = +Math.max(c.min, Math.min(c.max, value)).toFixed(c.decimals);
  renderControl(key);
}

/** Set from a user interaction: render and push to the server. */
function changeControl(key, value) {
  setControl(key, value);
  CONTROLS[key].send(CONTROLS[key].value);
}

function initControls() {
  for (const [key, c] of Object.entries(CONTROLS)) {
    const row = document.querySelector(`[data-ctrl="${key}"]`);
    if (!row) continue;
    c.el = {
      value: row.querySelector('[data-role="value"]'),
      bar:   row.querySelector('[data-role="bar"]'),
      track: row.querySelector('[data-role="track"]'),
    };
    row.querySelectorAll('[data-dir]').forEach(btn => {
      const dir = +btn.dataset.dir;
      btn.addEventListener('pointerdown', e => {
        e.preventDefault();
        pressStart(() => changeControl(key, c.value + dir * c.step));
      });
      btn.addEventListener('pointerup', pressStop);
      btn.addEventListener('pointerleave', pressStop);
    });
    makeDraggable(c.el.track, key);
    renderControl(key);
  }
}

// Auto-repeat while a ± button is held down.
let pressTimer = null, pressInterval = null;
function pressStart(fn) { fn(); pressTimer = setTimeout(() => { pressInterval = setInterval(fn, 80); }, 450); }
function pressStop()    { clearTimeout(pressTimer); clearInterval(pressInterval); pressTimer = pressInterval = null; }

function makeDraggable(track, key) {
  if (!track) return;
  const c = CONTROLS[key];
  let active = false;
  function fromPointer(e) {
    const rect = track.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    changeControl(key, c.min + frac * (c.max - c.min));
  }
  track.addEventListener('pointerdown', e => { e.preventDefault(); active = true; track.setPointerCapture(e.pointerId); fromPointer(e); });
  track.addEventListener('pointermove', e => { if (active) fromPointer(e); });
  track.addEventListener('pointerup',     () => { active = false; });
  track.addEventListener('pointercancel', () => { active = false; });
}

// ---------------------------------------------------------------------------
// Gait selection
// ---------------------------------------------------------------------------

let localGait = 'tripod';

function selectGait(g) { localGait = g; setGait(g); send({type:'gait', gait:g}); }

function setGait(g) {
  document.querySelectorAll('[data-gait]').forEach(el => {
    el.className = 'gait-btn' + (el.dataset.gait === g ? ' active' : '');
  });
}

// ---------------------------------------------------------------------------
// Status updates
// ---------------------------------------------------------------------------

function updateStatus(d) {
  if (d.busy)           badge('b-robot', 'Busy…',    'warn');
  else if (d.stored)    badge('b-robot', 'Stored',   'warn');
  else if (d.free_mode) badge('b-robot', 'Free',     'ok');
  else if (d.walk_mode) badge('b-robot', 'Walking',  'ok');
  else if (d.standing)  badge('b-robot', 'Standing', 'good');
  else                  badge('b-robot', 'Sitting',  'off');
  setText('msg', d.message || '');

  const p = d.pose;
  if (p && 'x' in p) {
    setText('px', p.x.toFixed(1));
    setText('py', p.y.toFixed(1));
    setText('pz', p.z.toFixed(1));
    setText('pr', p.roll.toFixed(1)  + '°');
    setText('pp', p.pitch.toFixed(1) + '°');
    setText('pw', p.yaw.toFixed(1)   + '°');
  } else {
    ['px','py','pz','pr','pp','pw'].forEach(id => setText(id, '—'));
  }

  for (const key of Object.keys(CONTROLS)) {
    if (d[key] !== undefined) setControl(key, d[key]);
  }

  if (d.gait_type !== undefined && d.gait_type !== localGait) { localGait = d.gait_type; setGait(d.gait_type); }

  if (d.ik_errors !== undefined) {
    const el = document.getElementById('b-ik');
    el.textContent = 'IK ' + d.ik_errors;
    el.className = 'badge ' + (d.ik_errors > 0 ? 'warn' : 'off');
    setText('msg-ik', d.last_ik_error || '');
  }
}

// ---------------------------------------------------------------------------
// Gamepad
// ---------------------------------------------------------------------------

const seenIdx = new Set();

function activateGamepad(gp) {
  if (seenIdx.has(gp.index)) return;
  seenIdx.add(gp.index);
  badge('b-gp', gp.id.slice(0, 26), 'ok');
  document.getElementById('hint').style.display = 'none';
  const wrap = document.getElementById('btns');
  wrap.innerHTML = '';
  BTN_NAMES.forEach((n, i) => {
    const d = document.createElement('span');
    d.className = 'btn'; d.id = `bn${i}`; d.textContent = n; wrap.appendChild(d);
  });
}

window.addEventListener('gamepadconnected', e => activateGamepad(e.gamepad));
window.addEventListener('gamepaddisconnected', e => {
  seenIdx.delete(e.gamepad.index);
  if (!seenIdx.size) { badge('b-gp', 'Controller: none', 'off'); document.getElementById('hint').style.display = ''; }
});

function drawStick(id, x, y) {
  const c = document.getElementById(id), ctx = c.getContext('2d');
  const cx = c.width / 2, cy = c.height / 2, r = cx - 2;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = '#30363d';
  [[cx-r,cy,cx+r,cy],[cx,cy-r,cx,cy+r]].forEach(([x1,y1,x2,y2]) => { ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); });
  ctx.fillStyle = '#58a6ff';
  ctx.beginPath(); ctx.arc(cx + x*r*0.88, cy + y*r*0.88, 4, 0, 2*Math.PI); ctx.fill();
}

let lastSend = 0;
function loop() {
  requestAnimationFrame(loop);
  const gps = navigator.getGamepads();
  let gp = null;
  for (const g of gps) { if (g) { gp = g; break; } }
  if (!gp) return;
  activateGamepad(gp);
  drawStick('ls', gp.axes[0]||0, gp.axes[1]||0);
  drawStick('rs', gp.axes[2]||0, gp.axes[3]||0);
  setText('lt-val', (gp.buttons[6]?.value||0).toFixed(2));
  setText('rt-val', (gp.buttons[7]?.value||0).toFixed(2));
  BTN_NAMES.forEach((_, i) => {
    const el = document.getElementById(`bn${i}`);
    if (el) el.className = 'btn' + (gp.buttons[i]?.pressed ? ' on' : '');
  });
  const now = performance.now();
  if (now - lastSend < 1000/30 || !wsOk) return;
  lastSend = now;
  send({axes: Array.from(gp.axes), buttons: Array.from(gp.buttons, b => b.value), connected: true});
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.querySelectorAll('[data-command]').forEach(el => {
  el.addEventListener('click', () => sendCommand(el.dataset.command));
});
document.querySelectorAll('[data-gait]').forEach(el => {
  el.addEventListener('click', () => selectGait(el.dataset.gait));
});

initControls();
setGait(localGait);
connect();
requestAnimationFrame(loop);
