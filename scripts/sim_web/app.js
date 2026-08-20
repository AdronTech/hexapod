"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let snap = null;
let trail = [];          // world-frame path the body has walked
let packetRate = 0;
let lastPackets = null;
let lastRateAt = 0;

const LEG_NAMES = {
  1: "L1 front R", 2: "L2 mid R", 3: "L3 rear R",
  4: "L4 rear L", 5: "L5 mid L", 6: "L6 front L",
};

const COLORS = {
  bg: "#0d1117",
  grid: "#1b222c",
  gridAxis: "#2b3644",
  body: "#1f6feb",
  bodyFill: "rgba(31,111,235,0.28)",
  coxa: "#58a6ff",
  femur: "#79c0ff",
  tibia: "#a5d6ff",
  footDown: "#3fb950",
  footUp: "#d29922",
  shadow: "rgba(0,0,0,0.45)",
  trail: "rgba(88,166,255,0.55)",
  origin: "#30363d",
  heading: "rgba(165,214,255,0.8)",
  support: "rgba(63,185,80,0.10)",
  supportEdge: "rgba(63,185,80,0.45)",
  supportBad: "rgba(248,81,73,0.12)",
  supportBadEdge: "rgba(248,81,73,0.5)",
  label: "#6e7681",
};

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

const HOME = { az: 2.5, el: 0.45, dist: 95 };
const cam = { ...HOME, target: [0, 0, 8] };

// The camera tracks the robot so it stays centred while the world slides past.
function followRobot() {
  if (!snap || !snap.robot || !snap.robot.world) return;
  const w = snap.robot.world;
  cam.target[0] += (w.x - cam.target[0]) * 0.12;
  cam.target[1] += (w.y - cam.target[1]) * 0.12;
}

function eye() {
  const ce = Math.cos(cam.el), se = Math.sin(cam.el);
  return [
    cam.target[0] + cam.dist * ce * Math.cos(cam.az),
    cam.target[1] + cam.dist * ce * Math.sin(cam.az),
    cam.target[2] + cam.dist * se,
  ];
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

// Returns {x, y, depth} in CSS pixels, or null when behind the camera.
function project(p, view) {
  const v = sub(p, view.eye);
  const z = dot(v, view.fwd);
  if (z < 1) return null;
  return {
    x: view.cx + view.f * dot(v, view.right) / z,
    y: view.cy - view.f * dot(v, view.up) / z,
    depth: z,
  };
}

function makeView(w, h) {
  const e = eye();
  const fwd = norm(sub(cam.target, e));
  const right = norm(cross(fwd, [0, 0, 1]));
  const up = cross(right, fwd);
  return { eye: e, fwd, right, up, cx: w / 2, cy: h / 2, f: h * 1.05 };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);

function line(a, b, color, width) {
  if (!a || !b) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function disc(p, r, color) {
  if (!p) return;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fill();
}

function polygon(points, fill, stroke) {
  const pts = points.filter(Boolean);
  if (pts.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
}

function drawGrid(view) {
  // Grid lines live in the world and are drawn around wherever the camera
  // looks, so walking visibly moves the robot across them.
  const extent = 60, step = 5;
  const bx = Math.round(cam.target[0] / step) * step;
  const by = Math.round(cam.target[1] / step) * step;
  for (let i = -extent; i <= extent; i += step) {
    const x = bx + i, y = by + i;
    line(project([x, by - extent, 0], view), project([x, by + extent, 0], view),
         x === 0 ? COLORS.gridAxis : COLORS.grid, x === 0 ? 1.4 : 1);
    line(project([bx - extent, y, 0], view), project([bx + extent, y, 0], view),
         y === 0 ? COLORS.gridAxis : COLORS.grid, y === 0 ? 1.4 : 1);
  }

  // World origin — the fixed landmark the robot walks away from
  const o = project([0, 0, 0], view);
  const ox = project([6, 0, 0], view);
  const oy = project([0, 6, 0], view);
  line(o, ox, COLORS.origin, 2);
  line(o, oy, COLORS.origin, 2);
  if (ox) {
    ctx.fillStyle = COLORS.label;
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("0,0", ox.x + 8, ox.y + 4);
  }
}

function drawTrail(view) {
  if (trail.length < 2) return;
  ctx.lineWidth = 1.5;
  ctx.lineCap = "round";
  for (let i = 1; i < trail.length; i++) {
    const a = project([trail[i - 1][0], trail[i - 1][1], 0], view);
    const b = project([trail[i][0], trail[i][1], 0], view);
    if (!a || !b) continue;
    ctx.globalAlpha = i / trail.length;
    ctx.strokeStyle = COLORS.trail;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawSupport(view, robot) {
  const feet = robot.legs.filter((l) => l.grounded).map((l) => l.joints[3]);
  if (feet.length < 3) return;
  const hull = convexHull(feet.map((f) => [f[0], f[1]]));
  const pts = hull.map((p) => project([p[0], p[1], 0], view));
  polygon(pts, robot.stable ? COLORS.support : COLORS.supportBad,
          robot.stable ? COLORS.supportEdge : COLORS.supportBadEdge);
}

function drawRobot(view, robot) {
  // Ground shadows first
  for (const leg of robot.legs) {
    const f = leg.joints[3];
    disc(project([f[0], f[1], 0], view), leg.grounded ? 4 : 3, COLORS.shadow);
  }

  // Legs, far ones first so nearer legs overlap them
  const legs = [...robot.legs].sort((a, b) => {
    const da = dot(sub(a.joints[3], view.eye), view.fwd);
    const db = dot(sub(b.joints[3], view.eye), view.fwd);
    return db - da;
  });

  for (const leg of legs) {
    const p = leg.joints.map((j) => project(j, view));
    line(p[0], p[1], COLORS.coxa, 6);
    line(p[1], p[2], COLORS.femur, 5);
    line(p[2], p[3], COLORS.tibia, 4);
    disc(p[1], 3.5, "#0d1117");
    disc(p[1], 2.5, COLORS.coxa);
    disc(p[2], 3.5, "#0d1117");
    disc(p[2], 2.5, COLORS.femur);
    disc(p[3], 4, leg.grounded ? COLORS.footDown : COLORS.footUp);

    if (p[0]) {
      ctx.fillStyle = COLORS.label;
      ctx.font = "10px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("L" + leg.leg, p[0].x, p[0].y - 9);
    }
  }

  // Body plate
  polygon(robot.legs.map((l) => project(l.joints[0], view)),
          COLORS.bodyFill, COLORS.body);

  // Body center and its ground projection — the stability check
  const w = robot.world || { x: 0, y: 0, heading: 0 };
  const pc = project([w.x, w.y, robot.body_z], view);
  const pg = project([w.x, w.y, 0], view);
  if (pc && pg) {
    ctx.strokeStyle = robot.stable ? COLORS.supportEdge : COLORS.supportBadEdge;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(pc.x, pc.y);
    ctx.lineTo(pg.x, pg.y);
    ctx.stroke();
    ctx.setLineDash([]);
    disc(pg, 3, robot.stable ? COLORS.footDown : "#f85149");
  }

  // Heading arrow on the ground
  const h = (w.heading || 0) * Math.PI / 180;
  const tip = project([w.x + 14 * Math.cos(h), w.y + 14 * Math.sin(h), 0], view);
  line(pg, tip, COLORS.heading, 1.5);
  disc(tip, 2.5, COLORS.heading);
}

function render() {
  const rect = canvas.getBoundingClientRect();
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, rect.width, rect.height);

  followRobot();
  const view = makeView(rect.width, rect.height);
  drawGrid(view);
  drawTrail(view);
  if (snap && snap.robot) {
    drawSupport(view, snap.robot);
    drawRobot(view, snap.robot);
  }
  requestAnimationFrame(render);
}

// Counter-clockwise monotone chain, mirroring hexapod/sim/model.py
function convexHull(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cr(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (const p of [...pts].reverse()) {
    while (upper.length >= 2 && cr(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

// ---------------------------------------------------------------------------
// Mouse / touch orbit
// ---------------------------------------------------------------------------

let dragging = null;

canvas.addEventListener("pointerdown", (e) => {
  dragging = { x: e.clientX, y: e.clientY };
  canvas.classList.add("dragging");
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  cam.az -= (e.clientX - dragging.x) * 0.008;
  cam.el = Math.max(-0.2, Math.min(1.45, cam.el + (e.clientY - dragging.y) * 0.006));
  dragging = { x: e.clientX, y: e.clientY };
});
const endDrag = () => { dragging = null; canvas.classList.remove("dragging"); };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  cam.dist = Math.max(35, Math.min(220, cam.dist * (1 + Math.sign(e.deltaY) * 0.1)));
}, { passive: false });
canvas.addEventListener("dblclick", () => Object.assign(cam, HOME));

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------

function badge(el, cls, text) {
  el.className = "badge " + cls;
  el.textContent = text;
}

function updatePanels(s) {
  document.getElementById("device").textContent = s.device;
  badge(document.getElementById("b-client"), s.connected ? "good" : "off",
        s.connected ? "Bus active" : "Bus idle");

  const r = s.robot;
  badge(document.getElementById("b-stable"), r.stable ? "good" : "bad",
        r.stable ? "Support: stable" : "Support: tipping");
  document.getElementById("i-height").textContent = r.body_z.toFixed(1) + " cm";
  document.getElementById("i-tilt").textContent =
    `${r.roll.toFixed(0)}\u00b0 / ${r.pitch.toFixed(0)}\u00b0`;
  document.getElementById("i-margin").textContent = r.margin.toFixed(1) + " cm";
  document.getElementById("i-feet").textContent =
    r.legs.filter((l) => l.grounded).length + " / 6";
  const w = r.world || { x: 0, y: 0, heading: 0 };
  document.getElementById("i-world").textContent =
    `${w.x.toFixed(1)}, ${w.y.toFixed(1)} cm`;
  document.getElementById("i-heading").textContent = w.heading.toFixed(0) + "\u00b0";

  const last = trail[trail.length - 1];
  if (!last || Math.hypot(w.x - last[0], w.y - last[1]) > 0.4) {
    trail.push([w.x, w.y]);
    if (trail.length > 600) trail.shift();
  }

  const now = performance.now();
  if (lastPackets !== null && now - lastRateAt > 500) {
    packetRate = Math.round((s.bus.packets - lastPackets) * 1000 / (now - lastRateAt));
    lastPackets = s.bus.packets;
    lastRateAt = now;
  } else if (lastPackets === null) {
    lastPackets = s.bus.packets;
    lastRateAt = now;
  }
  badge(document.getElementById("b-rate"), packetRate > 0 ? "ok" : "off",
        packetRate + " pkt/s");

  document.getElementById("s-packets").textContent = s.bus.packets;
  document.getElementById("s-responses").textContent = s.bus.responses;
  document.getElementById("s-bad").textContent = s.bus.bad_checksum;
  document.getElementById("s-unknown").textContent = s.bus.unknown_id;
  document.getElementById("s-last").textContent = s.bus.last;

  renderServos(s.servos);
}

function renderServos(servos) {
  const byId = new Map(servos.map((s) => [s.id, s]));
  const body = document.getElementById("servo-body");
  const rows = [];
  for (let leg = 1; leg <= 6; leg++) {
    const cells = [`<td class="leg">${LEG_NAMES[leg]}</td>`];
    let temp = null;
    for (let joint = 1; joint <= 3; joint++) {
      const s = byId.get(leg * 10 + joint);
      if (!s) { cells.push('<td class="limp">—</td>'); continue; }
      temp = Math.max(temp === null ? -Infinity : temp, s.temp);
      const cls = !s.torque ? "limp" : s.moving ? "moving" : "";
      const goal = s.pos !== s.goal ? ` <span class="goal">${s.goal}</span>` : "";
      cells.push(`<td class="${cls}">${s.pos}${goal}</td>`);
    }
    cells.push(`<td class="temp">${temp === null ? "—" : temp.toFixed(0)}</td>`);
    rows.push(`<tr>${cells.join("")}</tr>`);
  }
  body.innerHTML = rows.join("");
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

let socket = null;

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  socket = ws;
  const b = document.getElementById("b-ws");
  ws.onopen = () => badge(b, "good", "WS: connected");
  ws.onclose = () => {
    badge(b, "bad", "WS: reconnecting…");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    snap = JSON.parse(ev.data);
    updatePanels(snap);
  };
}

// Start the view first: a problem wiring up a control must not cost us the
// whole page.
resize();
requestAnimationFrame(render);
connect();

document.getElementById("btn-reset").addEventListener("click", () => {
  trail = [];
  cam.target = [0, 0, 8];
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "reset_world" }));
  }
});
