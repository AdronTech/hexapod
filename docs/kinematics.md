# Hexapod Kinematics Reference

## Body Geometry

The body is a regular hexagon. The coxa motors are mounted **at the six corners**.

- L1 = FRONT_RIGHT, L2 = MID_RIGHT, L3 = REAR_RIGHT
- L4 = REAR_LEFT,   L5 = MID_LEFT,  L6 = FRONT_LEFT

**Body diagonal (corner to corner):** 17.7 cm  
**Body radius (center to corner):** ≈ 8.85 cm

The hexagon is flat-sided front and back (not vertex-forward), so the corner angles
measured from the forward (X) axis are:

| Leg | Corner angle from forward |
|-----|--------------------------|
| L1 FRONT_RIGHT | −30° |
| L2 MID_RIGHT   | −90° |
| L3 REAR_RIGHT  | −150° |
| L4 REAR_LEFT   | +150° |
| L5 MID_LEFT    | +90° |
| L6 FRONT_LEFT  | +30° |

Servo ID = `leg × 10 + joint` — e.g. leg 2 femur = ID 22

---

## Link Lengths

| Segment | From → To | Length |
|---------|-----------|--------|
| Coxa | Body attachment → Femur pivot | 6.4 cm |
| Femur | Femur pivot → Tibia pivot | 11.0 cm |
| Tibia | Tibia pivot → Foot tip | 15.0 cm |

---

## Coordinate Frames

### Body frame

Origin at body center, fixed to the chassis. X = forward, Y = left, Z = up.

### Leg frame

Origin at the coxa pivot (the corner). X_leg points **radially outward from the body center through the corner** — this is the neutral coxa direction. Z_leg = up. Y_leg = Z_leg × X_leg.

The X_leg directions in the body frame (X=forward, Y=left):

| Leg | X_leg |
|-----|-------|
| L1 | (cos −30°, sin −30°) = (+0.866, −0.500) |
| L2 | (cos −90°, sin −90°) = (0, −1) |
| L3 | (cos −150°, sin −150°) = (−0.866, −0.500) |
| L4 | (cos 150°, sin 150°) = (−0.866, +0.500) |
| L5 | (cos 90°, sin 90°) = (0, +1) |
| L6 | (cos 30°, sin 30°) = (+0.866, +0.500) |

---

## Joint Definitions and Neutral (tick 2048)

### Coxa — horizontal rotation (yaw)

- **2048** = coxa pointing radially outward from the body center through its corner.
- Tick **increases clockwise** when viewed from above.
- Hardware range: ≈ ±90°

### Femur — vertical rotation (pitch, lift)

- **2048** = femur horizontal, parallel to the ground.
- Tick **increases as the femur tip moves down** (clockwise when viewing the leg with the body attachment on the left).
- Note: the joint sits close to one hardware stop at neutral (see limits below).

### Tibia — vertical rotation (pitch, knee)

- **2048** = tibia vertical, pointing straight down.
- Tick **decreases as the tibia moves inward** toward the body (clockwise when viewing the leg with the body attachment on the left); equivalently, tick increases as the tibia swings outward.
- 2048 is near the center of the physical range.

### Tick-to-angle sign summary

Defining angles as positive in the intuitive/IK-friendly direction (coxa forward,
femur up, tibia outward):

| Joint | Positive angle means | Conversion |
|-------|---------------------|------------|
| Coxa  | CCW from above (toward front for right-side legs) | `θ = −(tick − 2048) × 0.08789°` |
| Femur | tip rises above horizontal | `θ = −(tick − 2048) × 0.08789°` |
| Tibia | tip swings outward | `θ = +(tick − 2048) × 0.08789°` |

### Neutral foot position

At tick 2048 on all joints, the foot is located in the leg frame at:

- x = 6.4 + 11.0 = **17.4 cm** (straight out from body)
- z = **−15.0 cm** (straight down)
- y = 0

---

## Tick ↔ Angle Conversion

The encoder has 4096 ticks per full revolution:

    degrees = (tick − 2048) × (360 / 4096) = (tick − 2048) × 0.08789°
    tick    = 2048 + round(degrees / 0.08789)

See the sign summary table in the joint definitions above for the per-joint sign.

---

## Hardware Limits

Measured at the physical stops in raw ticks **before OFS calibration**.
After calibration the tick values shift; the angular range stays the same.

| Joint | Stop A (tick) | Stop B (tick) | Wraps 0/4096? | Range (ticks) | Range (°) |
|-------|--------------|--------------|---------------|---------------|-----------|
| Coxa  | 3080 | 1062 | yes | ≈ 2078 | ≈ 183° |
| Femur | 1934 | 556  | yes | ≈ 2718 | ≈ 238° |
| Tibia | 268  | 1997 | no  | ≈ 1729 | ≈ 152° |

**Wraps** means the servo passes through the 0/4096 tick boundary within its physical
range. The angle limit registers (0x09/0x0B) must both be set to 0 to allow this —
otherwise the firmware will block movement across the boundary.

Post-calibration symmetric ranges around 2048 are tbd — re-measure the stops after
calibration with the monitor script.

---

## Foot Reachability (in the neutral leg plane, y = 0)

Neutral foot position: (17.4 cm, 0, −15 cm) in the leg frame.

Theoretical reach limits (ignoring joint angle constraints):

- r_max = 6.4 + 11.0 + 15.0 = **32.4 cm**
- r_min ≈ 6.4 + |11.0 − 15.0| = **10.4 cm**

Actual workspace is further constrained by joint angle limits and body/ground collisions.
