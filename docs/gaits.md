# Gaits

Four engines live in [`hexapod/gait.py`](../hexapod/gait.py). All of them share
the same contract: given a world-frame velocity they return a body pose and six
world-frame foot positions, which `body_pose_ik` turns into joint angles.

| Gait | Groups | Duty | Feet down | Character |
|------|--------|------|-----------|-----------|
| Tripod | 2 × 3 | 50% | 3 | fastest |
| Ripple | 3 × 2 | 67% | 4 | medium |
| Wave | 6 × 1 | 83% | 5 | slowest, most stable |
| Free | event-driven | varies | ≥ 3 | steps only when a foot needs it |

Tripod, ripple and wave are time-driven: legs swing on a fixed schedule whether
or not they need to. The free gait is event-driven — a leg steps when its foot
has drifted more than `step_threshold` from where it ought to be.

## Where a swing foot lands

Both target calculations aim at the neutral the leg will have **when it touches
down**, not the one it is leaving. This matters more than it sounds: the swing
takes `step_time`, and the neutral keeps travelling for all of it. Aiming at
the lift-off neutral lands every foot a full swing's travel short.

- **Phased gaits** land half a stance-stride ahead of the touchdown neutral, so
  mid-stance coincides with neutral and the foot's excursion is symmetric about
  it — the largest stability window the stride allows.
- **Free gait** reaches `_LANDING_LEAD × step_threshold` past the touchdown
  neutral, along the direction the foot will drift. The foot then travels back
  through neutral and trips the threshold on the far side, so stance duration
  scales with speed instead of being fixed.

## The shape of the arch

Every gait swings its feet along the same curve: a quintic Bezier from lift-off
to landing, with six control points — `p0, p0, c1, c2, p3, p3`.

The doubled end points are what makes it quiet. A Bezier's velocity at an
endpoint is *degree* x the offset to the next control point, so two coincident
points there means the foot **arrives with zero velocity in every axis** and
decelerates into contact rather than driving into it. A stance foot is fixed in
the world, so zero world velocity at touchdown is also exactly the stance
velocity seen from the body: the arch peels off the flat stance line and
settles back onto it without a corner.

The cubic this replaced put its two control points directly above the
end-points, which is the opposite — vertical speed is *maximal* at both ends.
With a 4 cm step height over a 0.40 s swing the foot hit the ground at 40 cm/s,
which is what the robot sounded like.

`_SWING_CTRL` holds the two lifted points as `(fraction of the chord, multiple
of step_height)`, symmetric about the middle:

```python
_SWING_CTRL = ((0.35, 1.6), (0.65, 1.6))
```

With both at the same height the arch is symmetric, `z(t) = 10*H*t^2*(1-t)^2`,
peaking at `0.625 * H` — hence 1.6, which puts the apex on exactly
`step_height`. Moving the pair apart flattens the top and steepens the flanks;
moving them together gives a rounder, pointier arch. Peak foot speed rises to
about 1.9 x the chord per `step_time` (the cubic's was 1.5 x), since the same
distance is now covered with both ends standing still.

## The free gait's speed budget

A planted foot drifts at its neutral's speed and must be lifted before that
drift passes `step_threshold`; a leg cannot swing more often than once per
stance. Together those cap the neutral speed the gait can service:

```
budget = (1 + _LANDING_LEAD) × step_threshold / step_time
```

With the defaults (threshold 3 cm, step time 0.40 s) that is **14.25 cm/s**.

`FreeGait.step` scales `vx`, `vy` and `omega` together to stay inside it, so the
robot still travels in the commanded direction — just as fast as its legs can
carry it. The controller shows *"Free — speed limited to N%"* when this bites,
and recordings carry the factor as `scale`.

Commanding more than the budget does not make the robot faster. The feet simply
fall behind, every lift becomes an emergency, the stability guard stops applying
and the legs get dragged until the IK gives out.

**Rotation hits the ceiling first.** An outer leg's neutral sits about 26 cm
from the body centre, so 60 °/s moves it at 27 cm/s — nearly double the budget.
Turning in free mode is limited to roughly 31 °/s at default settings.

To go faster, raise `step_threshold` (longer strides) or lower `step_time`
(quicker swings). Both are sliders in the web UI, and both trade smoothness for
pace.

## Staying upright

The free gait may have at most three legs in the air, and adjacent legs never
swing together unless a foot is past `FREE_STEP_EMERGENCY` (6 cm).

Three feet down is not sufficient on its own: the three legs of one side are
nearly collinear, so the support polygon is a sliver the body centre falls
outside of — which is exactly what a robot tipping over looks like. Every lift
is therefore also checked against the support polygon itself, projected to the
end of the swing, and refused if the body centre would come within
`_SUPPORT_MARGIN_MIN` of its edge. The emergency override may skip the
adjacency rule but never this one: a late leg is recoverable, a robot on its
side is not.

## Tuning

| Knob | Effect |
|------|--------|
| `step_height` | how high the swing arch peaks |
| `step_time` | swing duration; also sets the free gait's step rate ceiling |
| `step_threshold` | free gait only — drift a foot tolerates before stepping |
| `neutral_reach` | how far out the feet stand from the body |

`docs/recording.md` covers how to record a session and read the step statistics
back out; it is the fastest way to tell whether a tuning change helped.
