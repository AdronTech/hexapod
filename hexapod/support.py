"""
Support-polygon geometry.

A legged robot stays upright while its centre of mass sits inside the convex
hull of the feet that are on the ground.  Both the free gait (deciding whether
a leg can be spared) and the simulator (reporting whether the robot would tip)
need the same answer, so the geometry lives here.

Coordinates are 2D and relative to the point being tested — pass foot positions
already offset by the body centre.
"""

import math

Point2 = tuple[float, float]


def support_margin(feet: list[Point2]) -> float:
    """
    Signed distance from the origin to the support polygon edge.

    Positive means the origin is inside the hull of *feet* — that many cm from
    tipping.  Negative means it is already outside.  Fewer than three feet is
    never supported.
    """
    hull = convex_hull(feet)
    if len(hull) < 3:
        return -1.0

    inside = True
    best = math.inf
    for i, p in enumerate(hull):
        q = hull[(i + 1) % len(hull)]
        ex, ey = q[0] - p[0], q[1] - p[1]
        # Hull is counter-clockwise: an interior point is left of every edge
        if ex * (0.0 - p[1]) - ey * (0.0 - p[0]) < 0:
            inside = False
        best = min(best, _segment_distance(p, q))
    return best if inside else -best


def _segment_distance(p: Point2, q: Point2) -> float:
    """Distance from the origin to the segment p→q."""
    ex, ey = q[0] - p[0], q[1] - p[1]
    length_sq = ex * ex + ey * ey
    if length_sq == 0.0:
        return math.hypot(p[0], p[1])
    t = max(0.0, min(1.0, -(p[0] * ex + p[1] * ey) / length_sq))
    return math.hypot(p[0] + t * ex, p[1] + t * ey)


def convex_hull(points: list[Point2]) -> list[Point2]:
    """Monotone chain hull, counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o: Point2, a: Point2, b: Point2) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point2] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Point2] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]
