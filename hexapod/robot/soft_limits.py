"""
Soft joint angle limits — loaded from soft_limits.json at the repo root.

Angles are stored in the IK-friendly sign convention (docs/kinematics.md):
  coxa  positive = CCW from above
  femur positive = tip rises above horizontal
  tibia positive = tip swings outward
"""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent.parent.parent / "soft_limits.json"


class SoftLimitError(ValueError):
    pass


@dataclass
class JointLimits:
    min_deg: float
    max_deg: float

    def contains(self, angle: float) -> bool:
        return self.min_deg <= angle <= self.max_deg


@dataclass
class SoftLimits:
    coxa: JointLimits
    femur: JointLimits
    tibia: JointLimits

    def check(self, coxa_deg: float, femur_deg: float, tibia_deg: float) -> None:
        """Raise SoftLimitError if any angle violates its limit."""
        violations = []
        for name, lim, val in (
            ("coxa",  self.coxa,  coxa_deg),
            ("femur", self.femur, femur_deg),
            ("tibia", self.tibia, tibia_deg),
        ):
            if not lim.contains(val):
                violations.append(
                    f"{name} {val:+.1f}° outside [{lim.min_deg:+.1f}°, {lim.max_deg:+.1f}°]"
                )
        if violations:
            raise SoftLimitError("; ".join(violations))

    def save(self, path: Path = DEFAULT_PATH) -> None:
        data = {
            "coxa":  {"min_deg": self.coxa.min_deg,  "max_deg": self.coxa.max_deg},
            "femur": {"min_deg": self.femur.min_deg, "max_deg": self.femur.max_deg},
            "tibia": {"min_deg": self.tibia.min_deg, "max_deg": self.tibia.max_deg},
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "SoftLimits | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(
            coxa=JointLimits(**data["coxa"]),
            femur=JointLimits(**data["femur"]),
            tibia=JointLimits(**data["tibia"]),
        )
