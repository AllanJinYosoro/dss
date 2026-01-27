"""
Doctor supplementation logic triggered when overload persists.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List

from random import Random

from .config import REGIONS, LANGUAGES, SimulationConfig
from .models import Doctor
from .data_generation import _new_doctor_from_rng


class StaffingManager:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self._next_id = 10_000  # distinct id range for hires

    def maybe_hire(
        self,
        quarter_turnaways: Dict[str, int],
        quarter_bookings: Dict[str, int],
        roster: List[Doctor],
        as_of: date,
    ) -> List[Doctor]:
        additions: List[Doctor] = []
        for specialty, turns in quarter_turnaways.items():
            bookings = max(1, quarter_bookings.get(specialty, 1))
            overload = turns / bookings
            if overload > 0.08 or bookings > 800:  # persistent saturation
                additions.append(self._new_doctor(specialty, as_of))
        roster.extend(additions)
        return additions

    def _new_doctor(self, specialty: str, hire_date: date) -> Doctor:
        rng = Random(self._next_id * 13 + int(hire_date.strftime("%j")))
        doc = _new_doctor_from_rng(self._next_id, specialty, rng, self.cfg)
        doc.hires_at = hire_date
        self._next_id += 1
        return doc
