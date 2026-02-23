"""
Doctor supplementation logic triggered when overload persists.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List
import numpy as np 
from random import Random

from .config import REGIONS, LANGUAGES, SimulationConfig
from .models import Doctor
from .data_generation import _create_doctor


class StaffingManager:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self._next_id = cfg.base_doctor_counts_all+1  # distinct id range for hires

    def maybe_hire(self,doctors,specialty) -> bool:
        specialist_doctors = [doc for doc in doctors if doc.specialty == specialty]
        if not specialist_doctors:
            return False
        sum1 = np.sum([doc.expected_workload for doc in specialist_doctors])
        sum2 = np.sum([doc.max_workload for doc in specialist_doctors])
        result = sum1 > 0.995*sum2
        return result

    def _new_doctor(self, specialty: str, hire_date: date) -> Doctor:
        rng = Random(self._next_id * 13 + int(hire_date.strftime("%j")))
        doc = _create_doctor(self._next_id, specialty, rng, self.cfg)
        doc.hires_at = hire_date
        self._next_id += 1
        return doc
