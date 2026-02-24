"""
Allocation algorithm: rank physicians for an arrival using patient preferences.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Sequence

from .config import SimulationConfig
from .models import Doctor, Patient


@dataclass
class AllocationEngine:
    cfg: SimulationConfig

    def pick_specialty(self, patient: Patient) -> str:
        return patient.specialty_request

    def rank_doctors(self, patient: Patient, doctors: Sequence[Doctor], day: date) -> List[Doctor]:
        ranked = []
        for doc in doctors:
            annual_remaining = doc.annual_remaining_minutes(day.year)
            if annual_remaining <= 0:
                continue

            pref = 0.0
            pref += patient.preference_vector["region_bias"] * (doc.region == patient.region)
            pref += patient.preference_vector["language_bias"] * (doc.language == patient.language)
            pref += patient.preference_vector["quality_bias"] * doc.quality_score
            pref += patient.preference_vector["gender_bias"] * (doc.gender == patient.gender)
            pref += patient.preference_vector["race_bias"] * (doc.race == patient.race)
            pref += patient.preference_vector["service_type_bias"] * (doc.service_type == "group")
            pref += patient.preference_vector["service_count_bias"] * (doc.services_count / 5)
            pref += patient.preference_vector['experience_bias'] * (round(doc.experience_years/10)*0.25)
            pref += patient.preference_vector['board_certification_bias']* doc.board_certified
            annual_capacity = max(1.0, float(doc.annual_capacity_minutes(day.year)))
            remaining_ratio = annual_remaining / annual_capacity
            capacity_term = 0.35 * remaining_ratio
            ranked.append((pref + capacity_term, doc))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked]
