"""
Centralized simulation defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List


GENDERS = ["M", "F"]
LANGUAGES = ["en", "es", "zh"]
RACES = ["White", "Black", "Asian", "Hispanic", "Other"]
REGIONS = ["North", "South", "East", "West", "Central"]
SERVICE_TYPES = ["solo", "group"]
AGE_GROUPS = ["AD", "MA", "SE", "EL"]
NO_SHOW_BASE_RATE = 0.08
PCP_SPECIALTIES = ["family_practice", "internal_medicine", "pediatrics"]
SPECIALTY_ABBREVIATIONS = {
    "family_practice": "FP",
    "internal_medicine": "IM",
    "pediatrics": "PD"
}

@dataclass
class SimulationConfig:
    start_date: date = date(2023, 1, 1)
    years: int = 2
    patients_per_year: int = 30000
    seed: int = 42
    """ 暂时没找到用处
    service_minutes: Dict[str, int] = field(
        default_factory=lambda: {"general": 20, "internal": 25, "pediatrics": 20}
    ) """
    doctor_daily_minutes: int = 6 * 60  # usable minutes per physician per workday
    base_doctor_counts: Dict[str, int] = field(
        default_factory=lambda: {"family_practice": 200, "internal_medicine": 100, "pediatrics": 80}
    )
    base_doctor_counts_all: int = 380
    avg_patients_per_doctor :int = 1500 #新增
    appointment_duration : int = 20 #新增
    pcp_surge_multiplier: float = 1.35  # Q1-Q2 of year 2
    universal_surge_multiplier: float = 1.20  # Q3-Q4 of year 2
    overbook_floor: float = 0.03  # minimal overbook rate
    overbook_ceiling: float = 0.18  # max overbook rate driven by pr
    max_wait_days: int = 35
    preference_noise: float = 0.15
    baseline_no_show: float = 0.08
