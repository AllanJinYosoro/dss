"""
Centralized simulation defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List


SPECIALTIES: List[str] = ["general", "internal", "pediatrics"]
REGIONS: List[str] = ["north", "central", "south"]
LANGUAGES: List[str] = ["en", "es", "zh"]
RACES: List[str] = ["white", "black", "asian", "hispanic", "other"]
GENDERS: List[str] = ["F", "M"]
SERVICE_TYPES: List[str] = ["group", "solo"]


@dataclass
class SimulationConfig:
    start_date: date = date(2026, 1, 1)
    years: int = 2
    patients_per_year: int = 30_000
    seed: int = 42
    service_minutes: Dict[str, int] = field(
        default_factory=lambda: {"general": 20, "internal": 25, "pediatrics": 20}
    )
    doctor_daily_minutes: int = 7 * 60  # usable minutes per physician per workday
    base_doctor_counts: Dict[str, int] = field(
        default_factory=lambda: {"general": 20, "internal": 12, "pediatrics": 10}
    )
    pcp_surge_multiplier: float = 1.35  # Q1-Q2 of year 2
    universal_surge_multiplier: float = 1.20  # Q3-Q4 of year 2
    overbook_floor: float = 0.03  # minimal overbook rate
    overbook_ceiling: float = 0.18  # max overbook rate driven by pr
    max_wait_days: int = 35
    preference_noise: float = 0.15
    baseline_no_show: float = 0.10
