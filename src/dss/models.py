"""
Typed containers used throughout the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class Patient:
    patient_id: int
    age_group: str  # AD/MA/SE/EL
    gender: str
    race: str
    region: str
    language: str
    visit_freq: str  # high / low
    specialty_request: str
    preference_vector: Dict[str, float]
    cp_group: str
    cp_hours: float
    service_minutes: int


@dataclass
class Doctor:
    doctor_id: int
    specialty: str
    region: str
    language: str
    quality_score: float
    daily_minutes: int
    gender: str
    age: int
    race: str
    service_type: str
    services_count: int
    schedule: Dict[date, int] = field(default_factory=dict)
    hires_at: Optional[date] = None

    def remaining_minutes(self, day: date) -> int:
        return self.daily_minutes - self.schedule.get(day, 0)

    def book(self, day: date, minutes: int) -> None:
        self.schedule[day] = self.schedule.get(day, 0) + minutes


@dataclass
class Appointment:
    patient_id: int
    arrival_id: int
    doctor_id: Optional[int]
    specialty: str
    scheduled_date: Optional[date]
    arrival_date: date
    latest_date: date
    wait_days: Optional[int]
    allocated: bool
    reason: Optional[str] = None
    no_show: Optional[bool] = None


@dataclass
class Arrival:
    arrival_id: int
    patient_id: int
    arrival_date: date
    latest_date: date
    service_minutes: int
    specialty_request: str
    no_show_risk: float


@dataclass
class QuarterState:
    quarter_index: int
    cp_bias: float
    no_show_rate: float
    turnaways: int = 0
    bookings: int = 0

    def register_booking(self, made: bool) -> None:
        self.bookings += 1
        if not made:
            self.turnaways += 1
