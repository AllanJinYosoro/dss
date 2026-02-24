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
    age: int
    age_group: str  # AD/MA/SE/EL
    gender: str
    race: str
    region: str
    language: str
    historical_visits: float
    cp: float
    cp_group: str
    specialty_request: str
    allocated_doctor_id: Optional[int]
    preference_vector: Dict[str, float]


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
    experience_years: int
    board_certified: bool
    current_panel_size: int
    expected_workload: float
    max_workload: float
    # year -> list of ISO week numbers (1..52) when this doctor works
    working_weeks_by_year: Dict[int, List[int]]
    hires_at: Optional[date] = None
    schedule: Dict[date, int] = field(default_factory=dict)

    def remaining_minutes(self, day: date) -> int:
        return self.daily_minutes - self.schedule.get(day, 0)

    def book(self, day: date, minutes: int) -> None:
        self.schedule[day] = self.schedule.get(day, 0) + minutes

    def is_working_day(self, day: date) -> bool:
        if day.weekday() >= 5:
            return False
        return day.isocalendar().week in set(self.working_weeks_by_year.get(day.year, []))

    def annual_capacity_minutes(self, year: int) -> int:
        return len(self.working_weeks_by_year.get(year, [])) * 5 * self.daily_minutes

    def annual_booked_minutes(self, year: int) -> int:
        return sum(v for d, v in self.schedule.items() if d.year == year)

    def annual_remaining_minutes(self, year: int) -> int:
        return self.annual_capacity_minutes(year) - self.annual_booked_minutes(year)


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
    patient_class: str
    expected_visits: float


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

