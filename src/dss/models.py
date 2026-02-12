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
    age:int #新增
    age_group: str  # AD/MA/SE/EL
    gender: str
    race: str
    region: str
    language: str
    historical_visits: float #原visit_freq: str  # high / low
    cp_hours: float 
    cp_group: str #class_code 
    specialty_request: str
    service_minutes:int
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
    board_certified: bool #新增
    current_panel_size: int
    expected_workload: int
    hires_at: Optional[date] = None
    
    schedule: Dict[date, int] = field(default_factory=dict) 
    

    def remaining_minutes(self, day: date) -> int:
        return self.daily_minutes - self.schedule.get(day, 0)

    def book(self, day: date, minutes: int) -> None:
        self.schedule[day] = self.schedule.get(day, 0) + minutes


@dataclass
class Appointment: #先放着，原本的没有
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
    patient_class: str #新加的
    expected_visits: float #新加的


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
