"""
Scheduling algorithm: locate earliest feasible slot and account for no-show driven overbooking.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Sequence

from .config import SimulationConfig
from .models import Appointment, Doctor, Arrival, QuarterState


class Scheduler:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    def _overbook_factor(self, state: QuarterState) -> float:
        # Allow modest overbooking proportional to observed no-show rate.
        return min(self.cfg.overbook_ceiling, max(self.cfg.overbook_floor, state.no_show_rate * 1.2))

    def _first_available_day(
        self, doctor: Doctor, start: date, latest: date, need_minutes: int, state: QuarterState
    ) -> Optional[date]:
        overbook = self._overbook_factor(state)
        day = start
        while day <= latest:
            capacity = doctor.daily_minutes * (1 + overbook)
            if doctor.schedule.get(day, 0) + need_minutes <= capacity:
                return day
            day += timedelta(days=1)
        return None

    def schedule(
        self, arrival: Arrival, doctor_choices: Sequence[Doctor], state: QuarterState
    ) -> Appointment:
        for doctor in doctor_choices:
            slot = self._first_available_day(
                doctor, arrival.arrival_date, arrival.latest_date, arrival.service_minutes, state
            )
            if slot:
                doctor.book(slot, arrival.service_minutes)
                wait = (slot - arrival.arrival_date).days
                return Appointment(
                    patient_id=arrival.patient_id,
                    arrival_id=arrival.arrival_id,
                    doctor_id=doctor.doctor_id,
                    specialty=doctor.specialty,
                    scheduled_date=slot,
                    arrival_date=arrival.arrival_date,
                    latest_date=arrival.latest_date,
                    wait_days=wait,
                    allocated=True,
                )
        return Appointment(
            patient_id=arrival.patient_id,
            arrival_id=arrival.arrival_id,
            doctor_id=None,
            specialty=doctor_choices[0].specialty if doctor_choices else "unknown",
            scheduled_date=None,
            arrival_date=arrival.arrival_date,
            latest_date=arrival.latest_date,
            wait_days=None,
            allocated=False,
            reason="No capacity before latest acceptable date",
        )
