"""
Scheduling algorithm with no-show aware overbooking and doctor work-calendar constraints.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Sequence, Tuple

from .config import SimulationConfig
from .models import Appointment, Arrival, Doctor, QuarterState


class Scheduler:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    def _overbook_factor(self, state: QuarterState) -> float:
        return min(self.cfg.overbook_ceiling, max(self.cfg.overbook_floor, state.no_show_rate * 1.2))

    def _first_available_day(
        self,
        doctor: Doctor,
        start: date,
        latest: date,
        need_minutes: int,
        state: QuarterState,
    ) -> Optional[date]:
        if doctor.annual_remaining_minutes(start.year) < need_minutes:
            return None

        overbook = self._overbook_factor(state)
        day = start
        while day <= latest:
            if not doctor.is_working_day(day):
                day += timedelta(days=1)
                continue

            daily_capacity = doctor.daily_minutes * (1 + overbook)
            current_minutes = doctor.schedule.get(day, 0)
            if current_minutes + need_minutes <= daily_capacity:
                return day
            day += timedelta(days=1)
        return None

    def _book(self, doctor: Doctor, arrival: Arrival, slot: date) -> Appointment:
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

    def schedule(
        self, arrival: Arrival, doctor_choices: Sequence[Doctor], state: QuarterState
    ) -> Appointment:
        if not doctor_choices:
            return Appointment(
                patient_id=arrival.patient_id,
                arrival_id=arrival.arrival_id,
                doctor_id=None,
                specialty="unknown",
                scheduled_date=None,
                arrival_date=arrival.arrival_date,
                latest_date=arrival.latest_date,
                wait_days=None,
                allocated=False,
                reason="No candidate doctor",
            )

        # Primary doctor first.
        primary = doctor_choices[0]
        slot = self._first_available_day(
            primary, arrival.arrival_date, arrival.latest_date, arrival.service_minutes, state
        )
        if slot:
            return self._book(primary, arrival, slot)

        # Fallback: earliest feasible slot among all candidates.
        best: Optional[Tuple[Doctor, date]] = None
        for doctor in doctor_choices:
            slot = self._first_available_day(
                doctor, arrival.arrival_date, arrival.latest_date, arrival.service_minutes, state
            )
            if slot and (best is None or slot < best[1]):
                best = (doctor, slot)

        if best:
            return self._book(best[0], arrival, best[1])

        return Appointment(
            patient_id=arrival.patient_id,
            arrival_id=arrival.arrival_id,
            doctor_id=None,
            specialty=doctor_choices[0].specialty,
            scheduled_date=None,
            arrival_date=arrival.arrival_date,
            latest_date=arrival.latest_date,
            wait_days=None,
            allocated=False,
            reason="No capacity before latest acceptable date",
        )

