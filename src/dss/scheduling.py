"""
Scheduling algorithm: locate earliest feasible slot and account for no-show driven overbooking.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Sequence

from .config import SimulationConfig
from .models import Appointment, Arrival, Doctor, QuarterState


class Scheduler:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    def _overbook_factor(self, state: QuarterState) -> float:
        # Allow modest overbooking proportional to observed no-show rate.
        return min(
            self.cfg.overbook_ceiling,
            max(self.cfg.overbook_floor, state.no_show_rate * 1.2),
        )

    def _first_available_day(
        self,
        doctor: Doctor,
        start: date,
        latest: date,
        need_minutes: int,
        state: QuarterState,
    ) -> Optional[date]:
        """Find the first available weekday (Mon-Fri) with enough capacity.

        Requirements:
        - Doctor works only 5 days per week (weekdays only, Monday-Friday)
        - Doctor works at most 6 hours (360 minutes) per day
        """
        overbook = self._overbook_factor(state)
        day = start
        while day <= latest:
            # Check if it's a weekday (Monday=0 to Friday=4)
            # Doctors only work Monday-Friday (5 days per week)
            if day.weekday() < 5:  # 0-4 are Monday-Friday
                # Calculate daily capacity (6 hours = 360 minutes)
                daily_capacity = doctor.daily_minutes * (1 + overbook)
                current_minutes = doctor.schedule.get(day, 0)

                # Check if there's enough capacity for this appointment
                if current_minutes + need_minutes <= daily_capacity:
                    return day
            day += timedelta(days=1)
        return None

    def schedule(
        self, arrival: Arrival, doctor_choices: Sequence[Doctor], state: QuarterState
    ) -> Appointment:
        # 首先检查primarydoctor是否可行，可行则直接安排，
            doctor = doctor_choices[0] if doctor_choices else None
            slot = self._first_available_day(
                doctor,
                arrival.arrival_date,
                arrival.latest_date,
                arrival.service_minutes,
                state,
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
            else:
                # find the earliest available slot among all candidates (including primary if exists)
                earliest_slot = None
                for doctor in doctor_choices:
                    slot = self._first_available_day(
                        doctor,
                        arrival.arrival_date,
                        arrival.latest_date,
                        arrival.service_minutes,
                        state,
                    )
                    if slot and (not earliest_slot or slot < earliest_slot[1]):
                        earliest_slot = (doctor, slot)
                if earliest_slot:
                    # print("find earliest")
                    doctor, slot = earliest_slot
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
                else:
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
                 
                 
        # for doctor in doctor_choices:
        #     slot = self._first_available_day(
        #         doctor,
        #         arrival.arrival_date,
        #         arrival.latest_date,
        #         arrival.service_minutes,
        #         state,
        #     )
        #     if slot:
        #         doctor.book(slot, arrival.service_minutes)
        #         wait = (slot - arrival.arrival_date).days
        #         return Appointment(
        #             patient_id=arrival.patient_id,
        #             arrival_id=arrival.arrival_id,
        #             doctor_id=doctor.doctor_id,
        #             specialty=doctor.specialty,
        #             scheduled_date=slot,
        #             arrival_date=arrival.arrival_date,
        #             latest_date=arrival.latest_date,
        #             wait_days=wait,
        #             allocated=True,
        #         )
        # return Appointment(
        #     patient_id=arrival.patient_id,
        #     arrival_id=arrival.arrival_id,
        #     doctor_id=None,
        #     specialty=doctor_choices[0].specialty if doctor_choices else "unknown",
        #     scheduled_date=None,
        #     arrival_date=arrival.arrival_date,
        #     latest_date=arrival.latest_date,
        #     wait_days=None,
        #     allocated=False,
        #     reason="No capacity before latest acceptable date",
        # )
