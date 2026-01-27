"""
End-to-end simulation wiring: generation -> allocation -> scheduling -> staffing.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .allocation import AllocationEngine
from .config import SimulationConfig
from .data_generation import (
    DEFAULT_DATA_DIR,
    generate_doctors,
    generate_patients,
    generate_arrivals,
    load_data,
    save_data,
)
from .models import Appointment, Doctor, Patient, QuarterState, Arrival
from .scheduling import Scheduler
from .staffing import StaffingManager


class Simulation:
    def __init__(self, cfg: SimulationConfig, data_dir: Optional[Path] = None, regenerate: bool = False):
        self.cfg = cfg
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.regenerate = regenerate
        self.allocator = AllocationEngine(cfg)
        self.scheduler = Scheduler(cfg)
        self.staffing = StaffingManager(cfg)

    def run(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        data_exists = all(
            (self.data_dir / f).exists() for f in ["patients.csv", "doctors.csv", "arrivals.csv"]
        )
        if data_exists and not self.regenerate:
            patients, doctors, arrivals = load_data(self.data_dir)
        else:
            patients = generate_patients(self.cfg)
            doctors = generate_doctors(self.cfg)
            arrivals, _cal = generate_arrivals(self.cfg, patients)
            save_data(patients, doctors, arrivals, self.data_dir)

        appointments: List[Appointment] = []
        quarter_state = QuarterState(quarter_index=0, cp_bias=0.0, no_show_rate=self.cfg.baseline_no_show)
        quarter_no_show = 0
        quarter_seen = 0

        quarter_turnaways: Dict[str, int] = defaultdict(int)
        quarter_bookings: Dict[str, int] = defaultdict(int)

        calendar_start = arrivals[0].arrival_date if arrivals else self.cfg.start_date

        patient_lookup = {p.patient_id: p for p in patients}
        primary_doctor: Dict[int, int] = {}

        for arrival in arrivals:
            patient = patient_lookup[arrival.patient_id]
            q_idx = self._quarter_index(calendar_start, arrival.arrival_date)
            if q_idx != quarter_state.quarter_index:
                # finalize prior quarter no-show estimate
                if quarter_seen > 0:
                    quarter_state.no_show_rate = quarter_no_show / quarter_seen
                quarter_state.cp_bias = 0.0
                self.staffing.maybe_hire(
                    quarter_turnaways, quarter_bookings, doctors, as_of=arrival.arrival_date
                )
                quarter_turnaways = defaultdict(int)
                quarter_bookings = defaultdict(int)
                quarter_no_show = 0
                quarter_seen = 0
                quarter_state.quarter_index = q_idx

            specialty = self.allocator.pick_specialty(patient)

            doctor_candidates = [
                d
                for d in doctors
                if d.specialty == specialty and (d.hires_at is None or arrival.arrival_date >= d.hires_at)
            ]

            if arrival.patient_id not in primary_doctor:
                ranked = self.allocator.rank_doctors(patient, doctor_candidates, arrival.arrival_date)
                if ranked:
                    primary_doctor[arrival.patient_id] = ranked[0].doctor_id
                ordered = ranked
            else:
                prim_id = primary_doctor[arrival.patient_id]
                primary_doc = next((d for d in doctor_candidates if d.doctor_id == prim_id), None)
                others = [d for d in doctor_candidates if d.doctor_id != prim_id]
                ranked_others = self.allocator.rank_doctors(patient, others, arrival.arrival_date)
                ordered = ([primary_doc] if primary_doc else []) + ranked_others

            appt = self.scheduler.schedule(arrival, ordered, quarter_state)

            # simulate no-show outcome if scheduled
            if appt.allocated and appt.scheduled_date:
                pr = min(0.8, arrival.no_show_risk + 0.5 * quarter_state.no_show_rate)
                appt.no_show = bool(np.random.random() < pr)
                quarter_seen += 1
                if appt.no_show:
                    quarter_no_show += 1
            else:
                appt.no_show = None

            appointments.append(appt)
            quarter_bookings[specialty] += 1
            if not appt.allocated:
                quarter_turnaways[specialty] += 1

        df = self._to_dataframe(appointments, patients, arrivals)
        metrics = self._compute_metrics(df)
        return df, metrics

    def _quarter_index(self, start: date, current: date) -> int:
        months = (current.year - start.year) * 12 + (current.month - start.month)
        return months // 3

    def _to_dataframe(
        self, appointments: List[Appointment], patients: List[Patient], arrivals: List[Arrival]
    ) -> pd.DataFrame:
        patient_lookup = {p.patient_id: p for p in patients}
        arrival_lookup = {a.arrival_id: a for a in arrivals}
        records = []
        for appt in appointments:
            p = patient_lookup[appt.patient_id]
            a = arrival_lookup[appt.arrival_id]
            records.append(
                {
                    "patient_id": appt.patient_id,
                    "arrival_id": appt.arrival_id,
                    "doctor_id": appt.doctor_id,
                    "specialty": appt.specialty,
                    "arrival_date": appt.arrival_date,
                    "latest_date": appt.latest_date,
                    "scheduled_date": appt.scheduled_date,
                    "wait_days": appt.wait_days,
                    "allocated": appt.allocated,
                    "no_show": appt.no_show,
                    "age_group": p.age_group,
                    "gender": p.gender,
                    "race": p.race,
                    "region": p.region,
                    "language": p.language,
                    "visit_freq": p.visit_freq,
                    "specialty_request": p.specialty_request,
                    "service_minutes": a.service_minutes,
                }
            )
        return pd.DataFrame.from_records(records)

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["fill_rate"] = df["allocated"].mean()
        metrics["avg_wait_if_scheduled"] = df[df["allocated"]]["wait_days"].mean()
        metrics["no_show_rate"] = df["no_show"].dropna().mean()
        metrics["general_match_rate"] = (df["specialty"] == "general").mean()
        return metrics
