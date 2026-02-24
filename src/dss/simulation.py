"""
End-to-end simulation wiring: generation -> allocation -> scheduling -> staffing.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .allocation import AllocationEngine
from .config import SimulationConfig
from .data_generation import generate_arrivals, generate_doctors, generate_patients, load_data, save_data
from .models import Appointment, Arrival, Doctor, Patient, QuarterState
from .scheduling import Scheduler
from .staffing import StaffingManager

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class Simulation:
    def __init__(self, cfg: SimulationConfig, data_dir: Optional[Path] = None, regenerate: bool = False):
        self.cfg = cfg
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.regenerate = regenerate
        self.allocator = AllocationEngine(cfg)
        self.scheduler = Scheduler(cfg)
        self.staffing = StaffingManager(cfg)

    def run(self) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame]:
        data_exists = all((self.data_dir / f).exists() for f in ["patients.csv", "doctors.csv", "arrivals.csv"])
        if data_exists and not self.regenerate:
            patients, doctors, arrivals = load_data(self.data_dir)
        else:
            patients = generate_patients(self.cfg)
            doctors = generate_doctors(self.cfg)
            arrivals, _calendar = generate_arrivals(self.cfg, patients)
            save_data(patients, doctors, arrivals, self.data_dir)

        appointments: List[Appointment] = []
        schedule_log: List[Dict[str, object]] = []
        quarter_state = QuarterState(quarter_index=0, cp_bias=0.0, no_show_rate=self.cfg.baseline_no_show)
        quarter_no_show = 0
        quarter_seen = 0
        quarter_turnaways: Dict[str, int] = defaultdict(int)
        quarter_bookings: Dict[str, int] = defaultdict(int)

        calendar_start = arrivals[0].arrival_date if arrivals else self.cfg.start_date
        patient_lookup = {p.patient_id: p for p in patients}
        doctor_lookup = {d.doctor_id: d for d in doctors}

        for arrival in arrivals:
            # Keep this model assumption unchanged.
            arrival.service_minutes = 30

            patient = patient_lookup[arrival.patient_id]
            specialty = self.allocator.pick_specialty(patient)

            if pd.isna(patient.allocated_doctor_id):
                first_candidates = [
                    d
                    for d in doctors
                    if d.specialty == specialty
                    and d.expected_workload < d.max_workload
                    and d.annual_remaining_minutes(arrival.arrival_date.year) >= arrival.service_minutes
                ]
                if not first_candidates:
                    continue
                ranked = self.allocator.rank_doctors(patient, first_candidates, arrival.arrival_date)
                if not ranked:
                    continue
                selected_doc = ranked[0]
                patient.allocated_doctor_id = selected_doc.doctor_id
                selected_doc.current_panel_size += 1
                selected_doc.expected_workload += patient.cp

            primary_id = patient.allocated_doctor_id
            specialty_candidates = [
                d
                for d in doctors
                if d.specialty == specialty
                and d.annual_remaining_minutes(arrival.arrival_date.year) >= arrival.service_minutes
            ]
            primary_doc = next((d for d in specialty_candidates if d.doctor_id == primary_id), None)

            ordered_raw = ([primary_doc] if primary_doc else []) + specialty_candidates
            seen = set()
            ordered: List[Doctor] = []
            for d in ordered_raw:
                if d.doctor_id not in seen:
                    seen.add(d.doctor_id)
                    ordered.append(d)

            q_idx = self._quarter_index(calendar_start, arrival.arrival_date)
            if q_idx != quarter_state.quarter_index:
                if quarter_seen > 0:
                    quarter_state.no_show_rate = quarter_no_show / quarter_seen
                quarter_state.cp_bias = 0.0
                quarter_turnaways = defaultdict(int)
                quarter_bookings = defaultdict(int)
                quarter_no_show = 0
                quarter_seen = 0
                quarter_state.quarter_index = q_idx

            appt = self.scheduler.schedule(arrival, ordered, quarter_state)

            if appt.allocated and appt.scheduled_date and appt.doctor_id is not None:
                doc = doctor_lookup.get(appt.doctor_id)
                if doc:
                    schedule_log.append(
                        {
                            "arrival_id": arrival.arrival_id,
                            "patient_id": arrival.patient_id,
                            "doctor_id": appt.doctor_id,
                            "arrival_date": arrival.arrival_date,
                            "scheduled_date": appt.scheduled_date,
                            "service_minutes": arrival.service_minutes,
                            "total_minutes_on_day": doc.schedule.get(appt.scheduled_date, 0),
                            "specialty": appt.specialty,
                            "is_working_day": doc.is_working_day(appt.scheduled_date),
                            "year_week": f"{appt.scheduled_date.isocalendar().year}-{appt.scheduled_date.isocalendar().week:02d}",
                        }
                    )

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

            for sp in ["family_practice", "internal_medicine", "pediatrics"]:
                if self.staffing.maybe_hire(doctors, sp):
                    new_doc = self.staffing._new_doctor(sp, arrival.arrival_date)
                    doctors.append(new_doc)
                    doctor_lookup[new_doc.doctor_id] = new_doc

        df = self._to_dataframe(appointments, patients, arrivals)
        df_doctor = pd.DataFrame(
            [
                {
                    "doctor_id": d.doctor_id,
                    "specialty": d.specialty,
                    "region": d.region,
                    "language": d.language,
                    "quality_score": d.quality_score,
                    "daily_minutes": d.daily_minutes,
                    "gender": d.gender,
                    "age": d.age,
                    "race": d.race,
                    "service_type": d.service_type,
                    "services_count": d.services_count,
                    "experience_years": d.experience_years,
                    "board_certified": d.board_certified,
                    "current_panel_size": d.current_panel_size,
                    "expected_workload": d.expected_workload,
                    "max_workload": d.max_workload,
                    "working_weeks_by_year": d.working_weeks_by_year,
                    "hires_at": d.hires_at,
                }
                for d in doctors
            ]
        )
        df_schedule = pd.DataFrame(schedule_log)
        if not df_schedule.empty:
            df_schedule.to_csv(self.data_dir / "schedule_log.csv", index=False)

        metrics = self._compute_metrics(df)
        return df, metrics, df_doctor, df_schedule

    def _quarter_index(self, start: date, current: date) -> int:
        months = (current.year - start.year) * 12 + (current.month - start.month)
        return months // 3

    def _to_dataframe(self, appointments: List[Appointment], patients: List[Patient], arrivals: List[Arrival]) -> pd.DataFrame:
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
                    "historical_visits": p.historical_visits,
                    "specialty_request": p.specialty_request,
                    "service_minutes": a.service_minutes,
                }
            )
        return pd.DataFrame.from_records(records)

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["fill_rate"] = df["allocated"].mean() if not df.empty else 0.0
        metrics["avg_wait_if_scheduled"] = (
            df[df["allocated"]]["wait_days"].mean() if not df[df["allocated"]].empty else 0.0
        )
        no_show_series = df["no_show"].dropna() if "no_show" in df.columns else pd.Series(dtype=float)
        metrics["no_show_rate"] = no_show_series.mean() if not no_show_series.empty else 0.0
        metrics["general_match_rate"] = (df["specialty"] == "general").mean() if "specialty" in df.columns else 0.0
        return metrics

