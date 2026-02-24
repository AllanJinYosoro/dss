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
    generate_doctors,
    generate_patients,
    generate_arrivals,
    load_data,
    save_data,
)
from .models import Appointment, Doctor, Patient, QuarterState, Arrival
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

    def run(self) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
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
        doctor_lookup = {d.doctor_id: d for d in doctors}

        for arrival in arrivals:
            patient = patient_lookup[arrival.patient_id]
            specialty = self.allocator.pick_specialty(patient)
            if pd.isna(patient.allocated_doctor_id):  #第一次来访 分配PCP
                doctor_candidates = [d for d in doctors if d.specialty == specialty and d.expected_workload <d.max_workload]
                if doctor_candidates == []:
                    print('warning:',patient.patient_id)
                ranked = self.allocator.rank_doctors(patient, doctor_candidates, arrival.arrival_date)
                selected_doc_id = ranked[0].doctor_id
                original_doctor_obj = doctor_lookup[selected_doc_id]
        # 修改原始对象
                patient.allocated_doctor_id = selected_doc_id
                original_doctor_obj.current_panel_size += 1 
                original_doctor_obj.expected_workload += patient.cp
                

            prim_id = patient.allocated_doctor_id
            doctor_candidates = [d for d in doctors if d.specialty == specialty]
            primary_doc = next((d for d in doctor_candidates if d.doctor_id == prim_id), None)
            """ 
            others = [d for d in doctor_candidates if d.doctor_id != prim_id]
            ranked_others = self.allocator.rank_doctors(patient, others, arrival.arrival_date)
            ordered = ([primary_doc] if primary_doc else []) + ranked_others """

            q_idx = self._quarter_index(calendar_start, arrival.arrival_date)
            if q_idx != quarter_state.quarter_index:
                # finalize prior quarter no-show estimate
                if quarter_seen > 0:
                    quarter_state.no_show_rate = quarter_no_show / quarter_seen
                quarter_state.cp_bias = 0.0
                
                quarter_turnaways = defaultdict(int)
                quarter_bookings = defaultdict(int)
                quarter_no_show = 0
                quarter_seen = 0
                quarter_state.quarter_index = q_idx
            
            #需要改动
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

            
            for specialty in ['family_practice','internal_medicine','pediatrics']:
                if self.staffing.maybe_hire(doctors,specialty):
                    doc = self.staffing._new_doctor(specialty,arrival.arrival_date)
                    doctors.append(doc)
                    doctor_lookup[doc.doctor_id] = doc

        df = self._to_dataframe(appointments, patients, arrivals)

        data_doctor = [
            {
                "doctor_id": doc.doctor_id,
                "specialty":doc.specialty,
                "region":doc.region,
                "language":doc.language,
                "quality_score":doc.quality_score,
                "daily_minutes":doc.daily_minutes,
                "gender":doc.gender,
                "age":doc.age,
                "race":doc.race,
                "service_type":doc.service_type,
                "services_count":doc.services_count,
                "experience_years":doc.experience_years,
                "board_certified":doc.board_certified,
                "current_panel_size":doc.current_panel_size,
                "expected_workload": doc.expected_workload,
                "max_workload": doc.max_workload,
                "hires_at":doc.hires_at
            } 
            for doc in doctors
        ]

        # 直接创建 DataFrame
        df_doctor = pd.DataFrame(data_doctor)

        metrics = self._compute_metrics(df)
        return df, metrics,df_doctor

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
                    "historical_visits": p.historical_visits,
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
