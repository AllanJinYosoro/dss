"""
Synthetic data generation with seasonality and demand shocks.

Now supports persisting generated doctors/patients to disk and re-loading
so the rest of the system can run on prebuilt or externally supplied datasets.
"""

from __future__ import annotations

from datetime import date, timedelta
from math import sin, tau
from pathlib import Path
from random import Random
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import GENDERS, LANGUAGES, RACES, REGIONS, SERVICE_TYPES, SimulationConfig, SPECIALTIES
from .models import Arrival, Doctor, Patient

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PATIENTS_CSV = "patients.csv"
DOCTORS_CSV = "doctors.csv"


def _seasonal_multiplier(day_of_year: int) -> float:
    # Smooth annual seasonality: peaks mid-year, troughs in winter.
    return 1.0 + 0.2 * sin(2 * tau * day_of_year / 365)


def _quarter_index(start: date, current: date) -> int:
    months = (current.year - start.year) * 12 + (current.month - start.month)
    return months // 3


def generate_doctors(cfg: SimulationConfig) -> List[Doctor]:
    doctors: List[Doctor] = []
    rng = Random(cfg.seed + 999)
    doc_id = 1
    for specialty, count in cfg.base_doctor_counts.items():
        for _ in range(count):
            doctors.append(_new_doctor_from_rng(doc_id, specialty, rng, cfg))
            doc_id += 1
    return doctors


def _new_doctor_from_rng(doc_id: int, specialty: str, rng: Random, cfg: SimulationConfig) -> Doctor:
    return Doctor(
        doctor_id=doc_id,
        specialty=specialty,
        region=rng.choice(REGIONS),
        language=rng.choice(LANGUAGES),
        quality_score=round(rng.uniform(0.55, 0.92), 2),
        daily_minutes=cfg.doctor_daily_minutes,
        gender=rng.choice(GENDERS),
        age=int(np.clip(rng.normalvariate(45, 10), 28, 70)),
        race=rng.choice(RACES),
        service_type=rng.choice(SERVICE_TYPES),
        services_count=rng.randint(1, 5),
    )


AGE_BUCKETS = ["AD", "MA", "SE", "EL"]

CP_TABLE_HOURS = {
    "M-AD-V1": 1.2205,
    "M-AD-V2": 25.2923,
    "F-AD-V1": 3.1106,
    "F-AD-V2": 25.2923,
    "F-MA-V1": 1.0585,
    "F-MA-V2": 6.3564,
    "M-MA-V1": 3.1930,
    "M-MA-V2": 21.3801,
    "M-SE-V1": 3.0342,
    "M-SE-V2": 33.3108,
    "F-SE-V1": 2.3556,
    "F-SE-V2": 24.8231,
    "M-EL-V1": 2.2273,
    "M-EL-V2": 29.4249,
    "F-EL-V1": 1.3654,
    "F-EL-V2": 17.3586,
}

def _class_from_demographics(gender: str, age_group: str, visit_freq: str) -> str:
    vflag = "V2" if visit_freq == "high" else "V1"
    return f"{gender}-{age_group}-{vflag}"

def _service_minutes(cp_hours: float) -> int:
    # Derive per-visit consumption: scale hours then clamp to [15, 90] minutes
    minutes = cp_hours * 60 / max(1, cp_hours * 2) * 30  # keep variability mild
    # fallback simpler: proportional to cp_hours*2
    minutes = cp_hours * 2 * 60 / 2
    minutes = max(15, min(90, int(round(minutes))))
    return minutes


def _draw_patient(
    patient_id: int, rng: Random, cfg: SimulationConfig, year_idx: int
) -> Patient:
    age_group = rng.choices(AGE_BUCKETS, weights=[0.35, 0.30, 0.22, 0.13])[0]
    gender = rng.choice(GENDERS)
    race = rng.choice(RACES)
    region = rng.choice(REGIONS)
    language = rng.choices(LANGUAGES, weights=[0.72, 0.18, 0.10])[0]

    specialty_request = rng.choice(SPECIALTIES)
    if year_idx == 1:
        # second year shock: pediatrics and internal gain mild lift
        specialty_request = rng.choices(
            SPECIALTIES, weights=[1.0, 1.1, 1.15], k=1
        )[0]

    visit_freq = "high" if rng.random() < 0.4 else "low"
    cp_group = _class_from_demographics(gender, age_group, visit_freq)
    cp_hours = CP_TABLE_HOURS[cp_group]
    service_minutes = _service_minutes(cp_hours)

    preference_vector = {
        "region_bias": rng.uniform(0.35, 0.9),
        "language_bias": rng.uniform(0.1, 0.5),
        "quality_bias": rng.uniform(0.2, 0.6),
        "gender_bias": rng.uniform(0.05, 0.25),
        "race_bias": rng.uniform(0.05, 0.25),
        "service_type_bias": rng.uniform(0.05, 0.3),
        "service_count_bias": rng.uniform(0.05, 0.3),
    }

    return Patient(
        patient_id=patient_id,
        age_group=age_group,
        gender=gender,
        race=race,
        region=region,
        language=language,
        visit_freq=visit_freq,
        specialty_request=specialty_request,
        preference_vector=preference_vector,
        cp_group=cp_group,
        cp_hours=cp_hours,
        service_minutes=service_minutes,
    )


def generate_patients(cfg: SimulationConfig) -> List[Patient]:
    rng = Random(cfg.seed)
    patients: List[Patient] = []
    # Generate a larger unique panel: years * patients_per_year (e.g., 2 years -> 60k)
    total_patients = int(cfg.patients_per_year * cfg.years)
    for i in range(total_patients):
        year_idx = 0
        patients.append(_draw_patient(i + 1, rng, cfg, year_idx))
    return patients


def generate_arrivals(cfg: SimulationConfig, patients: List[Patient]) -> Tuple[List[Arrival], List[date]]:
    rng = Random(cfg.seed + 123)
    arrivals: List[Arrival] = []
    calendar: List[date] = []

    days_total = cfg.years * 365
    for offset in range(days_total):
        current_day = cfg.start_date + timedelta(days=offset)
        calendar.append(current_day)
        doy = current_day.timetuple().tm_yday
        year_idx = offset // 365
        quarter_idx = _quarter_index(cfg.start_date, current_day)
        seasonal = _seasonal_multiplier(doy)

        for p in patients:
            # approximate expected visits per year from cp_hours (cp_hours*2 ~= visits)
            expected_visits = p.cp_hours * 2
            lam = expected_visits * seasonal / 365
            count = np.random.poisson(lam)
            for _ in range(count):
                latest_gap = int(np.clip(np.round(rng.normalvariate(14, 5)), 3, cfg.max_wait_days))
                latest_date = current_day + timedelta(days=latest_gap)
                service_minutes = p.service_minutes
                no_show_risk = min(0.4, 0.08 + rng.random() * 0.1)
                arrivals.append(
                    Arrival(
                        arrival_id=len(arrivals) + 1,
                        patient_id=p.patient_id,
                        arrival_date=current_day,
                        latest_date=latest_date,
                        service_minutes=service_minutes,
                        specialty_request=p.specialty_request,
                        no_show_risk=no_show_risk,
                    )
                )
    return arrivals, calendar


# ---------------- Persistence helpers ----------------


def patients_to_df(patients: List[Patient]) -> pd.DataFrame:
    records = []
    for p in patients:
        records.append(
            {
                "patient_id": p.patient_id,
                "age_group": p.age_group,
                "gender": p.gender,
                "race": p.race,
                "region": p.region,
                "language": p.language,
                "visit_freq": p.visit_freq,
                "specialty_request": p.specialty_request,
                "cp_group": p.cp_group,
                "cp_hours": p.cp_hours,
                "service_minutes": p.service_minutes,
                "region_bias": p.preference_vector["region_bias"],
                "language_bias": p.preference_vector["language_bias"],
                "quality_bias": p.preference_vector["quality_bias"],
                "gender_bias": p.preference_vector["gender_bias"],
                "race_bias": p.preference_vector["race_bias"],
                "service_type_bias": p.preference_vector["service_type_bias"],
                "service_count_bias": p.preference_vector["service_count_bias"],
            }
        )
    return pd.DataFrame.from_records(records)


def doctors_to_df(doctors: List[Doctor]) -> pd.DataFrame:
    records = []
    for d in doctors:
        records.append(
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
                "hires_at": d.hires_at,
            }
        )
    return pd.DataFrame.from_records(records)


def arrivals_to_df(arrivals: List[Arrival]) -> pd.DataFrame:
    records = []
    for a in arrivals:
        records.append(
            {
                "arrival_id": a.arrival_id,
                "patient_id": a.patient_id,
                "arrival_date": a.arrival_date,
                "latest_date": a.latest_date,
                "service_minutes": a.service_minutes,
                "specialty_request": a.specialty_request,
                "no_show_risk": a.no_show_risk,
            }
        )
    return pd.DataFrame.from_records(records)


def save_data(
    patients: List[Patient], doctors: List[Doctor], arrivals: List[Arrival], out_dir: Path = DEFAULT_DATA_DIR
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    patients_to_df(patients).to_csv(out_dir / PATIENTS_CSV, index=False)
    doctors_to_df(doctors).to_csv(out_dir / DOCTORS_CSV, index=False)
    arrivals_to_df(arrivals).to_csv(out_dir / "arrivals.csv", index=False)


def _patient_from_row(row: pd.Series) -> Patient:
    return Patient(
        patient_id=int(row["patient_id"]),
        age_group=str(row["age_group"]),
        gender=str(row["gender"]),
        race=str(row["race"]),
        region=str(row["region"]),
        language=str(row["language"]),
        visit_freq=str(row["visit_freq"]),
        specialty_request=str(row["specialty_request"]),
        cp_group=str(row["cp_group"]),
        cp_hours=float(row["cp_hours"]),
        service_minutes=int(row["service_minutes"]),
        preference_vector={
            "region_bias": float(row["region_bias"]),
            "language_bias": float(row["language_bias"]),
            "quality_bias": float(row["quality_bias"]),
            "gender_bias": float(row["gender_bias"]),
            "race_bias": float(row["race_bias"]),
            "service_type_bias": float(row["service_type_bias"]),
            "service_count_bias": float(row["service_count_bias"]),
        },
    )


def _doctor_from_row(row: pd.Series) -> Doctor:
    hires_at_val = pd.to_datetime(row["hires_at"]).date() if pd.notna(row["hires_at"]) else None
    return Doctor(
        doctor_id=int(row["doctor_id"]),
        specialty=str(row["specialty"]),
        region=str(row["region"]),
        language=str(row["language"]),
        quality_score=float(row["quality_score"]),
        daily_minutes=int(row["daily_minutes"]),
        gender=str(row["gender"]),
        age=int(row["age"]),
        race=str(row["race"]),
        service_type=str(row["service_type"]),
        services_count=int(row["services_count"]),
        hires_at=hires_at_val,
    )


def _arrival_from_row(row: pd.Series) -> Arrival:
    return Arrival(
        arrival_id=int(row["arrival_id"]),
        patient_id=int(row["patient_id"]),
        arrival_date=pd.to_datetime(row["arrival_date"]).date(),
        latest_date=pd.to_datetime(row["latest_date"]).date(),
        service_minutes=int(row["service_minutes"]),
        specialty_request=str(row["specialty_request"]),
        no_show_risk=float(row["no_show_risk"]),
    )


def load_data(data_dir: Path = DEFAULT_DATA_DIR) -> Tuple[List[Patient], List[Doctor], List[Arrival]]:
    patients_path = data_dir / PATIENTS_CSV
    doctors_path = data_dir / DOCTORS_CSV
    arrivals_path = data_dir / "arrivals.csv"
    if not (patients_path.exists() and doctors_path.exists() and arrivals_path.exists()):
        raise FileNotFoundError(f"Missing patients/doctors/arrivals CSV under {data_dir}")

    p_df = pd.read_csv(patients_path)
    d_df = pd.read_csv(doctors_path)
    a_df = pd.read_csv(arrivals_path)

    patients = [_patient_from_row(r) for _, r in p_df.iterrows()]
    doctors = [_doctor_from_row(r) for _, r in d_df.iterrows()]
    arrivals = [_arrival_from_row(r) for _, r in a_df.iterrows()]
    arrivals.sort(key=lambda a: a.arrival_date)
    return patients, doctors, arrivals
