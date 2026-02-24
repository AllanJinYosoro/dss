"""
Synthetic data generation with seasonality and demand shocks.

Now supports persisting generated doctors/patients to disk and re-loading
so the rest of the system can run on prebuilt or externally supplied datasets.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from math import sin, pi
from pathlib import Path
from random import Random, uniform, choice, choices, randint, random
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

from .config import GENDERS, LANGUAGES, RACES, REGIONS, SERVICE_TYPES, AGE_GROUPS,NO_SHOW_BASE_RATE,SimulationConfig, PCP_SPECIALTIES,SPECIALTY_ABBREVIATIONS
from .models import Arrival, Doctor, Patient

DATA_DIR = "data"
PATIENTS_CSV = "patients.csv"
DOCTORS_CSV = "doctors.csv"
ARRIVALS_CSV = "arrivals.csv"


CLASS_DEFINITIONS = {
    ("M", "AD", "V1"): (3, 1.2205),
    ("M", "AD", "V2"): (float('inf'), 25.2923),
    ("F", "AD", "V1"): (10, 3.1106),
    ("F", "AD", "V2"): (float('inf'), 25.2923),
    ("F", "MA", "V1"): (2, 1.0585),
    ("F", "MA", "V2"): (float('inf'), 6.3564),
    ("M", "MA", "V1"): (10, 3.1929),
    ("M", "MA", "V2"): (float('inf'), 21.3801),
    ("M", "SE", "V1"): (12, 3.0342),
    ("M", "SE", "V2"): (float('inf'), 33.3108),
    ("F", "SE", "V1"): (10, 2.3556),
    ("F", "SE", "V2"): (float('inf'), 24.8231),
    ("M", "EL", "V1"): (10, 2.2273),
    ("M", "EL", "V2"): (float('inf'), 29.4249),
    ("F", "EL", "V1"): (6, 1.3654),
    ("F", "EL", "V2"): (float('inf'), 17.3586),
}

AGE_GROUP_RANGES = {
    "AD": (18, 40),
    "MA": (40, 60),
    "SE": (60, 75),
    "EL": (75, 100)
}

def _seasonal_multiplier(day_of_year, year, shock_type=None):
    tau = 2 * pi
    base = 1.0 + 0.15 * sin(2 * tau * day_of_year / 365) + 0.1 * sin(4 * tau * day_of_year / 365)
    
    if year == 1:  # Second year
        if day_of_year <= 182:  # First half
            if shock_type == "family_practice":
                base *= 1.3
        else:  # Second half
            if shock_type == "all":
                base *= 1.25
    
    return base

def _quarter_index(start_date, current_date):
    months_diff = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
    return months_diff // 3

def _classify_patient(gender, age, historical_visits):
    age_group = None
    for group, (min_age, max_age) in AGE_GROUP_RANGES.items():
        if min_age <= age < max_age:
            age_group = group
            break
    
    if age_group is None:
        age_group = "AD"
    
    visit_freq_group = "V1"
    for (g, ag, vfg), (threshold, _) in CLASS_DEFINITIONS.items():
        if g == gender and ag == age_group and vfg == "V1":
            if historical_visits <= threshold:
                visit_freq_group = "V1"
            else:
                visit_freq_group = "V2"
            break
    
    expected_visits = CLASS_DEFINITIONS.get((gender, age_group, visit_freq_group), (None, 1.0))[1]
    class_code = f"{gender}-{age_group}-{visit_freq_group}"
    
    return class_code, expected_visits

def _generate_historical_visits(age_group, gender):
    base_visits = {
        "AD": uniform(1, 5),
        "MA": uniform(3, 8),
        "SE": uniform(6, 15),
        "EL": uniform(8, 20)
    }
    
    adjustment = 1.2 if gender == "F" else 1.0
    randomness = uniform(0.8, 1.2)
    
    return base_visits[age_group] * adjustment * randomness


def _create_doctor(doctor_id, specialty, rng, cfg):
    base_quality = uniform(0.55, 0.92)
    if specialty == "internal_medicine":
        base_quality += 0.05
    elif specialty == "pediatrics":
        base_quality += 0.03
    
    experience = randint(1, 40)
    quality_adjustment = min(0.1, experience * 0.002)
    
    working_weeks_by_year = _sample_working_weeks_by_year(cfg, rng)

    return Doctor(
        doctor_id=doctor_id,
        specialty=specialty,
        region=choice(REGIONS),
        language=choice(LANGUAGES),
        quality_score=min(1.0, round(base_quality + quality_adjustment, 3)),
        daily_minutes=cfg.doctor_daily_minutes,
        gender=choice(GENDERS),
        age=randint(30, 65),
        race=choice(RACES),
        service_type=choice(SERVICE_TYPES),
        services_count=randint(1, 5),
        experience_years=experience,
        board_certified=random() > 0.1,
        current_panel_size=0,
        expected_workload=0.0,
        max_workload = cfg.doctor_work_weeks_per_year * 5 * cfg.doctor_daily_minutes,
        working_weeks_by_year=working_weeks_by_year,
    )


def _sample_working_weeks_by_year(cfg, rng):
    years = range(cfg.start_date.year, cfg.start_date.year + cfg.years)
    weeks = list(range(1, 53))
    count = max(1, min(52, cfg.doctor_work_weeks_per_year))
    return {int(y): sorted(rng.sample(weeks, count)) for y in years}

def generate_doctors(cfg):
    doctors = []
    rng = Random(cfg.seed + 999)
    doctor_id = 1
    
    for specialty, count in cfg.base_doctor_counts.items():
        for _ in range(count):
            doctors.append(_create_doctor(doctor_id, specialty, rng, cfg))
            doctor_id += 1
    
    return doctors


def generate_patients(cfg):
    rng = Random(cfg.seed)
    patients = []
    
    total_patients = int(cfg.patients_per_year * cfg.years)
    
    for patient_id in range(1, total_patients + 1):
        age_group = choices(list(AGE_GROUP_RANGES.keys()), weights=[0.35, 0.30, 0.22, 0.13])[0]
        min_age, max_age = AGE_GROUP_RANGES[age_group]
        age = randint(min_age, max_age - 1)
        
        gender = choice(GENDERS)
        race = choice(RACES)
        region = choice(REGIONS)
        language = choices(LANGUAGES, weights=[0.72, 0.18, 0.10])[0]
        
        historical_visits = _generate_historical_visits(age_group, gender)
        class_code, expected_visits = _classify_patient(gender, age, historical_visits)
        
        year_idx = patient_id // cfg.patients_per_year if cfg.patients_per_year > 0 else 0
        
        if year_idx == 0:
            specialty_weights = [0.5, 0.3, 0.2]
        else:
            if patient_id % cfg.patients_per_year < cfg.patients_per_year / 2:
                specialty_weights = [0.7, 0.2, 0.1]
            else:
                specialty_weights = [0.4, 0.35, 0.25]
        
        specialty_request = choices(PCP_SPECIALTIES, weights=specialty_weights)[0]
        
        
        preference_vector = {
            "region_bias": uniform(0.35, 0.9),
            "language_bias": uniform(0.1, 0.5),
            "quality_bias": uniform(0.2, 0.6),
            "gender_bias": uniform(0.05, 0.25),
            "race_bias": uniform(0.05, 0.25),
            "service_type_bias": uniform(0.05, 0.3),
            "service_count_bias": uniform(0.05, 0.3),
            "experience_bias": uniform(0.2, 0.6),
            "board_certification_bias":uniform(0.2,0.6)
        }
        
        total_weight = sum(preference_vector.values())
        preference_vector = {k: v/total_weight for k, v in preference_vector.items()}
        
        patient = Patient(
            patient_id=patient_id,
            age=age,
            age_group=age_group,
            gender=gender,
            race=race,
            region=region,
            language=language,
            historical_visits=historical_visits,
            cp=expected_visits,
            cp_group=class_code,
            specialty_request=specialty_request,
            allocated_doctor_id= None,
            preference_vector=preference_vector,
        )
        
        patients.append(patient)
    
    return patients



def generate_arrivals(cfg, patients):
    rng = Random(cfg.seed + 123)
    arrivals = []
    calendar = []
    
    days_total = cfg.years * 365
    patient_last_visit = {}
    
    for day_offset in range(days_total):
        current_date = cfg.start_date + timedelta(days=day_offset)
        calendar.append(current_date)
        
        day_of_year = current_date.timetuple().tm_yday
        year_idx = day_offset // 365
        
        for patient in patients:
            patient_start_day = (patient.patient_id % 365)
            if day_offset < patient_start_day:
                continue
            
            expected_visits = patient.cp #新改的
            
            seasonal_factor = _seasonal_multiplier(
                day_of_year, 
                year_idx,
                "family_practice" if year_idx == 1 and day_of_year <= 182 else "all" if year_idx == 1 else None
            )
            
            patient_factor = 0.8 + (patient.patient_id % 10) * 0.04
            daily_probability = (expected_visits * seasonal_factor * patient_factor) / 365.0
            
            last_visit = patient_last_visit.get(patient.patient_id)
            if last_visit and (current_date - last_visit).days < 7:
                daily_probability *= 0.1
            
            if rng.random() < min(daily_probability, 0.3):
                latest_gap = int(np.clip(np.random.normal(14, 5), 3, cfg.max_wait_days))
                latest_date = current_date + timedelta(days=latest_gap)
                
                base_risk = NO_SHOW_BASE_RATE
                age_adjustment = -0.02 if patient.age > 60 else 0.01 if patient.age < 30 else 0
                history_adjustment = -0.01 * min(patient.historical_visits / 10, 0.5)
                patient_risk_factor = (patient.patient_id % 20) * 0.005
                
                no_show_risk = max(0.01, min(0.4, 
                    base_risk + age_adjustment + history_adjustment + patient_risk_factor
                ))
                
                if patient.specialty_request == "family_practice":
                    service_minutes = randint(15, 25)
                elif patient.specialty_request == "internal_medicine":
                    service_minutes = randint(20, 30)
                else:
                    service_minutes = randint(15, 20)

                arrival = Arrival(
                    arrival_id=len(arrivals) + 1,
                    patient_id=patient.patient_id,
                    arrival_date=current_date,
                    latest_date=latest_date,
                    service_minutes=service_minutes,
                    specialty_request=patient.specialty_request,
                    no_show_risk=round(no_show_risk, 3),
                    patient_class=patient.cp_group,
                    expected_visits=patient.cp,
                )
                
                arrivals.append(arrival)
                patient_last_visit[patient.patient_id] = current_date
    
    return arrivals, calendar


# ---------------- Persistence helpers ----------------


# Data persistence functions
def patients_to_df(patients):
    records = []
    for p in patients:
        record = {
            "patient_id": p.patient_id,
            "age": p.age,
            "age_group": p.age_group,
            "gender": p.gender,
            "race": p.race,
            "region": p.region,
            "language": p.language,
            "historical_visits": p.historical_visits,
            "cp": p.cp,
            "cp_group": p.cp_group,
            "specialty_request": p.specialty_request,
            "allocated_doctor_id":p.allocated_doctor_id
            
        }
        for key, value in p.preference_vector.items():
            record[key] = value
        records.append(record)
    
    return pd.DataFrame.from_records(records)

def doctors_to_df(doctors):
    records = []
    for d in doctors:
        record = {
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
            "hires_at": d.hires_at,
            "current_panel_size": d.current_panel_size,
            "expected_workload": d.expected_workload,
            "max_workload":d.max_workload,
            "working_weeks_by_year": json.dumps(d.working_weeks_by_year, sort_keys=True),
        }
        records.append(record)
    
    return pd.DataFrame.from_records(records)

def arrivals_to_df(arrivals):
    records = []
    for a in arrivals:
        record = {
            "arrival_id": a.arrival_id,
            "patient_id": a.patient_id,
            "arrival_date": a.arrival_date,
            "latest_date": a.latest_date,
            "service_minutes": a.service_minutes,
            "specialty_request": a.specialty_request,
            "no_show_risk": a.no_show_risk,
            "patient_class": a.patient_class,
            "expected_visits": a.expected_visits,
        }
        records.append(record)
    
    return pd.DataFrame.from_records(records)

def save_data(patients, doctors, arrivals, out_dir=DATA_DIR):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Convert to absolute paths
    patients_path = os.path.join(out_dir, PATIENTS_CSV)
    doctors_path = os.path.join(out_dir, DOCTORS_CSV)
    arrivals_path = os.path.join(out_dir, ARRIVALS_CSV)
    
    # Save dataframes to CSV
    patients_df = patients_to_df(patients)
    doctors_df = doctors_to_df(doctors)
    arrivals_df = arrivals_to_df(arrivals)
    
    patients_df.to_csv(patients_path, index=False)
    doctors_df.to_csv(doctors_path, index=False)
    arrivals_df.to_csv(arrivals_path, index=False)
    
    print(f"Patients data saved to: {patients_path}")
    print(f"Doctors data saved to: {doctors_path}")
    print(f"Arrivals data saved to: {arrivals_path}")
    
    # Print some statistics
    print(f"\nData Summary:")
    print(f"  Total patients: {len(patients)}")
    print(f"  Total doctors: {len(doctors)}")
    print(f"  Total arrivals: {len(arrivals)}")
    
    if len(arrivals) > 0:
        print(f"  Date range: {arrivals[0].arrival_date} to {arrivals[-1].arrival_date}")
        #print(f"  Average arrivals per day: {len(arrivals) / (cfg.years * 365):.1f}")
    
    return patients_df, doctors_df, arrivals_df

def _patient_from_row(row):
    preference_keys = ["region_bias", "language_bias", "quality_bias", "gender_bias", 
                      "race_bias", "service_type_bias", "service_count_bias", 
                       "experience_bias","board_certification_bias"]
    
    preference_vector = {}
    for k in preference_keys:
        if k in row:
            preference_vector[k] = float(row[k])
    
    return Patient(
        patient_id=int(row["patient_id"]),
        age=int(row["age"]),
        age_group=str(row["age_group"]),
        gender=str(row["gender"]),
        race=str(row["race"]),
        region=str(row["region"]),
        language=str(row["language"]),
        historical_visits=float(row.get("historical_visits", 0)),
        cp=float(row["cp"]),
        cp_group=str(row["cp_group"]),
        specialty_request=str(row["specialty_request"]),
        allocated_doctor_id=row["allocated_doctor_id"],
        preference_vector=preference_vector,
    )

def _doctor_from_row(row):
    hires_at = None
    if pd.notna(row["hires_at"]) and str(row["hires_at"]).strip() != "":
        hires_at = pd.to_datetime(row["hires_at"]).date()
    
    parsed_weeks = json.loads(row["working_weeks_by_year"])
    working_weeks_by_year = {int(k): [int(w) for w in v] for k, v in parsed_weeks.items()}

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
        experience_years=int(row.get("experience_years", 10)),
        board_certified=bool(row.get("board_certified", True)),
        hires_at=hires_at,
        current_panel_size=int(row.get("current_panel_size", 0)),
        expected_workload=float(row.get("expected_workload", 0.0)),
        max_workload = float(row.get("max_workload",1560.0)),
        working_weeks_by_year=working_weeks_by_year,
    )

def _arrival_from_row(row):
    return Arrival(
        arrival_id=int(row["arrival_id"]),
        patient_id=int(row["patient_id"]),
        arrival_date=pd.to_datetime(row["arrival_date"]).date(),
        latest_date=pd.to_datetime(row["latest_date"]).date(),
        service_minutes=int(row["service_minutes"]),
        specialty_request=str(row["specialty_request"]),
        no_show_risk=float(row["no_show_risk"]),
        patient_class=str(row.get("patient_class", "")),
        expected_visits=float(row.get("expected_visits", 0)),
    )

def load_data(data_dir=DATA_DIR):
    patients_path = os.path.join(data_dir, PATIENTS_CSV)
    doctors_path = os.path.join(data_dir, DOCTORS_CSV)
    arrivals_path = os.path.join(data_dir, ARRIVALS_CSV)
    
    if not all(os.path.exists(p) for p in [patients_path, doctors_path, arrivals_path]):
        raise FileNotFoundError(f"Missing data files in {data_dir}")
    
    p_df = pd.read_csv(patients_path)
    d_df = pd.read_csv(doctors_path)
    a_df = pd.read_csv(arrivals_path)
    
    patients = [_patient_from_row(r) for _, r in p_df.iterrows()]
    doctors = [_doctor_from_row(r) for _, r in d_df.iterrows()]
    arrivals = [_arrival_from_row(r) for _, r in a_df.iterrows()]
    
    arrivals.sort(key=lambda a: a.arrival_date)
    
    return patients, doctors, arrivals

def analyze_data(patients, doctors, arrivals):
    """Analyze the generated data and print statistics."""
    print("\n=== Data Analysis ===")
    
    # Patient statistics
    print("\nPatient Statistics:")
    print(f"  Total patients: {len(patients)}")
    
    age_groups = {}
    genders = {}
    specialties = {}
    
    for p in patients:
        age_groups[p.age_group] = age_groups.get(p.age_group, 0) + 1
        genders[p.gender] = genders.get(p.gender, 0) + 1
        specialties[p.specialty_request] = specialties.get(p.specialty_request, 0) + 1
    
    print(f"  Age groups: {dict(age_groups)}")
    print(f"  Genders: {dict(genders)}")
    print(f"  Specialty requests: {dict(specialties)}")
    
    # Doctor statistics
    print("\nDoctor Statistics:")
    print(f"  Total doctors: {len(doctors)}")
    
    doc_specialties = {}
    for d in doctors:
        doc_specialties[d.specialty] = doc_specialties.get(d.specialty, 0) + 1
    
    print(f"  Specialties: {dict(doc_specialties)}")
    
    # Arrival statistics
    print("\nArrival Statistics:")
    print(f"  Total arrivals: {len(arrivals)}")
    
    if arrivals:
        arrival_dates = [a.arrival_date for a in arrivals]
        print(f"  Date range: {min(arrival_dates)} to {max(arrival_dates)}")
        
        # Count arrivals by month
        arrivals_by_month = {}
        for a in arrivals:
            month_key = f"{a.arrival_date.year}-{a.arrival_date.month:02d}"
            arrivals_by_month[month_key] = arrivals_by_month.get(month_key, 0) + 1
        
        print(f"  Average arrivals per day: {len(arrivals) / len(set(arrival_dates)):.1f}")
        
        # No-show statistics
        avg_no_show = np.mean([a.no_show_risk for a in arrivals])
        print(f"  Average no-show risk: {avg_no_show:.3f}")
    
    return {
        "total_patients": len(patients),
        "total_doctors": len(doctors),
        "total_arrivals": len(arrivals),
        "patient_age_groups": age_groups,
        "patient_genders": genders,
        "patient_specialties": specialties,
        "doctor_specialties": doc_specialties
    }

