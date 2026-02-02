#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
Synthetic data generation for medical facility DSS simulation.

"""

import os
from datetime import date, timedelta
from math import sin, pi
from pathlib import Path
from random import Random, uniform, choice, choices, randint, random
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# Constants
GENDERS = ["M", "F"]
LANGUAGES = ["en", "es", "zh"]
RACES = ["White", "Black", "Asian", "Hispanic", "Other"]
REGIONS = ["North", "South", "East", "West", "Central"]
SERVICE_TYPES = ["solo", "group"]
AGE_GROUPS = ["AD", "MA", "SE", "EL"]
NO_SHOW_BASE_RATE = 0.08
PCP_SPECIALTIES = ["family_practice", "internal_medicine", "pediatrics"]
SPECIALTY_ABBREVIATIONS = {
    "family_practice": "FP",
    "internal_medicine": "IM",
    "pediatrics": "PD"
}

# Default configuration
class SimulationConfig:
    def __init__(self):
        self.seed = 42
        self.years = 2
        self.patients_per_year = 30000
        self.start_date = date(2023, 1, 1)
        self.max_wait_days = 30
        self.doctor_daily_minutes = 360  # 6 hours
        self.base_doctor_counts = {
            "family_practice": 20,
            "internal_medicine": 10,
            "pediatrics": 8
        }
        self.avg_patients_per_doctor = 1500
        self.appointment_duration = 20  # minutes per appointment

# Models
class Patient:
    def __init__(self, patient_id, age, age_group, gender, race, region, language,
                 historical_visits, expected_visits_per_year, class_code,
                 specialty_request, service_minutes, preference_vector):
        self.patient_id = patient_id
        self.age = age
        self.age_group = age_group
        self.gender = gender
        self.race = race
        self.region = region
        self.language = language
        self.historical_visits = historical_visits
        self.expected_visits_per_year = expected_visits_per_year
        self.class_code = class_code
        self.specialty_request = specialty_request
        self.service_minutes = service_minutes
        self.preference_vector = preference_vector

class Doctor:
    def __init__(self, doctor_id, specialty, region, language, quality_score,
                 daily_minutes, gender, age, race, service_type, services_count,
                 experience_years, board_certified, hires_at=None,
                 current_panel_size=0, expected_workload=0.0):
        self.doctor_id = doctor_id
        self.specialty = specialty
        self.region = region
        self.language = language
        self.quality_score = quality_score
        self.daily_minutes = daily_minutes
        self.gender = gender
        self.age = age
        self.race = race
        self.service_type = service_type
        self.services_count = services_count
        self.experience_years = experience_years
        self.board_certified = board_certified
        self.hires_at = hires_at
        self.current_panel_size = current_panel_size
        self.expected_workload = expected_workload

class Arrival:
    def __init__(self, arrival_id, patient_id, arrival_date, latest_date,
                 service_minutes, specialty_request, no_show_risk,
                 patient_class="", expected_visits=0):
        self.arrival_id = arrival_id
        self.patient_id = patient_id
        self.arrival_date = arrival_date
        self.latest_date = latest_date
        self.service_minutes = service_minutes
        self.specialty_request = specialty_request
        self.no_show_risk = no_show_risk
        self.patient_class = patient_class
        self.expected_visits = expected_visits

# Constants
DATA_DIR = "data"
PATIENTS_CSV = "patients.csv"
DOCTORS_CSV = "doctors.csv"
ARRIVALS_CSV = "arrivals.csv"

# From Table 5.3 and 5.4 in the document
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
        
        if specialty_request == "family_practice":
            service_minutes = randint(15, 25)
        elif specialty_request == "internal_medicine":
            service_minutes = randint(20, 30)
        else:
            service_minutes = randint(15, 20)
        
        preference_vector = {
            "region_bias": uniform(0.35, 0.9),
            "language_bias": uniform(0.1, 0.5),
            "quality_bias": uniform(0.2, 0.6),
            "gender_bias": uniform(0.05, 0.25),
            "race_bias": uniform(0.05, 0.25),
            "service_type_bias": uniform(0.05, 0.3),
            "service_count_bias": uniform(0.05, 0.3),
            "wait_time_bias": uniform(0.3, 0.7),
            "experience_bias": uniform(0.2, 0.6),
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
            expected_visits_per_year=expected_visits,
            class_code=class_code,
            specialty_request=specialty_request,
            service_minutes=service_minutes,
            preference_vector=preference_vector,
        )
        
        patients.append(patient)
    
    return patients

def _create_doctor(doctor_id, specialty, rng, cfg, hire_date):
    base_quality = uniform(0.55, 0.92)
    if specialty == "internal_medicine":
        base_quality += 0.05
    elif specialty == "pediatrics":
        base_quality += 0.03
    
    experience = randint(1, 40)
    quality_adjustment = min(0.1, experience * 0.002)
    
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
        hires_at=hire_date,
        current_panel_size=0,
        expected_workload=0.0,
    )

def generate_doctors(cfg):
    doctors = []
    rng = Random(cfg.seed + 999)
    doctor_id = 1
    
    for specialty, count in cfg.base_doctor_counts.items():
        for _ in range(count):
            doctors.append(_create_doctor(doctor_id, specialty, rng, cfg, None))
            doctor_id += 1
    
    total_patients = cfg.patients_per_year * cfg.years
    doctors_needed = int(total_patients / cfg.avg_patients_per_doctor)
    additional_doctors = max(0, doctors_needed - len(doctors))
    
    if additional_doctors > 0:
        for i in range(additional_doctors):
            if i < additional_doctors * 0.5:
                hire_date = date(cfg.start_date.year + 1, 4, 1)
                specialty = "family_practice"
            else:
                hire_date = date(cfg.start_date.year + 1, 10, 1)
                specialty = choice(list(cfg.base_doctor_counts.keys()))
            
            doctors.append(_create_doctor(doctor_id, specialty, rng, cfg, hire_date))
            doctor_id += 1
    
    return doctors

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
            
            expected_visits = patient.expected_visits_per_year
            
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
                
                arrival = Arrival(
                    arrival_id=len(arrivals) + 1,
                    patient_id=patient.patient_id,
                    arrival_date=current_date,
                    latest_date=latest_date,
                    service_minutes=patient.service_minutes,
                    specialty_request=patient.specialty_request,
                    no_show_risk=round(no_show_risk, 3),
                    patient_class=patient.class_code,
                    expected_visits=patient.expected_visits_per_year,
                )
                
                arrivals.append(arrival)
                patient_last_visit[patient.patient_id] = current_date
    
    return arrivals, calendar

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
            "expected_visits_per_year": p.expected_visits_per_year,
            "class_code": p.class_code,
            "specialty_request": p.specialty_request,
            "service_minutes": p.service_minutes,
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
        print(f"  Average arrivals per day: {len(arrivals) / (cfg.years * 365):.1f}")
    
    return patients_df, doctors_df, arrivals_df

def _patient_from_row(row):
    preference_keys = ["region_bias", "language_bias", "quality_bias", "gender_bias", 
                      "race_bias", "service_type_bias", "service_count_bias", 
                      "wait_time_bias", "experience_bias"]
    
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
        expected_visits_per_year=float(row["expected_visits_per_year"]),
        class_code=str(row["class_code"]),
        specialty_request=str(row["specialty_request"]),
        service_minutes=int(row["service_minutes"]),
        preference_vector=preference_vector,
    )

def _doctor_from_row(row):
    hires_at = None
    if pd.notna(row["hires_at"]) and str(row["hires_at"]).strip() != "":
        hires_at = pd.to_datetime(row["hires_at"]).date()
    
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

# Example usage
if __name__ == "__main__":
    # Create configuration
    cfg = SimulationConfig()
    
    print("=== Medical Facility DSS Data Generation ===")
    print(f"Configuration:")
    print(f"  Years: {cfg.years}")
    print(f"  Patients per year: {cfg.patients_per_year}")
    print(f"  Start date: {cfg.start_date}")
    print(f"  Base doctors: {cfg.base_doctor_counts}")
    
    print("\nGenerating patients...")
    patients = generate_patients(cfg)
    print(f"Generated {len(patients)} patients")
    
    print("Generating doctors...")
    doctors = generate_doctors(cfg)
    print(f"Generated {len(doctors)} doctors")
    
    print("Generating arrivals...")
    arrivals, calendar = generate_arrivals(cfg, patients)
    print(f"Generated {len(arrivals)} arrivals over {len(calendar)} days")
    
    print("\nSaving data...")
    patients_df, doctors_df, arrivals_df = save_data(patients, doctors, arrivals)
    
    # Analyze the data
    stats = analyze_data(patients, doctors, arrivals)
    
    # Example: Load the data back
    print("\n=== Loading data back to verify ===")
    try:
        loaded_patients, loaded_doctors, loaded_arrivals = load_data()
        print(f"✓ Successfully loaded:")
        print(f"  Patients: {len(loaded_patients)}")
        print(f"  Doctors: {len(loaded_doctors)}")
        print(f"  Arrivals: {len(loaded_arrivals)}")
        
        # Verify data integrity
        if (len(patients) == len(loaded_patients) and 
            len(doctors) == len(loaded_doctors) and 
            len(arrivals) == len(loaded_arrivals)):
            print("✓ Data integrity verified")
        else:
            print("✗ Data integrity check failed")
            
    except Exception as e:
        print(f"✗ Error loading data: {e}")
    
    print("\n=== Data Generation Complete ===")
    print(f"Files saved in '{DATA_DIR}' directory:")
    print(f"  - {PATIENTS_CSV}")
    print(f"  - {DOCTORS_CSV}")
    print(f"  - {ARRIVALS_CSV}")


# In[ ]:




