"""
Synthetic Data Generator for MedGemma Sentinel
Generates realistic medical data for testing and demonstration
"""

import random
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Any, Optional, Tuple
import uuid


class SyntheticDataGenerator:
    """
    Generator for synthetic medical data.
    
    Creates realistic patient data, vital signs, clinical events,
    and consultation data for testing the MedGemma Sentinel system.
    """
    
    # Sample data pools
    FIRST_NAMES_MALE = [
        "Jean", "Pierre", "Michel", "André", "Louis", "François", "Paul",
        "Henri", "Jacques", "Bernard", "Mohamed", "Mamadou", "Omar"
    ]
    
    FIRST_NAMES_FEMALE = [
        "Marie", "Jeanne", "Françoise", "Catherine", "Anne", "Monique",
        "Nicole", "Fatima", "Aïcha", "Aminata", "Mariam"
    ]
    
    LAST_NAMES = [
        "Dupont", "Martin", "Bernard", "Thomas", "Robert", "Richard",
        "Petit", "Durand", "Diallo", "Traoré", "Koné", "Camara", "Ba"
    ]
    
    CONDITIONS = [
        ("Hypertension artérielle", "I10", "chronic"),
        ("Diabète type 2", "E11", "chronic"),
        ("Insuffisance cardiaque", "I50", "chronic"),
        ("BPCO", "J44", "chronic"),
        ("Asthme", "J45", "active"),
        ("Insuffisance rénale chronique", "N18", "chronic"),
        ("Anémie", "D64", "active"),
        ("Fibrillation auriculaire", "I48", "chronic"),
        ("AVC ancien", "I63", "resolved"),
        ("Arthrose", "M15", "chronic")
    ]
    
    MEDICATIONS = [
        ("Amlodipine 5mg", "1x/jour"),
        ("Metformine 500mg", "2x/jour"),
        ("Furosémide 40mg", "1x/jour"),
        ("Aspirine 100mg", "1x/jour"),
        ("Oméprazole 20mg", "1x/jour"),
        ("Atorvastatine 20mg", "1x/jour"),
        ("Ramipril 5mg", "1x/jour"),
        ("Bisoprolol 2.5mg", "1x/jour"),
        ("Insuline Lantus", "1x/jour"),
        ("Salbutamol inh.", "si besoin")
    ]
    
    ALLERGIES = [
        ("Pénicilline", "severe", "Anaphylaxie"),
        ("Sulfamides", "moderate", "Éruption cutanée"),
        ("AINS", "moderate", "Asthme"),
        ("Iode", "mild", "Urticaire"),
        ("Latex", "moderate", "Dermatite")
    ]
    
    SYMPTOMS = {
        "cardio": [
            "Douleur thoracique", "Palpitations", "Dyspnée d'effort",
            "Œdème des membres inférieurs", "Syncope"
        ],
        "respiratoire": [
            "Toux", "Expectoration", "Dyspnée", "Sifflement",
            "Hémoptysie"
        ],
        "digestif": [
            "Douleur abdominale", "Nausées", "Vomissements",
            "Diarrhée", "Constipation"
        ],
        "neurologique": [
            "Céphalées", "Vertiges", "Confusion", "Faiblesse",
            "Troubles visuels"
        ],
        "general": [
            "Fièvre", "Fatigue", "Perte de poids", "Anorexie",
            "Sueurs nocturnes"
        ]
    }
    
    NIGHT_EVENT_TYPES = [
        ("desaturation", "SpO2 bas", "critical"),
        ("tachycardia", "Tachycardie", "high"),
        ("bradycardia", "Bradycardie", "high"),
        ("apnea", "Apnée détectée", "critical"),
        ("agitation", "Agitation nocturne", "medium"),
        ("fever", "Fièvre", "high"),
        ("abnormal_breathing", "Respiration anormale", "medium"),
    ]

    # Hard safety bounds for generated vitals (clinical realism guardrails).
    CLINICAL_BOUNDS = {
        "spo2": (70, 100),
        "heart_rate": (30, 220),
        "temperature": (34.0, 42.0),
        "respiratory_rate": (6, 45),
        "sbp": (60, 220),
        "dbp": (30, 130),
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility"""
        if seed is not None:
            random.seed(seed)

    @staticmethod
    def _clamp(value: float, low: float, high: float):
        """Clamp a generated value to strict clinical bounds."""
        return max(low, min(high, value))
    
    def generate_patient(
        self,
        patient_id: Optional[str] = None,
        age_range: Tuple[int, int] = (45, 85),
        condition_count: Tuple[int, int] = (1, 4),
        medication_count: Tuple[int, int] = (2, 6)
    ) -> Dict[str, Any]:
        """Generate a synthetic patient"""
        
        patient_id = patient_id or f"P{random.randint(1000, 9999)}"
        gender = random.choice(["male", "female"])
        
        first_names = self.FIRST_NAMES_MALE if gender == "male" else self.FIRST_NAMES_FEMALE
        name = f"{random.choice(first_names)} {random.choice(self.LAST_NAMES)}"
        
        age = random.randint(*age_range)
        dob = datetime.now() - timedelta(days=age*365 + random.randint(0, 364))
        
        # Generate conditions
        num_conditions = random.randint(*condition_count)
        conditions = random.sample(self.CONDITIONS, min(num_conditions, len(self.CONDITIONS)))
        
        # Generate medications
        num_meds = random.randint(*medication_count)
        medications = random.sample(self.MEDICATIONS, min(num_meds, len(self.MEDICATIONS)))
        
        # Maybe add allergies
        allergies = []
        if random.random() > 0.6:
            allergies = random.sample(self.ALLERGIES, random.randint(1, 2))
        
        # Risk factors based on conditions
        risk_factors = []
        if age > 65:
            risk_factors.append("Âge > 65 ans")
        if any("Diabète" in c[0] for c in conditions):
            risk_factors.append("Diabète")
        if any("Hypertension" in c[0] for c in conditions):
            risk_factors.append("HTA")
        if any("Insuffisance cardiaque" in c[0] for c in conditions):
            risk_factors.append("Insuffisance cardiaque")
        
        return {
            "patient_id": patient_id,
            "name": name,
            "gender": gender,
            "date_of_birth": dob.strftime("%Y-%m-%d"),
            "age": age,
            "room": f"{random.randint(1, 5)}{random.randint(0, 9):02d}",
            "bed": random.choice(["A", "B"]),
            "height_cm": random.randint(155, 185) if gender == "male" else random.randint(150, 175),
            "weight_kg": random.randint(55, 95) if gender == "male" else random.randint(45, 85),
            "blood_type": random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
            "conditions": [
                {
                    "name": c[0],
                    "icd_code": c[1],
                    "status": c[2]
                }
                for c in conditions
            ],
            "medications": [
                {
                    "name": m[0],
                    "frequency": m[1]
                }
                for m in medications
            ],
            "allergies": [
                {
                    "substance": a[0],
                    "severity": a[1],
                    "reaction": a[2]
                }
                for a in allergies
            ],
            "risk_factors": risk_factors,
            "admission_date": (datetime.now() - timedelta(days=random.randint(1, 7))).isoformat(),
            "admission_reason": random.choice([
                "Décompensation cardiaque",
                "Pneumopathie",
                "AEG",
                "Chute",
                "Surveillance post-opératoire"
            ])
        }
    
    def generate_vitals_reading(
        self,
        patient: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        scenario: str = "normal"
    ) -> Dict[str, Any]:
        """Generate a single vital signs reading"""
        
        timestamp = timestamp or datetime.now()
        
        # Base values modified by scenario
        if scenario == "normal":
            spo2 = random.randint(95, 99)
            hr = random.randint(60, 90)
            temp = round(random.uniform(36.2, 37.0), 1)
            rr = random.randint(14, 18)
            sbp = random.randint(110, 135)
            dbp = random.randint(65, 85)
        elif scenario == "desaturation":
            spo2 = random.randint(82, 89)
            hr = random.randint(90, 120)
            temp = round(random.uniform(36.5, 37.5), 1)
            rr = random.randint(22, 30)
            sbp = random.randint(100, 130)
            dbp = random.randint(60, 80)
        elif scenario == "tachycardia":
            spo2 = random.randint(92, 97)
            hr = random.randint(110, 150)
            temp = round(random.uniform(36.5, 38.0), 1)
            rr = random.randint(18, 24)
            sbp = random.randint(90, 120)
            dbp = random.randint(55, 75)
        elif scenario == "fever":
            spo2 = random.randint(93, 97)
            hr = random.randint(85, 110)
            temp = round(random.uniform(38.5, 40.0), 1)
            rr = random.randint(18, 26)
            sbp = random.randint(95, 125)
            dbp = random.randint(55, 75)
        elif scenario == "critical":
            spo2 = random.randint(75, 84)
            hr = random.randint(40, 55) if random.random() > 0.5 else random.randint(140, 180)
            temp = round(random.uniform(35.0, 35.5) if random.random() > 0.5 else random.uniform(39.5, 41.0), 1)
            rr = random.randint(8, 10) if random.random() > 0.5 else random.randint(30, 40)
            sbp = random.randint(70, 90)
            dbp = random.randint(40, 55)
        else:  # default
            spo2 = random.randint(93, 98)
            hr = random.randint(55, 100)
            temp = round(random.uniform(36.0, 37.5), 1)
            rr = random.randint(12, 22)
            sbp = random.randint(100, 145)
            dbp = random.randint(60, 90)

        # Enforce strict clinical bounds
        spo2 = int(self._clamp(spo2, *self.CLINICAL_BOUNDS["spo2"]))
        hr = int(self._clamp(hr, *self.CLINICAL_BOUNDS["heart_rate"]))
        temp = round(float(self._clamp(temp, *self.CLINICAL_BOUNDS["temperature"])), 1)
        rr = int(self._clamp(rr, *self.CLINICAL_BOUNDS["respiratory_rate"]))
        sbp = int(self._clamp(sbp, *self.CLINICAL_BOUNDS["sbp"]))
        dbp = int(self._clamp(dbp, *self.CLINICAL_BOUNDS["dbp"]))
        if dbp >= sbp:
            dbp = max(self.CLINICAL_BOUNDS["dbp"][0], sbp - 10)
        
        return {
            "timestamp": timestamp.isoformat(),
            "spo2": spo2,
            "heart_rate": hr,
            "temperature": temp,
            "respiratory_rate": rr,
            "blood_pressure": {
                "systolic": sbp,
                "diastolic": dbp
            },
            "source": "sensor"
        }
    
    def generate_night_vitals_timeline(
        self,
        patient: Dict[str, Any],
        duration_hours: int = 8,
        reading_interval_minutes: int = 15,
        anomaly_probability: float = 0.1,
        clinical_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a timeline of vitals readings for a night"""
        
        readings = []
        if clinical_date is None:
            base_date = datetime.now().date()
        elif isinstance(clinical_date, datetime):
            base_date = clinical_date.date()
        else:
            base_date = clinical_date
        start_time = datetime.combine(base_date, time(hour=22, minute=0))
        
        num_readings = (duration_hours * 60) // reading_interval_minutes
        
        for i in range(num_readings):
            timestamp = start_time + timedelta(minutes=i * reading_interval_minutes)
            
            # Determine scenario
            if random.random() < anomaly_probability:
                scenario = random.choice(["desaturation", "tachycardia", "fever"])
            else:
                scenario = "normal"
            
            reading = self.generate_vitals_reading(patient, timestamp, scenario)
            readings.append(reading)
        
        return readings
    
    def generate_audio_event(
        self,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate a synthetic audio event"""
        
        timestamp = timestamp or datetime.now()
        
        event_types = [
            ("apnea", 0.85, {"duration": random.randint(10, 30)}),
            ("stridor", 0.75, {}),
            ("wheeze", 0.80, {}),
            ("cough", 0.90, {"count": random.randint(3, 10)}),
            ("vocal_distress", 0.70, {}),
            ("snoring", 0.85, {})
        ]
        
        event_type, confidence, extra_data = random.choice(event_types)
        confidence = float(self._clamp(confidence + random.uniform(-0.1, 0.05), 0.0, 1.0))
        
        return {
            "timestamp": timestamp.isoformat(),
            "type": event_type,
            "confidence": round(confidence, 3),
            **extra_data
        }
    
    def generate_vision_event(
        self,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate a synthetic vision (IR camera) event"""
        
        timestamp = timestamp or datetime.now()
        
        event_types = [
            ("agitation", 0.80, {}),
            ("fall", 0.90, {}),
            ("abnormal_posture", 0.75, {}),
            ("sitting_up", 0.85, {}),
            ("leaving_bed", 0.88, {})
        ]
        
        event_type, confidence, extra_data = random.choice(event_types)
        confidence = float(self._clamp(confidence + random.uniform(-0.1, 0.05), 0.0, 1.0))
        
        return {
            "timestamp": timestamp.isoformat(),
            "type": event_type,
            "confidence": round(confidence, 3),
            **extra_data
        }
    
    def generate_night_scenario(
        self,
        patient: Dict[str, Any],
        scenario_type: str = "moderate",
        clinical_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete night surveillance scenario.
        
        Args:
            patient: Patient data
            scenario_type: "calm", "moderate", "eventful", "critical"
        """
        
        # Configure based on scenario
        configs = {
            "calm": {
                "anomaly_prob": 0.02,
                "audio_events": (0, 2),
                "vision_events": (0, 1)
            },
            "moderate": {
                "anomaly_prob": 0.08,
                "audio_events": (2, 5),
                "vision_events": (1, 3)
            },
            "eventful": {
                "anomaly_prob": 0.15,
                "audio_events": (4, 8),
                "vision_events": (2, 5)
            },
            "critical": {
                "anomaly_prob": 0.25,
                "audio_events": (5, 10),
                "vision_events": (3, 6)
            }
        }
        
        config = configs.get(scenario_type, configs["moderate"])
        
        # Generate vitals timeline
        if clinical_date is None:
            base_date = datetime.now().date()
        elif isinstance(clinical_date, datetime):
            base_date = clinical_date.date()
        else:
            base_date = clinical_date

        vitals = self.generate_night_vitals_timeline(
            patient,
            anomaly_probability=config["anomaly_prob"],
            clinical_date=base_date,
        )
        
        # Generate audio events
        audio_events = []
        num_audio = random.randint(*config["audio_events"])
        start_time = datetime.combine(base_date, time(hour=22, minute=0))
        for _ in range(num_audio):
            ts = start_time + timedelta(minutes=random.randint(0, 480))
            audio_events.append(self.generate_audio_event(ts))
        
        # Generate vision events
        vision_events = []
        num_vision = random.randint(*config["vision_events"])
        for _ in range(num_vision):
            ts = start_time + timedelta(minutes=random.randint(0, 480))
            vision_events.append(self.generate_vision_event(ts))
        
        return {
            "patient_id": patient["patient_id"],
            "patient_name": patient["name"],
            "room": patient["room"],
            "clinical_date": base_date.isoformat(),
            "scenario_type": scenario_type,
            "vitals_input": vitals,
            "audio_input": audio_events,
            "vision_input": vision_events,
            "patient_context": {
                "name": patient["name"],
                "age": patient["age"],
                "room": patient["room"],
                "conditions": [c["name"] for c in patient["conditions"]],
                "risk_factors": patient["risk_factors"]
            }
        }
    
    def generate_consultation_scenario(
        self,
        patient: Dict[str, Any],
        consultation_mode: str = "general",
        clinical_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Generate a day consultation scenario"""
        
        # Get symptoms based on mode
        symptoms_pool = self.SYMPTOMS.get(consultation_mode, self.SYMPTOMS["general"])
        symptoms = random.sample(symptoms_pool, random.randint(2, 4))
        
        # Generate vitals
        if clinical_date is None:
            base_date = datetime.now().date()
        elif isinstance(clinical_date, datetime):
            base_date = clinical_date.date()
        else:
            base_date = clinical_date

        consultation_ts = datetime.combine(base_date, time(hour=10, minute=0))
        vitals_reading = self.generate_vitals_reading(
            patient,
            timestamp=consultation_ts,
            scenario="normal",
        )
        
        # Physical exam
        exam = {}
        if consultation_mode == "cardio":
            exam = {
                "Auscultation cardiaque": random.choice([
                    "BDC réguliers, pas de souffle",
                    "Souffle systolique 2/6 au foyer aortique",
                    "BDC irréguliers, FA probable"
                ]),
                "Pouls périphériques": "Présents et symétriques",
                "OMI": random.choice(["Absents", "Légers bilatéraux", "Modérés"])
            }
        elif consultation_mode == "respiratoire":
            exam = {
                "Auscultation pulmonaire": random.choice([
                    "MV normal bilatéral",
                    "Sibilants diffus",
                    "Crépitants bases pulmonaires"
                ]),
                "SpO2 air ambiant": f"{vitals_reading['spo2']}%",
                "FR": f"{vitals_reading['respiratory_rate']}/min"
            }
        else:
            exam = {
                "État général": random.choice(["Bon", "Altéré", "Conservé"]),
                "Conscience": "Alerte et orienté",
                "Abdomen": "Souple, indolore"
            }
        
        return {
            "patient_id": patient["patient_id"],
            "patient_name": patient["name"],
            "clinical_date": base_date.isoformat(),
            "consultation_timestamp": consultation_ts.isoformat(),
            "consultation_mode": consultation_mode,
            "symptoms_input": symptoms,
            "day_vitals_input": {
                "SpO2": f"{vitals_reading['spo2']}%",
                "FC": f"{vitals_reading['heart_rate']} bpm",
                "T°": f"{vitals_reading['temperature']}°C",
                "PA": f"{vitals_reading['blood_pressure']['systolic']}/{vitals_reading['blood_pressure']['diastolic']} mmHg"
            },
            "exam_input": exam,
            "patient_context": {
                "name": patient["name"],
                "age": patient["age"],
                "conditions": [c["name"] for c in patient["conditions"]],
                "medications": [m["name"] for m in patient["medications"]],
                "allergies": [a["substance"] for a in patient["allergies"]]
            },
            "presenting_complaint": symptoms[0] if symptoms else "Consultation de suivi"
        }


# Convenience functions
def generate_demo_patient() -> Dict[str, Any]:
    """Generate a demo patient with realistic data"""
    gen = SyntheticDataGenerator()
    return gen.generate_patient(patient_id="DEMO001")


def generate_demo_night_scenario() -> Dict[str, Any]:
    """Generate a complete demo night scenario"""
    gen = SyntheticDataGenerator()
    patient = gen.generate_patient(patient_id="DEMO001")
    return gen.generate_night_scenario(patient, scenario_type="moderate")


def generate_demo_consultation() -> Dict[str, Any]:
    """Generate a demo consultation scenario"""
    gen = SyntheticDataGenerator()
    patient = gen.generate_patient(patient_id="DEMO001")
    return gen.generate_consultation_scenario(patient, consultation_mode="cardio")


if __name__ == "__main__":
    # Demo
    gen = SyntheticDataGenerator()
    
    print("=== Generating Demo Patient ===")
    patient = gen.generate_patient()
    print(f"Patient: {patient['name']}, {patient['age']} ans")
    print(f"Conditions: {[c['name'] for c in patient['conditions']]}")
    print(f"Medications: {[m['name'] for m in patient['medications']]}")
    print()
    
    print("=== Generating Night Scenario ===")
    night = gen.generate_night_scenario(patient, "moderate")
    print(f"Vitals readings: {len(night['vitals_input'])}")
    print(f"Audio events: {len(night['audio_input'])}")
    print(f"Vision events: {len(night['vision_input'])}")
    print()
    
    print("=== Generating Consultation ===")
    consultation = gen.generate_consultation_scenario(patient, "cardio")
    print(f"Mode: {consultation['consultation_mode']}")
    print(f"Symptoms: {consultation['symptoms_input']}")
