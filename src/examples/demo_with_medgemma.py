"""
MedGemma Sentinel with Integrated LLM - Complete Demo
Demonstrates full workflow with MedGemma model for report generation
"""

import json
from datetime import datetime, date
from pathlib import Path

# Import all modules
from src.models.patient import Patient, Gender, Condition, Medication, Allergy
from src.models.vitals import SpO2Reading, HeartRateReading, TemperatureReading, BloodPressureReading, VitalStatus
from src.models.events import ClinicalEvent, EventType, AlertLevel, NightEvent
from src.orchestration.graph import MedGemmaSentinelGraph
from src.orchestration.state import SentinelState, WorkflowPhase, NightData, DayData
from src.memory.patient_graph import PatientGraphRAG
from src.memory.retriever import GraphRetriever
from src.reporting.pdf_generator import PDFReportGenerator
from src.reporting.templates import NightReportTemplate, DayReportTemplate
from src.reporting.prompts import MedGemmaReportGenerator


def create_demo_patient() -> Patient:
    """Create a demo patient for testing"""
    return Patient(
        id="MEDGEMMA_001",
        name="Jean Camara",
        date_of_birth=date(1952, 5, 15),
        gender=Gender.MALE,
        height_cm=170,
        weight_kg=75,
        blood_type="A+",
        room="500",
        bed="A",
        conditions=[
            Condition(name="Hypertension arterielle", icd_code="I10", status="active"),
            Condition(name="Diabete type 2", icd_code="E11", status="active"),
            Condition(name="BPCO stade II", icd_code="J44", status="chronic"),
        ],
        allergies=[
            Allergy(substance="Penicilline", severity="severe", reaction="Anaphylaxie"),
        ],
        medications=[
            Medication(name="Amlodipine", dosage="5mg", frequency="1x/day", route="oral"),
            Medication(name="Metformine", dosage="500mg", frequency="2x/day", route="oral"),
            Medication(name="Spiriva", dosage="18mcg", frequency="1x/day", route="inhaled"),
        ],
        admission_date=datetime.now(),
        admission_reason="Suivi medical regulier",
        attending_physician="Dr. Martin",
        risk_factors=["Age > 65 ans", "Obesite", "Sedentarite"],
    )


def simulate_night_events() -> tuple[list, list]:
    """Simulate night surveillance events and vitals"""
    events = [
        NightEvent(
            id="EVT001",
            patient_id="MEDGEMMA_001",
            event_type=EventType.DESATURATION,
            timestamp=datetime.now(),
            alert_level=AlertLevel.HIGH,
            description="SpO2 bas: 86%",
        ),
        NightEvent(
            id="EVT002",
            patient_id="MEDGEMMA_001",
            event_type=EventType.FEVER,
            timestamp=datetime.now(),
            alert_level=AlertLevel.HIGH,
            description="Fievre: 39.5C",
        ),
        NightEvent(
            id="EVT003",
            patient_id="MEDGEMMA_001",
            event_type=EventType.AGITATION,
            timestamp=datetime.now(),
            alert_level=AlertLevel.MEDIUM,
            description="Agitation anormale detectee",
        ),
        NightEvent(
            id="EVT004",
            patient_id="MEDGEMMA_001",
            event_type=EventType.APNEA,
            timestamp=datetime.now(),
            alert_level=AlertLevel.CRITICAL,
            description="Apnee detectee (11s)",
        ),
    ]
    
    vitals = [
        SpO2Reading(value=95, timestamp=datetime.now()),
        HeartRateReading(value=75, timestamp=datetime.now()),
        TemperatureReading(value=36.8, timestamp=datetime.now()),
        BloodPressureReading(systolic=135, diastolic=82, timestamp=datetime.now()),
    ]
    
    return events, vitals


def run_complete_workflow_with_medgemma():
    """
    Run complete MedGemma Sentinel workflow with LLM integration
    """
    print("\n" + "="*70)
    print("MedGemma Sentinel - Complete Workflow with LLM Integration")
    print("="*70)
    
    # 1. Initialize components
    print("\n[INIT] Initializing components...")
    patient = create_demo_patient()
    graph = MedGemmaSentinelGraph()
    memory = PatientGraphRAG()
    retriever = GraphRetriever(memory)
    pdf_gen = PDFReportGenerator()
    medgemma_gen = MedGemmaReportGenerator()
    
    print(f"[OK] MedGemma Engine Status: {medgemma_gen.get_engine_status()['mode']}")
    
    # 2. Add patient to memory
    print("\n[MEMORY] Adding patient to knowledge graph...")
    memory.add_patient(
        patient_id=patient.id,
        name=patient.name,
        age=patient.age,
        conditions=[c.name for c in patient.conditions],
        medications=[m.name for m in patient.medications],
        allergies=[a.substance for a in patient.allergies],
        risk_factors=patient.risk_factors,
        room=patient.room or "Unknown"
    )
    print(f"[OK] Patient {patient.name} added to memory")
    
    # 3. Simulate night surveillance
    print("\n[NIGHT] Simulating night surveillance...")
    events, vitals = simulate_night_events()
    
    night_data = NightData(
        events=[
            {
                "id": e.id,
                "type": e.event_type.value,
                "severity": e.alert_level.value,
                "description": e.description,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in events
        ],
        alerts_triggered=len([e for e in events if e.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]]),
        critical_alerts=len([e for e in events if e.alert_level == AlertLevel.CRITICAL]),
        vitals_readings=[
            {"spo2": 95, "hr": 75, "temp": 36.8}
        ]
    )
    
    print(f"[OK] Detected {len(events)} events, {night_data.critical_alerts} critical")
    
    # 4. Generate Night Report with MedGemma
    print("\n[RAP1] Generating night report with MedGemma...")
    
    patient_context = {
        "name": patient.name,
        "id": patient.id,
        "age": patient.age,
        "conditions": [c.name for c in patient.conditions],
        "medications": [m.name for m in patient.medications],
    }
    
    night_report_content = medgemma_gen.generate_night_report(
        patient_context=patient_context,
        night_data=night_data,
    )
    
    # Save markdown report
    night_report_file = pdf_gen.output_dir / f"rap1_night_{patient.id}.md"
    with open(night_report_file, 'w', encoding='utf-8') as f:
        f.write(night_report_content)
    print(f"[OK] Night report (Markdown): {night_report_file.name}")
    
    # Generate PDF from markdown
    try:
        night_pdf_file = pdf_gen.generate_night_report({
            "patient_id": patient.id,
            "patient_name": patient.name,
            "room": patient.room,
            "summary": night_report_content[:500],
            "events": [e.__dict__ for e in events],
            "night_data": night_data.__dict__,
            "vitals_summary": {"SpO2": {"min": 88, "max": 98, "avg": 95}},
            "recommendations": ["Surveillance SpO2", "Reevaluation clinique"]
        })
        print(f"[OK] Night report (PDF): {Path(night_pdf_file).name}")
    except Exception as e:
        print(f"[WARN] PDF generation skipped: {e}")
    
    # 5. Simulate day consultation
    print("\n[DAY] Simulating day consultation...")
    
    day_data = DayData(
        consultation_mode="cardio",
        symptoms=["Palpitations", "Dyspnee d'effort", "Syncope"],
        presenting_complaint="Douleur thoracique",
        differential_diagnosis=["Angine stable", "Douleur parietale"],
        exam_findings={
            "Auscultation": "BDC reguliers, pas de souffle",
            "Pouls": "Presents et symetriques",
            "OMI": "Absents"
        }
    )
    
    print(f"[OK] Consultation mode: {day_data.consultation_mode}")
    print(f"[OK] Symptoms: {', '.join(day_data.symptoms)}")
    
    # 6. Generate Day Report with MedGemma
    print("\n[RAP2] Generating consultation report with MedGemma...")
    
    day_report_content = medgemma_gen.generate_day_report(
        patient_context=patient_context,
        night_context=night_report_content[:300],
        day_data=day_data.__dict__,
        specialty="cardio",
    )
    
    # Save markdown report
    day_report_file = pdf_gen.output_dir / f"rap2_day_{patient.id}.md"
    with open(day_report_file, 'w', encoding='utf-8') as f:
        f.write(day_report_content)
    print(f"[OK] Day report (Markdown): {day_report_file.name}")
    
    # Generate PDF from markdown
    try:
        day_pdf_file = pdf_gen.generate_day_report({
            "patient_id": patient.id,
            "patient_name": patient.name,
            "room": patient.room,
            "summary": day_report_content[:500],
            "events": [],
            "night_context": "Nuit avec alertes detectees",
            "day_data": day_data.__dict__,
            "recommendations": ["ECG", "Dosage troponines"]
        })
        print(f"[OK] Day report (PDF): {Path(day_pdf_file).name}")
    except Exception as e:
        print(f"[WARN] PDF generation skipped: {e}")
    
    # 7. Workflow summary
    print("\n" + "="*70)
    print("[SUCCESS] WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"""
Reports Generated:
- Night Surveillance (RAP1): {night_report_file.name}
- Day Consultation (RAP2): {day_report_file.name}

MedGemma Engine Status:
- Mode: {medgemma_gen.get_engine_status()['mode']}
- Model: {medgemma_gen.get_engine_status()['model_path'] or 'Simulated'}
- Temperature: {medgemma_gen.get_engine_status()['temperature']}
- Max Tokens: {medgemma_gen.get_engine_status()['max_tokens']}

Patient Memory:
- Patient ID: {patient.id}
- Stored conditions: {len(patient.conditions)}
- Stored medications: {len(patient.medications)}
- Graph nodes created

Next Steps:
1. To launch everything from Python:
   python launch.py
2. To start server only:
   python launch.py --server
3. To run demo only (server must be running):
   python launch.py --demo
    """)


if __name__ == "__main__":
    run_complete_workflow_with_medgemma()
