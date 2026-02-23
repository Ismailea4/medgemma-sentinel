"""
MedGemma Sentinel - Complete Demo Workflow
Demonstrates the full Night -> Rap1 -> Day -> Rap2 workflow
with synthetic data and all components.
"""

import sys
import os
import json
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import components
from src.orchestration import MedGemmaSentinelGraph, create_sentinel_graph
from src.memory import PatientGraphRAG, GraphRetriever
from src.reporting import (
    MedGemmaPrompts, PromptType,
    NightReportTemplate, DayReportTemplate,
    PDFReportGenerator
)
from data.synthetic.data_generator import SyntheticDataGenerator

try:
    from src.reporting.medgemma_engine import MedGemmaEngine
    MEDGEMMA_ENGINE_AVAILABLE = True
except ImportError:
    MedGemmaEngine = None  # type: ignore[assignment]
    MEDGEMMA_ENGINE_AVAILABLE = False


def print_header(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---\n")


def load_or_create_patient_profile(patient_id: str = "DEMO001") -> dict:
    """
    Load a non-random patient profile from JSON (or create it once if missing).
    """
    patients_dir = Path("./data/patients")
    patients_dir.mkdir(parents=True, exist_ok=True)
    profile_path = patients_dir / f"{patient_id}_profile.json"

    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    profile = {
        "patient_id": patient_id,
        "name": "Jean Camara",
        "gender": "male",
        "date_of_birth": "1963-05-15",
        "age": 62,
        "room": "500",
        "bed": "A",
        "height_cm": 170,
        "weight_kg": 78,
        "blood_type": "A+",
        "conditions": [
            {"name": "Insuffisance cardiaque", "icd_code": "I50", "status": "chronic"},
            {"name": "DiabÃ¨te type 2", "icd_code": "E11", "status": "chronic"},
        ],
        "medications": [
            {"name": "Metformine 500mg", "frequency": "2x/jour"},
            {"name": "Ramipril 5mg", "frequency": "1x/jour"},
            {"name": "Amlodipine 5mg", "frequency": "1x/jour"},
            {"name": "Bisoprolol 2.5mg", "frequency": "1x/jour"},
            {"name": "Insuline Lantus", "frequency": "1x/jour"},
            {"name": "Salbutamol inh.", "frequency": "si besoin"},
        ],
        "allergies": [
            {"substance": "PÃ©nicilline", "severity": "moderate", "reaction": "Ã‰ruption cutanÃ©e"},
        ],
        "risk_factors": ["DiabÃ¨te", "Insuffisance cardiaque"],
        "admission_date": datetime.now().isoformat(),
        "admission_reason": "Surveillance cardio-respiratoire",
    }

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return profile


def save_session_json(
    patient_data: dict,
    night_data: dict,
    consultation_data: dict,
    workflow_result: dict | None = None,
    clinical_date: str | None = None,
    day_index: int = 1,
) -> Path:
    """
    Persist one full run (static patient profile + random session data) to JSON.
    """
    sessions_dir = Path("./data/sessions")
    sessions_dir.mkdir(parents=True, exist_ok=True)

    session_ts = datetime.now().strftime("%H%M%S")
    clinical_day = clinical_date or datetime.now().strftime("%Y%m%d")
    clinical_day_safe = clinical_day.replace("-", "")
    session_file = sessions_dir / (
        f"session_{patient_data['patient_id']}_{clinical_day_safe}_d{day_index:02d}_{session_ts}.json"
    )

    payload = {
        "session_id": f"{clinical_day_safe}_d{day_index:02d}_{session_ts}",
        "created_at": datetime.now().isoformat(),
        "clinical_date": clinical_date,
        "day_index": day_index,
        "patient_profile": patient_data,
        "generated_session_data": {
            "night_scenario": night_data,
            "day_consultation": consultation_data,
        },
        "workflow_summary": None,
    }

    if workflow_result:
        payload["workflow_summary"] = {
            "phase": workflow_result.get("phase"),
            "guardrails_blocked": workflow_result.get("guardrails_blocked"),
            "total_events_processed": workflow_result.get("total_events_processed"),
            "total_alerts": workflow_result.get("total_alerts"),
            "guard_log": workflow_result.get("guard_log", []),
            "rap1_title": (workflow_result.get("rap1_report") or {}).get("title"),
            "rap2_title": (workflow_result.get("rap2_report") or {}).get("title"),
        }

    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    return session_file


def _safe_min(values: list[float]) -> float | None:
    """Return min value or None for empty list."""
    return min(values) if values else None


def _safe_max(values: list[float]) -> float | None:
    """Return max value or None for empty list."""
    return max(values) if values else None


def _build_session_snapshot(session_payload: dict, session_file: Path) -> dict:
    """Build compact metrics from one session JSON for longitudinal analysis."""
    generated = session_payload.get("generated_session_data", {})
    night = generated.get("night_scenario", {})
    consultation = generated.get("day_consultation", {})
    workflow = session_payload.get("workflow_summary", {})

    vitals_input = night.get("vitals_input", [])
    spo2_values = [float(v["spo2"]) for v in vitals_input if isinstance(v, dict) and "spo2" in v]
    hr_values = [float(v["heart_rate"]) for v in vitals_input if isinstance(v, dict) and "heart_rate" in v]
    temp_values = [float(v["temperature"]) for v in vitals_input if isinstance(v, dict) and "temperature" in v]

    return {
        "session_id": session_payload.get("session_id", session_file.stem),
        "session_file": session_file.name,
        "created_at": session_payload.get("created_at"),
        "clinical_date": session_payload.get("clinical_date"),
        "day_index": session_payload.get("day_index"),
        "scenario_type": night.get("scenario_type"),
        "total_alerts": int(workflow.get("total_alerts", 0) or 0),
        "total_events_processed": int(workflow.get("total_events_processed", 0) or 0),
        "audio_events": len(night.get("audio_input", [])),
        "vision_events": len(night.get("vision_input", [])),
        "spo2_min": _safe_min(spo2_values),
        "spo2_max": _safe_max(spo2_values),
        "heart_rate_max": _safe_max(hr_values),
        "temperature_max": _safe_max(temp_values),
        "symptoms": consultation.get("symptoms_input", []),
        "presenting_complaint": consultation.get("presenting_complaint", ""),
        "consultation_mode": consultation.get("consultation_mode", "general"),
    }


def load_recent_session_snapshots(patient_id: str, limit: int = 2) -> list[dict]:
    """
    Load and compact the most recent patient sessions.

    Returns snapshots in chronological order (oldest -> newest).
    """
    sessions_dir = Path("./data/sessions")
    if not sessions_dir.exists():
        return []

    candidates = sorted(sessions_dir.glob(f"session_{patient_id}_*.json"))
    parsed: list[dict] = []

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            parsed.append({
                "path": path,
                "payload": payload,
                "created_at": payload.get("created_at", ""),
            })
        except Exception:
            continue

    parsed.sort(key=lambda item: (item["created_at"], item["path"].name))
    recent = parsed[-limit:]
    return [_build_session_snapshot(item["payload"], item["path"]) for item in recent]


def analyze_recent_history_with_medgemma(
    patient_id: str,
    medgemma_engine: "MedGemmaEngine | None" = None,
) -> dict:
    """
    Analyze the last 2 sessions using threshold-based comparison.

    Produces a deterministic, guardrail-safe evolution report by comparing
    vitals trends, alert counts, and symptoms across consecutive sessions.
    The medgemma_engine parameter is kept for API compatibility but unused.
    """
    snapshots = load_recent_session_snapshots(patient_id=patient_id, limit=2)
    sources = [s.get("session_file", "") for s in snapshots if s.get("session_file")]

    if len(snapshots) < 2:
        return {
            "analysis_markdown": (
                "## Evolution Inter-Cycles\n\n"
                "- Historique insuffisant: 2 sessions sont necessaires pour une analyse evolutive."
            ),
            "sources": sources,
            "status": "insufficient_history",
        }

    prev, curr = snapshots[-2], snapshots[-1]

    # ---- Threshold-based trend analysis ----
    lines: list[str] = []
    vigilance: list[str] = []

    # -- SpO2 --
    prev_spo2 = prev.get("spo2_min")
    curr_spo2 = curr.get("spo2_min")
    if prev_spo2 is not None and curr_spo2 is not None:
        delta = round(curr_spo2 - prev_spo2, 1)
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        lines.append(f"- **SpO2 min**: {prev_spo2}% → {curr_spo2}% ({arrow} {abs(delta)}%)")
        if curr_spo2 < 90:
            vigilance.append("SpO2 min critique (<90%): evaluation oxygenation urgente.")
        elif curr_spo2 < 92:
            vigilance.append("SpO2 min basse (<92%): surveillance rapprochee recommandee.")
        if delta < -2:
            vigilance.append(f"Degradation SpO2 significative ({delta}%): rechercher cause.")
    elif curr_spo2 is not None:
        lines.append(f"- **SpO2 min**: {curr_spo2}% (session precedente: N/A)")

    # -- Heart Rate --
    prev_hr = prev.get("heart_rate_max")
    curr_hr = curr.get("heart_rate_max")
    if prev_hr is not None and curr_hr is not None:
        delta_hr = round(curr_hr - prev_hr, 1)
        arrow = "↑" if delta_hr > 0 else ("↓" if delta_hr < 0 else "→")
        lines.append(f"- **FC max**: {prev_hr} → {curr_hr} bpm ({arrow} {abs(delta_hr)})")
        if curr_hr > 110:
            vigilance.append("Tachycardie persistante (FC >110): bilan cardiaque a envisager.")
        elif curr_hr > 100:
            vigilance.append("FC elevee (>100 bpm): surveiller evolution.")
        if delta_hr > 10:
            vigilance.append(f"Augmentation FC notable (+{delta_hr} bpm).")
    elif curr_hr is not None:
        lines.append(f"- **FC max**: {curr_hr} bpm (session precedente: N/A)")

    # -- Temperature --
    prev_temp = prev.get("temperature_max")
    curr_temp = curr.get("temperature_max")
    if prev_temp is not None and curr_temp is not None:
        delta_t = round(curr_temp - prev_temp, 1)
        arrow = "↑" if delta_t > 0 else ("↓" if delta_t < 0 else "→")
        lines.append(f"- **T max**: {prev_temp}°C → {curr_temp}°C ({arrow} {abs(delta_t)}°C)")
        if curr_temp >= 38.5:
            vigilance.append("Fievre significative (≥38.5°C): rechercher foyer infectieux.")
        elif curr_temp >= 37.8:
            vigilance.append("Subfebrile (≥37.8°C): surveiller evolution thermique.")
    elif curr_temp is not None:
        lines.append(f"- **T max**: {curr_temp}°C (session precedente: N/A)")

    # -- Alerts --
    prev_alerts = prev.get("total_alerts", 0)
    curr_alerts = curr.get("total_alerts", 0)
    delta_alerts = curr_alerts - prev_alerts
    arrow = "↑" if delta_alerts > 0 else ("↓" if delta_alerts < 0 else "→")
    lines.append(f"- **Alertes**: {prev_alerts} → {curr_alerts} ({arrow} {abs(delta_alerts)})")
    if delta_alerts > 2:
        vigilance.append(f"Augmentation notable des alertes (+{delta_alerts}): reevaluer plan de soins.")
    elif curr_alerts == 0 and prev_alerts > 0:
        lines.append("  - *Amelioration: aucune alerte sur le dernier cycle.*")

    # -- Symptoms comparison --
    prev_symptoms = set(prev.get("symptoms", []))
    curr_symptoms = set(curr.get("symptoms", []))
    new_symptoms = curr_symptoms - prev_symptoms
    resolved_symptoms = prev_symptoms - curr_symptoms
    if new_symptoms:
        lines.append(f"- **Nouveaux symptomes**: {', '.join(new_symptoms)}")
        vigilance.append("Nouveaux symptomes apparus: evaluation clinique recommandee.")
    if resolved_symptoms:
        lines.append(f"- **Symptomes resolus**: {', '.join(resolved_symptoms)}")

    # -- Overall trend assessment --
    worsening = len(vigilance)
    if worsening == 0:
        trend = "**Tendance globale: STABLE** — Pas de degradation significative detectee."
    elif worsening <= 2:
        trend = "**Tendance globale: A SURVEILLER** — Quelques parametres necessitent attention."
    else:
        trend = "**Tendance globale: DEGRADATION** — Plusieurs indicateurs en hausse, reevaluation recommandee."

    # -- Build final markdown --
    md_parts = ["## Synthese Evolution Inter-Cycles\n"]
    md_parts.append(f"Comparaison: Session {prev.get('day_index', '?')} → Session {curr.get('day_index', '?')}\n")
    md_parts.extend(lines)
    md_parts.append(f"\n{trend}")

    if vigilance:
        md_parts.append("\n## Points de Vigilance\n")
        for v in vigilance:
            md_parts.append(f"- {v}")

    return {
        "analysis_markdown": "\n".join(md_parts),
        "sources": sources,
        "status": "threshold_analysis",
    }



def demo_graph_visualization(sentinel=None):
    """Show the workflow graph structure"""
    print_header("LANGGRAPH WORKFLOW STRUCTURE")
    
    if sentinel is None:
        sentinel = create_sentinel_graph(use_memory=False, trusted_internal_inputs=True)
    print(sentinel.get_graph_visualization())
    return sentinel


def demo_patient_graph():
    """Demonstrate the GraphRAG patient memory system"""
    print_header("GRAPHRAG PATIENT MEMORY")
    
    # Initialize GraphRAG
    graph_rag = PatientGraphRAG()
    
    # Use fixed patient profile (non-random) from JSON
    patient_data = load_or_create_patient_profile(patient_id="DEMO001")
    
    print(f"Creating patient: {patient_data['name']}")
    
    # Add patient to graph
    graph_rag.add_patient(
        patient_id=patient_data["patient_id"],
        name=patient_data["name"],
        age=patient_data["age"],
        conditions=[c["name"] for c in patient_data["conditions"]],
        medications=[m["name"] for m in patient_data["medications"]],
        allergies=[a["substance"] for a in patient_data["allergies"]],
        risk_factors=patient_data["risk_factors"],
        room=patient_data["room"]
    )
    
    # Add some historical events
    graph_rag.add_clinical_event(
        patient_id=patient_data["patient_id"],
        event_type="desaturation",
        description="Ã‰pisode de dÃ©saturation nocturne (SpO2 87%)",
        severity="high"
    )
    
    graph_rag.add_consultation(
        patient_id=patient_data["patient_id"],
        consultation_type="cardio",
        presenting_complaint="Douleur thoracique atypique",
        diagnosis="Douleur pariÃ©tale",
        provider="Dr. Example"
    )
    
    print_subheader("Patient Context Retrieved")
    context = graph_rag.get_patient_context(patient_data["patient_id"])
    print(f"  Name: {context.get('name')}")
    print(f"  Age: {context.get('age')} ans")
    print(f"  Room: {context.get('room')}")
    print(f"  Conditions: {[c['name'] for c in context.get('conditions', [])]}")
    print(f"  Medications: {[m['name'] for m in context.get('medications', [])]}")
    print(f"  Recent Events: {len(context.get('recent_events', []))}")
    
    print_subheader("Patient Summary for LLM")
    summary = graph_rag.get_patient_summary(patient_data["patient_id"])
    print(summary)
    
    print_subheader("Graph Statistics")
    stats = graph_rag.get_statistics()
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Edges: {stats['total_edges']}")
    print(f"  Patients: {stats['total_patients']}")
    
    return graph_rag, patient_data


def demo_medgemma_prompts():
    """Demonstrate MedGemma steering prompts"""
    print_header("MEDGEMMA STEERING PROMPTS")
    
    print("Available prompt types:")
    for prompt_info in MedGemmaPrompts.list_prompts():
        print(f"  - {prompt_info['name']}: {prompt_info['type']}")
    
    print_subheader("Night Surveillance Prompt")
    night_prompt = MedGemmaPrompts.get_prompt(PromptType.NIGHT_SURVEILLANCE)
    print(f"Name: {night_prompt.name}")
    print(f"Temperature: {night_prompt.temperature}")
    print(f"Max Tokens: {night_prompt.max_tokens}")
    print(f"Output Sections: {night_prompt.output_sections}")
    print("\nSystem Prompt Preview (first 500 chars):")
    print(night_prompt.system_prompt[:500] + "...")
    
    print_subheader("Day Consultation Prompt (Cardio Mode)")
    cardio_prompt = MedGemmaPrompts.get_prompt(PromptType.CARDIO_ANALYSIS)
    print(f"Name: {cardio_prompt.name}")
    print(f"Specialty context in system prompt:")
    # Extract specialty part
    lines = cardio_prompt.system_prompt.split('\n')
    for i, line in enumerate(lines):
        if "CARDIOLOGIE" in line:
            print('\n'.join(lines[i:i+5]))
            break


def demo_full_workflow(
    graph_rag: PatientGraphRAG,
    patient_data: dict,
    sentinel_graph=None,
    clinical_date: date | None = None,
    day_index: int = 1,
):
    """Run the complete workflow"""
    print_header("FULL WORKFLOW EXECUTION")
    
    # Generate synthetic data
    gen = SyntheticDataGenerator()
    night_data = gen.generate_night_scenario(
        patient_data,
        scenario_type="moderate",
        clinical_date=clinical_date,
    )
    consultation_data = gen.generate_consultation_scenario(
        patient_data,
        consultation_mode="cardio",
        clinical_date=clinical_date,
    )
    
    print(f"Patient: {patient_data['name']} (ID: {patient_data['patient_id']})")
    print(f"Room: {patient_data['room']}")
    if clinical_date:
        print(f"Clinical Date: {clinical_date.isoformat()} (Day {day_index})")
    print(f"Night Scenario: {night_data['scenario_type']}")
    print(f"  - Vitals readings: {len(night_data['vitals_input'])}")
    print(f"  - Audio events: {len(night_data['audio_input'])}")
    print(f"  - Vision events: {len(night_data['vision_input'])}")
    print(f"Consultation Mode: {consultation_data['consultation_mode']}")
    print(f"  - Symptoms: {consultation_data['symptoms_input']}")
    
    print_subheader("Executing LangGraph Workflow (Per-Node Guardrails)")
    
    # Reuse the sentinel graph (model already loaded) or create new
    sentinel = sentinel_graph if sentinel_graph else create_sentinel_graph(
        use_memory=False,
        trusted_internal_inputs=True,
    )
    
    # Show guardrails status
    guard_status = sentinel.get_guardrails_status()
    print(f"  Guardrails: {'ACTIVE' if guard_status['enabled'] else 'DISABLED'}")
    print(f"  Mode: {guard_status.get('mode', 'per_node')}")
    
    # Get patient context from GraphRAG
    patient_context = graph_rag.get_patient_context(patient_data["patient_id"])
    
    # Run the workflow â€” guardrails wrap every node
    print("Running: [G]â†’Nightâ†’[G] â†’ [G]â†’Rap1â†’[G] â†’ [G]â†’Dayâ†’[G] â†’ [G]â†’Rap2â†’[G]...")
    
    result = sentinel.run(
        patient_id=patient_data["patient_id"],
        patient_context=patient_context,
        vitals_input=night_data["vitals_input"],
        audio_input=night_data["audio_input"],
        vision_input=night_data["vision_input"],
        consultation_mode=consultation_data["consultation_mode"],
        symptoms_input=consultation_data["symptoms_input"],
        exam_input=consultation_data["exam_input"],
        day_vitals_input=consultation_data["day_vitals_input"],
        presenting_complaint=consultation_data.get("presenting_complaint", "")
    )
    
    # Show results
    blocked = result.get('guardrails_blocked', False)
    print(f"\n[OK] Workflow completed!")
    print(f"  Final Phase: {result.get('phase')}")
    print(f"  Guardrails Blocked: {blocked}")
    print(f"  Total Events Processed: {result.get('total_events_processed', 0)}")
    print(f"  Total Alerts: {result.get('total_alerts', 0)}")
    
    # Show per-node guard audit trail
    guard_log = result.get('guard_log', [])
    if guard_log:
        print(f"\n  --- Per-Node Guard Audit Trail ({len(guard_log)} checks) ---")
        for entry in guard_log:
            icon = {"passed": "âœ…", "blocked": "âŒ", "flagged": "âš ï¸", "filtered": "ðŸ”", "skipped": "â­ï¸"}.get(
                entry.get("status", ""), "?"
            )
            print(f"    {icon} {entry['node']}.{entry['check']}: {entry['status']}")
    
    # Attach synthetic data for report generation (vitals timeline for plots)
    result["_vitals_timeline"] = night_data["vitals_input"]
    result["_consultation_data"] = consultation_data
    session_file = save_session_json(
        patient_data=patient_data,
        night_data=night_data,
        consultation_data=consultation_data,
        workflow_result=result,
        clinical_date=night_data.get("clinical_date"),
        day_index=day_index,
    )
    result["_session_file"] = str(session_file)
    print(f"\n[DATA] Session JSON saved: {session_file}")
    
    return result


def demo_report_generation(
    result: dict,
    patient_data: dict,
    pdf_gen: PDFReportGenerator | None = None,
    night_template: NightReportTemplate | None = None,
    day_template: DayReportTemplate | None = None,
    medgemma_engine: "MedGemmaEngine | None" = None,
):
    """Demonstrate report generation with plots and structured parsing"""
    print_header("REPORT GENERATION")
    
    # If pipeline was blocked by guardrails, skip report generation
    if result.get("guardrails_blocked", False):
        print("  [GUARD] Pipeline was blocked by guardrails â€” no reports generated")
        print(f"  Blocked at: {result.get('input_guard_result', {}).get('blocked_at', 'unknown')}")
        return
    
    from src.reporting.clinical_plots import generate_night_report_plots
    
    # Night report
    print_subheader("Night Report (Rap1)")
    night_report = result.get("rap1_report", {})
    if night_report:
        print(f"  Title: {night_report.get('title')}")
        print(f"  Type: {night_report.get('report_type')}")
        print(f"  Sections: {len(night_report.get('sections', []))}")
        summary_text = night_report.get('summary', 'N/A') or 'N/A'
        print(f"  Summary: {summary_text[:100]}...")
    
    # Day report
    print_subheader("Day Report (Rap2)")
    day_report = result.get("rap2_report", {})
    if day_report:
        print(f"  Title: {day_report.get('title')}")
        print(f"  Type: {day_report.get('report_type')}")
        print(f"  Sections: {len(day_report.get('sections', []))}")
        summary_text = day_report.get('summary', 'N/A') or 'N/A'
        print(f"  Summary: {summary_text[:100]}...")

    # ---- Analyze evolution from last 2 sessions (night+day cycles) ----
    history_enrichment = analyze_recent_history_with_medgemma(
        patient_id=patient_data["patient_id"],
        medgemma_engine=medgemma_engine,
    )
    if history_enrichment.get("analysis_markdown"):
        print_subheader("Evolution Analysis (2 Sessions)")
        analysis_preview = history_enrichment["analysis_markdown"]
        print(analysis_preview[:600] + ("..." if len(analysis_preview) > 600 else ""))
    
    # ---- Compute real vitals summary from timeline data ----
    vitals_timeline = result.get("_vitals_timeline", [])
    night_data = result.get("night_data") or {}
    
    spo2_values = [v["spo2"] for v in vitals_timeline if "spo2" in v]
    hr_values = [v["heart_rate"] for v in vitals_timeline if "heart_rate" in v]
    temp_values = [v["temperature"] for v in vitals_timeline if "temperature" in v]
    
    vitals_summary = {}
    if spo2_values:
        vitals_summary["SpO2"] = {
            "min": min(spo2_values), "max": max(spo2_values),
            "avg": round(sum(spo2_values) / len(spo2_values), 1),
            "anomalies": sum(1 for v in spo2_values if v < 92)
        }
    if hr_values:
        vitals_summary["FC"] = {
            "min": min(hr_values), "max": max(hr_values),
            "avg": round(sum(hr_values) / len(hr_values), 1),
            "anomalies": sum(1 for v in hr_values if v > 100 or v < 60)
        }
    if temp_values:
        vitals_summary["T"] = {
            "min": min(temp_values), "max": max(temp_values),
            "avg": round(sum(temp_values) / len(temp_values), 1),
            "anomalies": sum(1 for v in temp_values if v > 37.5)
        }
    
    # ---- Generate clinical plots ----
    print_subheader("Generating Clinical Plots")
    events = night_data.get("events", [])
    plot_paths = generate_night_report_plots(
        vitals_timeline=vitals_timeline,
        events=events,
        output_dir="./data/reports/plots",
        patient_id=patient_data["patient_id"]
    )
    for name, path in plot_paths.items():
        print(f"  [PLOT] {name}: {path}")
    
    # ---- Generate PDF Reports ----
    print_subheader("Generating PDF Reports")
    
    pdf_gen = pdf_gen or PDFReportGenerator(output_dir="./data/reports")
    
    # Night report data (enriched with vitals timeline for plots)
    night_pdf_data = {
        "patient_id": patient_data["patient_id"],
        "patient_name": patient_data["name"],
        "room": patient_data["room"],
        "summary": night_report.get("summary", "Rapport de surveillance nocturne"),
        "events": events,
        "night_data": night_data,
        "vitals_timeline": vitals_timeline,  # For plot generation in PDF
        "vitals_summary": vitals_summary,
        "recommendations": [
            "Surveillance continue de la saturation (SpO2 < 92%)",
            "ECG de controle recommande - anomalies FC detectees",
            "Verifier hydratation et temperature",
            "Reevaluer les constantes a la prise de poste"
        ],
        "history_evolution_insights": history_enrichment.get("analysis_markdown", ""),
        "history_sources": history_enrichment.get("sources", []),
    }
    
    night_pdf_path = pdf_gen.generate_night_report(night_pdf_data)
    print(f"  Night Report PDF: {night_pdf_path}")
    
    # Day report data (enriched with diagnosis/reasoning from DayNode)
    day_data = result.get("day_data", {})
    consultation_data = result.get("_consultation_data", {})
    
    day_pdf_data = {
        "patient_id": patient_data["patient_id"],
        "patient_name": patient_data["name"],
        "day_data": day_data,
        "night_context": night_report.get("summary", ""),
        "illness_history": (
            "Patient suivi pour pathologie cardiaque connue. "
            "Consultation motivee par les symptomes apparus recemment. "
            "Nuit precedente marquee par des evenements cliniques significatifs."
        ),
        "diagnosis_reasoning": day_data.get("diagnosis_reasoning", ""),
        "tests_to_order": "ECG 12 derivations, Troponine, BNP, NFS, Ionogramme",
        "follow_up": "Reevaluation cardiologique a 48h, controle ECG a J3",
        "vigilance_points": (
            "Surveiller recidive des palpitations et syncope. "
            "Controle SpO2 toutes les 4h. "
            "Alerter si douleur thoracique ou dyspnee au repos."
        ),
        "orientation": "Hospitalisation en cardiologie pour bilan complet",
        "history_evolution_insights": history_enrichment.get("analysis_markdown", ""),
        "history_sources": history_enrichment.get("sources", []),
    }
    
    day_pdf_path = pdf_gen.generate_day_report(day_pdf_data)
    print(f"  Day Report PDF: {day_pdf_path}")
    
    # ---- Generate Markdown reports ----
    print_subheader("Generating Markdown Reports")
    
    md_path = Path("./data/reports")
    md_path.mkdir(parents=True, exist_ok=True)
    
    # Night markdown with plot references
    night_template = night_template or NightReportTemplate()
    night_md = night_template.render_markdown(night_pdf_data)
    
    # Replace placeholder with actual plot references
    if plot_paths:
        plots_md = "\n\n### Tableau de Bord des Constantes\n\n"
        if "vitals_dashboard" in plot_paths:
            plots_md += f"![Vitals Dashboard]({plot_paths['vitals_dashboard']})\n\n"
        if "spo2_trend" in plot_paths:
            plots_md += "### Tendance SpO2\n\n"
            plots_md += f"![SpO2 Trend]({plot_paths['spo2_trend']})\n\n"
        if "heart_rate_trend" in plot_paths:
            plots_md += "### Tendance Frequence Cardiaque\n\n"
            plots_md += f"![Heart Rate Trend]({plot_paths['heart_rate_trend']})\n\n"
        if "temperature_trend" in plot_paths:
            plots_md += "### Tendance Temperature\n\n"
            plots_md += f"![Temperature Trend]({plot_paths['temperature_trend']})\n\n"
        if "events_timeline" in plot_paths:
            plots_md += "### Chronologie des Evenements\n\n"
            plots_md += f"![Events Timeline]({plot_paths['events_timeline']})\n\n"
        if "severity_distribution" in plot_paths:
            plots_md += "### Distribution des Alertes\n\n"
            plots_md += f"![Severity Distribution]({plot_paths['severity_distribution']})\n\n"
        
        night_md = night_md.replace(
            "*[Les graphiques de tendances sont disponibles dans la version PDF compl\u00e8te]*",
            plots_md
        )
    
    night_md_path = md_path / f"rap1_night_{patient_data['patient_id']}.md"
    with open(night_md_path, "w", encoding="utf-8") as f:
        f.write(night_md)
    print(f"  Night Report Markdown: {night_md_path}")
    
    # Day markdown (enriched)
    day_template = day_template or DayReportTemplate()
    day_md = day_template.render_markdown(day_pdf_data)
    
    day_md_path = md_path / f"rap2_day_{patient_data['patient_id']}.md"
    with open(day_md_path, "w", encoding="utf-8") as f:
        f.write(day_md)
    print(f"  Day Report Markdown: {day_md_path}")


def demo_retrieval(graph_rag: PatientGraphRAG, patient_id: str):
    """Demonstrate the retrieval system"""
    print_header("CONTEXT RETRIEVAL")
    
    retriever = GraphRetriever(graph_rag)
    
    print_subheader("Night Surveillance Context")
    night_context = retriever.get_patient_context_for_night(patient_id)
    print(night_context[:500] + "..." if len(night_context) > 500 else night_context)
    
    print_subheader("Cardio Consultation Context")
    cardio_context = retriever.get_patient_context_for_consultation(patient_id, "cardio")
    print(cardio_context[:500] + "..." if len(cardio_context) > 500 else cardio_context)
    
    print_subheader("Semantic Search: 'desaturation respiratory'")
    result = retriever.retrieve(
        query="desaturation respiratory",
        patient_id=patient_id,
        max_results=5
    )
    print(f"Found {result.total_results} results in {result.retrieval_time_ms:.1f}ms")
    for node in result.nodes[:3]:
        print(f"  - [{node.get('type')}] {node.get('name')}: {node.get('description', '')[:50]}")


def main():
    """Run the complete demo"""
    parser = argparse.ArgumentParser(description="MedGemma Sentinel demo workflow")
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of sessions to run in one process (default: 1)",
    )
    args = parser.parse_args()
    days = max(1, args.days)

    print("\n" + "=" * 70)
    print("\n   MEDGEMMA SENTINEL - THE SCRIBE")
    print("   Memory & Reporting Engineer Demo")
    print("   Le Cerveau Clinique Offline pour une Sante Universelle")
    print("\n" + "=" * 70)
    
    try:
        # 1. Show graph visualization (also creates the sentinel graph)
        sentinel = demo_graph_visualization()
        
        # 2. Demo patient graph (GraphRAG)
        graph_rag, patient_data = demo_patient_graph()
        
        # 3. Demo MedGemma prompts
        demo_medgemma_prompts()
        
        # Reuse report objects across sessions to reduce repeated setup overhead.
        pdf_gen = PDFReportGenerator(output_dir="./data/reports")
        night_template = NightReportTemplate()
        day_template = DayReportTemplate()
        history_engine = None
        if MEDGEMMA_ENGINE_AVAILABLE:
            try:
                print_subheader("Initializing MedGemma History Analyzer")
                history_engine = MedGemmaEngine(temperature=0.2, max_tokens=512)
                status = history_engine.get_status()
                print(f"  [HISTORY] MedGemma mode: {status.get('mode')}")
            except Exception as history_err:
                print(f"  [WARN] MedGemma history analyzer unavailable: {history_err}")

        result = None
        base_clinical_date = date.today()
        for day_idx in range(1, days + 1):
            if days > 1:
                print_subheader(f"Session {day_idx}/{days}")
            clinical_day = base_clinical_date + timedelta(days=day_idx - 1)

            # 4. Run full workflow (reuse same sentinel graph)
            result = demo_full_workflow(
                graph_rag,
                patient_data,
                sentinel_graph=sentinel,
                clinical_date=clinical_day,
                day_index=day_idx,
            )

            # 5. Generate reports
            demo_report_generation(
                result,
                patient_data,
                pdf_gen=pdf_gen,
                night_template=night_template,
                day_template=day_template,
                medgemma_engine=history_engine,
            )

        # 6. Demo retrieval
        demo_retrieval(graph_rag, patient_data["patient_id"])
        
        print_header("DEMO COMPLETE")
        print("""
All components have been demonstrated:

1. [OK] LangGraph Orchestration (Per-Node Guardrails)
       [G]â†’Nightâ†’[G] â†’ [G]â†’Rap1â†’[G] â†’ [G]â†’Dayâ†’[G] â†’ [G]â†’Rap2â†’[G]
2. [OK] GraphRAG Patient Memory (LlamaIndex-based)
3. [OK] MedGemma Steering Prompts
4. [OK] PDF & Markdown Report Generation
5. [OK] Context Retrieval System

Generated files in ./data/reports/:
- rap1_night_DEMO001.pdf
- rap2_day_DEMO001.pdf
- rap1_night_DEMO001.md
- rap2_day_DEMO001.md

The system is ready for integration with MedGemma for full AI-powered
clinical decision support!
""")
        
    except Exception as e:
        print(f"\n[ERROR] Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


