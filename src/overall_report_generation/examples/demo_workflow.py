"""
MedGemma Sentinel - Complete Demo Workflow
Demonstrates the full Night -> Rap1 -> Day -> Rap2 workflow
with synthetic data and all components.
"""

import sys
import os
from datetime import datetime
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


def print_header(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---\n")


def demo_graph_visualization():
    """Show the workflow graph structure"""
    print_header("LANGGRAPH WORKFLOW STRUCTURE")
    
    sentinel = create_sentinel_graph(use_memory=False)
    print(sentinel.get_graph_visualization())


def demo_patient_graph():
    """Demonstrate the GraphRAG patient memory system"""
    print_header("GRAPHRAG PATIENT MEMORY")
    
    # Initialize GraphRAG
    graph_rag = PatientGraphRAG()
    
    # Generate synthetic patient
    gen = SyntheticDataGenerator(seed=42)
    patient_data = gen.generate_patient(patient_id="DEMO001")
    
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
        description="Épisode de désaturation nocturne (SpO2 87%)",
        severity="high"
    )
    
    graph_rag.add_consultation(
        patient_id=patient_data["patient_id"],
        consultation_type="cardio",
        presenting_complaint="Douleur thoracique atypique",
        diagnosis="Douleur pariétale",
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


def demo_full_workflow(graph_rag: PatientGraphRAG, patient_data: dict):
    """Run the complete workflow"""
    print_header("FULL WORKFLOW EXECUTION")
    
    # Generate synthetic data
    gen = SyntheticDataGenerator(seed=42)
    night_data = gen.generate_night_scenario(patient_data, scenario_type="moderate")
    consultation_data = gen.generate_consultation_scenario(patient_data, consultation_mode="cardio")
    
    print(f"Patient: {patient_data['name']} (ID: {patient_data['patient_id']})")
    print(f"Room: {patient_data['room']}")
    print(f"Night Scenario: {night_data['scenario_type']}")
    print(f"  - Vitals readings: {len(night_data['vitals_input'])}")
    print(f"  - Audio events: {len(night_data['audio_input'])}")
    print(f"  - Vision events: {len(night_data['vision_input'])}")
    print(f"Consultation Mode: {consultation_data['consultation_mode']}")
    print(f"  - Symptoms: {consultation_data['symptoms_input']}")
    
    print_subheader("Executing LangGraph Workflow")
    
    # Create and run the graph
    sentinel = create_sentinel_graph(use_memory=False)
    
    # Get patient context from GraphRAG
    patient_context = graph_rag.get_patient_context(patient_data["patient_id"])
    
    # Run the workflow
    print("Running: Night -> Rap1 -> Day -> Rap2...")
    
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
    
    print("\n[OK] Workflow completed!")
    print(f"  Final Phase: {result.get('phase')}")
    print(f"  Total Events Processed: {result.get('total_events_processed', 0)}")
    print(f"  Total Alerts: {result.get('total_alerts', 0)}")
    
    # Attach synthetic data for report generation (vitals timeline for plots)
    result["_vitals_timeline"] = night_data["vitals_input"]
    result["_consultation_data"] = consultation_data
    
    return result


def demo_report_generation(result: dict, patient_data: dict):
    """Demonstrate report generation with plots and structured parsing"""
    print_header("REPORT GENERATION")
    
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
    
    # ---- Compute real vitals summary from timeline data ----
    vitals_timeline = result.get("_vitals_timeline", [])
    night_data = result.get("night_data", {})
    
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
    
    pdf_gen = PDFReportGenerator(output_dir="./data/reports")
    
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
        ]
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
        "orientation": "Hospitalisation en cardiologie pour bilan complet"
    }
    
    day_pdf_path = pdf_gen.generate_day_report(day_pdf_data)
    print(f"  Day Report PDF: {day_pdf_path}")
    
    # ---- Generate Markdown reports ----
    print_subheader("Generating Markdown Reports")
    
    md_path = Path("./data/reports")
    md_path.mkdir(parents=True, exist_ok=True)
    
    # Night markdown with plot references
    night_template = NightReportTemplate()
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
    day_template = DayReportTemplate()
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
    print("\n" + "=" * 70)
    print("\n   MEDGEMMA SENTINEL - THE SCRIBE")
    print("   Memory & Reporting Engineer Demo")
    print("   Le Cerveau Clinique Offline pour une Sante Universelle")
    print("\n" + "=" * 70)
    
    try:
        # 1. Show graph visualization
        demo_graph_visualization()
        
        # 2. Demo patient graph (GraphRAG)
        graph_rag, patient_data = demo_patient_graph()
        
        # 3. Demo MedGemma prompts
        demo_medgemma_prompts()
        
        # 4. Run full workflow
        result = demo_full_workflow(graph_rag, patient_data)
        
        # 5. Generate reports
        demo_report_generation(result, patient_data)
        
        # 6. Demo retrieval
        demo_retrieval(graph_rag, patient_data["patient_id"])
        
        print_header("DEMO COMPLETE")
        print("""
All components have been demonstrated:

1. [OK] LangGraph Orchestration (Night -> Rap1 -> Day -> Rap2)
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
