# Reporting Model - Full Clinical Orchestration System 🏥

Comprehensive module for end-to-end clinical decision support with LangGraph orchestration, GraphRAG memory, automated reporting, and safety guardrails.

## 📋 Overview

The Reporting Model is the central nervous system of MedGemma Sentinel, coordinating:
1. **State Management:** LangGraph-based hierarchical state machine
2. **Temporal Processing:** Convert real-time vitals into events and observations
3. **Memory & Reasoning:** GraphRAG for patient longitudinal records and clinical evolution
4. **Report Generation:** Automated PDF and markdown clinical reports
5. **Safety Guardrails:** Input validation and output filtering using NeMo Guardrails

## 📁 Core Submodules

### `orchestration/` - Clinical Workflow Engine
LangGraph state machine for coordinating Reflex and Cognitive layers.

**Key Files:**
- `graph.py` - Define the main LangGraph workflow
- `nodes.py` - Individual processing nodes (vitals processing, inference, reporting)
- `state.py` - Define state schema (patient data, vitals, observations, reports)

**State Machine Architecture:**
```
REFLEX LAYER (Real-time)
├─ vitals_processor_node() - Parse vitals, detect anomalies
├─ audio_analyzer_node() - Process room audio (YamNet)
└─ event_detector_node() - Trigger alerts

COGNITIVE LAYER (Reasoning)
├─ llm_inference_node() - Call MedGemma with steered prompt
├─ clinical_assessment_node() - Extract structured findings
└─ recommendation_node() - Generate interventions

REPORTING LAYER
├─ pdf_generator_node() - Create PDF reports
├─ memory_update_node() - Store in GraphRAG
└─ escalation_node() - Determine alert severity
```

**Key Functions:**
- `build_graph()` - Construct LangGraph workflow
- `run_workflow(patient_data, vitals)` - Execute clinical pipeline
- `get_state(patient_id)` - Retrieve current patient state

---

### `memory/` - GraphRAG Patient Memory
Persistent knowledge graph for longitudinal patient records.

**Key Files:**
- `patient_graph.py` - Patient entity and relationship management
- `graph_store.py` - Neo4j graph database interface (or in-memory for edge)
- `retriever.py` - Query patient history and clinical evolution

**Key Functions:**
- `add_patient(patient_id, demographics)` - Register patient
- `add_visit(patient_id, date, findings)` - Log clinical encounter
- `get_patient_timeline(patient_id, days=30)` - Retrieve patient history
- `query_evolution(patient_id, condition)` - Ask about condition evolution
- `add_observation(patient_id, observation)` - Log vital signs or events

**Graph Schema:**
```
Patient --has--> Visit --contains--> Vitals
         --has--> Condition --evolves-in--> Timeline
         --has--> Medication
```

---

### `reporting/` - Clinical Report Generation
Automated PDF and markdown report creation with clinical insights.

**Key Files:**
- `pdf_generator.py` - Generate structured PDF reports
- `medgemma_engine.py` - LLM-based clinical summary generation
- `clinical_plots.py` - Vital sign visualizations and trend charts
- `prompts.py` - Report generation prompts
- `templates.py` - HTML/PDF templates for reports

**Key Functions:**
- `generate_report(patient, vitals, analysis)` - Create clinical PDF
- `generate_summary(observations)` - LLM-based clinical summary
- `plot_vitals_timeline(vitals_history)` - Generate trend visualization
- `format_findings(structured_data)` - Convert structured data to clinical text

**Report Sections:**
- Patient demographics and baseline
- Chief complaint and history of present illness
- Vital signs and trends (with plots)
- Physical examination findings
- Assessment and clinical impression
- Recommendations and management plan
- Risk stratification
- Follow-up instructions

---

### `guardrails/` - Safety Gate
NeMo Guardrails for input/output validation.

**Key Files:**
- `sentinel_guard.py` - Custom guard for medical context
- `input_guardrails.co` - Input constraint rules (prevent irrelevant inputs)
- `output_guardrails.co` - Output constraint rules (ensure clinical appropriateness)
- `config.yml` - Guardrail configuration

**Key Functions:**
- `check_input_safety(user_input)` - Validate clinical relevance
- `check_output_safety(llm_output)` - Ensure response appropriateness
- `apply_output_filter(response)` - Enforce safety constraints

**Guardrail Examples:**
- Reject non-medical queries ("Tell me a joke")
- Flag uncertain diagnoses ("This might be...")
- Require escalation for emergencies
- Prevent over-diagnosis without evidence

---

### `models/` - Pydantic Data Models
Structured data schemas for type safety.

**Key Classes:**
- `Patient` - Patient demographics and baseline vitals
- `Vitals` - Single vital signs measurement (HR, SpO2, RESP, BP)
- `VitalEvent` - Time-stamped vital with metadata
- `Observation` - Clinical finding or assessment
- `Event` - Detected event (alert, anomaly, intervention)

**Example:**
```python
from models import Patient, VitalEvent

patient = Patient(
    patient_id="402",
    age_years=65,
    gender="M",
    weight_kg=80,
    baseline_hr=70
)

event = VitalEvent(
    timestamp=datetime.now(),
    hr=125,
    spo2=88,
    severity="high"
)
```

---

## 🔄 Data Flow

```
Raw Vitals Input
       │
       ▼
Orchestration/Vitals Processor
├─ Parse vitals
├─ Detect anomalies
└─ Flag events
       │
       ▼
GraphRAG Memory
├─ Store vitals
├─ Link to patient timeline
└─ Retrieve history
       │
       ▼
LLM Inference (Steered)
├─ Build context-aware prompt
├─ Call MedGemma
└─ Extract clinical findings
       │
       ▼
Guardrails Check
├─ Validate output
├─ Filter unsafe content
└─ Escalate if needed
       │
       ▼
Report Generation
├─ Create PDF/markdown
├─ Include plots & trends
└─ Output recommendations
       │
       ▼
Clinical Output
```

---

## 🧠 State Machine Example

```
START
  │
  ├─→ [REFLEX] Parse Vitals
  │      ├─→ HR: 125 bpm (tachycardia)
  │      ├─→ SpO2: 88% (hypoxemia)
  │      └─→ Event: "Cardiac Distress"
  │
  ├─→ [COGNITIVE] LLM Assessment
  │      ├─→ Prompt: "65yo baseline HR 70, current 125 with SpO2 88..."
  │      ├─→ Response: "Possible atrial fibrillation or sepsis"
  │      └─→ Recommendation: "Immediate ECG + bloodwork"
  │
  ├─→ [MEMORY] Update GraphRAG
  │      ├─→ Add vital event to patient timeline
  │      ├─→ Link to tachycardia condition
  │      └─→ Flag escalation event
  │
  ├─→ [REPORTING] Generate Report
  │      ├─→ Create PDF with findings
  │      ├─→ Plot vital trends
  │      └─→ Output management plan
  │
  └─→ [ESCALATION] Check Severity
         ├─→ Risk Level: HIGH
         ├─→ Action: Alert night nurse
         └─→ Follow-up: Check in 5 minutes
```

---

## 🎯 Quick Start

### Basic Usage
```python
from reporting_model.orchestration.graph import build_graph
from reporting_model.models import Patient, VitalEvent
from datetime import datetime

# Build workflow
graph = build_graph()

# Create patient
patient = Patient(
    patient_id="402",
    age_years=65,
    baseline_hr=70
)

# Create vital event
event = VitalEvent(
    timestamp=datetime.now(),
    hr=125,
    spo2=88,
    severity="high"
)

# Run pipeline
result = graph.invoke({
    "patient": patient,
    "vitals": event
})

print(result["report"])  # PDF report
print(result["recommendations"])  # Clinical recommendations
```

---

### Advanced: Custom Nodes
```python
from reporting_model.orchestration.nodes import Node

class CustomAnalysisNode(Node):
    def invoke(self, state):
        # Custom analysis logic
        state["custom_analysis"] = run_custom_algo(state["vitals"])
        return state

# Add to graph
graph.add_node("custom_analysis", CustomAnalysisNode())
```

---

## 📊 Key Features

### Hierarchical Processing
- **Reflex Layer:** Sub-100ms edge processing (vitals parsing, anomaly detection)
- **Cognitive Layer:** 200-500ms reasoning (LLM inference, clinical assessment)
- **Reporting Layer:** Batch processing for PDF generation

### GraphRAG Integration
- Persistent patient knowledge graph
- Automatic relationship discovery
- Longitudinal trend analysis
- Clinical context injection into prompts

### Automated Reporting
- Structured PDF generation with clinical layout
- Vital sign trend visualization
- Evidence-based recommendations
- Risk stratification and escalation criteria

### Safety Guardrails
- Input validation (check medical relevance)
- Output filtering (prevent harmful suggestions)
- Confidence scoring (flag uncertain assessments)
- Escalation protocols (emergency detection)

---

## 📦 Dependencies

- `langgraph>=1.0.0` - Workflow orchestration
- `llama-index-core>=0.14.0` - GraphRAG integration
- `reportlab>=4.4.0` - PDF generation
- `nemoguardrails>=0.10.0` - Safety guardrails
- `pydantic>=2.8.0` - Data validation

---

## 🧪 Testing

```bash
# Test orchestration
python -m pytest tests/test_orchestration.py -v

# Test memory system
python -m pytest tests/test_memory.py -v

# Test reporting
python -m pytest tests/test_reporting.py -v
```

---

## 🚀 Next Steps

- Fine-tune LangGraph for your clinical protocols
- Customize guardrails for your facility
- Integrate with electronic health record (EHR) systems
- Add facility-specific vital sign thresholds

---

**Last Updated:** February 24, 2026  
**Version:** 1.0

* **Infrastructure:** Hamza — Model quantization and API deployment.
* **Steering:** Ismail — Activation engineering and vector injection.
* **Audio:** Youssra — Event detection and speech-to-text.
* **Vision & UI:** Saad/Othman — Fall detection and Streamlit dashboard.
* **Memory & Scribe:** Saad/Othman — LangGraph orchestration and PDF reporting.
## 🚀 Deployment

Designed for the **Kaggle MedGemma Impact Challenge**. Deadline: February 24, 2026.
