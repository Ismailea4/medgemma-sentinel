# Overall Report Generation System 🏥📊

## Overview

The **Overall Report Generation** system is a comprehensive clinical intelligence platform that bridges night-to-day patient surveillance with AI-powered differential diagnosis. It combines autonomous night monitoring (vigilance), intelligent memory management (GraphRAG), and day-mode clinical decision support into a unified workflow.

## 🎯 System Purpose

This module provides the complete infrastructure for:
- **Night Mode:** Autonomous patient surveillance with multimodal monitoring (vitals, audio)
- **Day Mode:** Doctor-assisted clinical assessment with AI-powered differential diagnosis
- **Memory System:** Persistent GraphRAG-based patient records linking night events to day findings
- **Report Generation:** Professional PDF reporting for clinical handovers and differential diagnosis

---

## 📁 Module Architecture

### **Core Modules**

#### 1. **orchestration/** - LangGraph State Machine
Manages the workflow coordination between Night and Day modes.

**Files:**
- `graph.py` - Main LangGraph workflow definition
- `nodes.py` - Workflow nodes (event detection, analysis, reporting)
- `state.py` - Shared state management across workflow

**Key Concepts:**
- Reflex layer: Fast rule-based threshold triggering (HR > 130, SpO2 < 88, Audio > 80)
- Cognitive layer: MedGemma AI reasoning with activation steering
- Memory integration: Event persistence in GraphRAG

---

#### 2. **memory/** - GraphRAG Patient Records
Persistent knowledge graph for longitudinal patient history.

**Files:**
- `graph_store.py` - Neo4j graph database interface
- `patient_graph.py` - Patient entity and relationship models
- `retriever.py` - Query interface for event lookup
- `storage.py` - File-based local storage (for edge deployment)

**Key Features:**
- Stores patient vitals snapshots with timestamps
- Links night events to morning assessments  
- Enables longitudinal evolution analysis
- Supports both Neo4j (cloud) and local file storage

**Example:**
```python
from src.memory.storage import LocalStorage

storage = LocalStorage(base_path="data/patients")
events = storage.get_night_events("PATIENT-001")
# Returns: List of critical events from last 12 hours
```

---

#### 3. **reporting/** - Clinical Document Generation
Professional HTML/PDF reports with medical formatting.

**Files:**
- `pdf_generator.py` - ReportLab-based PDF rendering
- `clinical_plots.py` - Vitals trends visualization
- `medgemma_engine.py` - LLM integration for AI analysis
- `prompts.py` - Clinical reasoning prompts
- `steering.py` - Activation steering for model personality
- `templates.py` - HTML email templates

**Report Types:**
1. **Document A - SBAR Critical Incident Report** (triggered by night critical alerts)
2. **Document B - Shift Handover Summary** (generated at night-to-day transition)
3. **Document C - RAP2 Differential Diagnosis** (generated in Day Mode after doctor input)

---

#### 4. **guardrails/** - NeMo Guardrails Safety System
Clinical safety validation for AI outputs.

---

#### 5. **models/** - Pydantic Data Models
Type-safe data structures for clinical entities (Patient, Vitals, Events).

---

#### 6. **Root App: app_overall_report_generation.py**
Streamlit dashboard with Night/Day mode interface (moved from ui/app.py).

**Night Mode Features:**
- Real-time vitals monitoring with alert escalation
- NEWS2 score calculation and trending
- Critical threshold detection (HR > 130, SpO2 < 88, Audio > 80)
- Event logging to GraphRAG memory
- SBAR report generation and PDF download

**Day Mode Features:**
- Night surveillance timeline (left) + Doctor interface (right)
- View all night critical events with timestamps
- Doctor clinical assessment input
- Medical specialty selection
- RAP2 differential diagnosis generation
- PDF/TXT report download

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama (for MedGemma-4b-it) OR local GGUF model
- Streamlit environment

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Install system dependencies (for PDF generation):**
   
   **macOS:**
   ```bash
   brew install cairo pango gdk-pixbuf libffi
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install python3-dev libcairo2-dev libpango1.0-dev libpangoft2-1.0-0
   ```
   
   **Windows:**
   Install [GTK+ Runtime](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer) if needed

3. **Configure MedGemma model:**
   
   **Option A - Ollama:**
   ```bash
   ollama pull amsaravi/medgemma-4b-it:q6
   ```
   
   **Option B - Local GGUF:**
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli download Ismailea04/medgemma-night-sentinel --local-dir ./models
   ```

### Running the App

```bash
streamlit run app_overall_report_generation.py
```

Opens at `http://localhost:8501`

---

## 📊 Data Flow

**Night → Day Workflow:**
Sensor Data → Reflex Layer → Cognitive Layer (MedGemma) → GraphRAG Storage → PDF Reports

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🎬 Demo Workflow

1. Click "🎬 Run Scene 1 (The Night Shift Demo)" in Night Mode
2. Watch vitals escalate from stable → warning → critical
3. Download SBAR PDF
4. Go to Day Mode and click Refresh Events
5. Enter clinical findings and generate RAP2 differential report

---

## 📚 Tech Stack

- **LLM:** MedGemma 2 (4B, Q4_K_M quantization)
- **Orchestration:** LangGraph >= 1.0.0
- **Memory:** LlamaIndex GraphRAG
- **PDF:** ReportLab >= 4.4.0
- **Web:** Streamlit >= 1.31.0
- **Safety:** NeMo Guardrails >= 0.10.0
- **Inference:** llama-cpp-python >= 0.3.0 or Ollama

---

**Last Updated:** February 24, 2026  
**Version:** 1.0 (Kaggle MedGemma Impact Challenge)

## 👥 Team

- **Infrastructure:** Hamza — Model quantization and API deployment.
- **Steering:** Ismail — Activation engineering and vector injection.
- **Audio:** Youssra — Event detection and speech-to-text.
- **Vision & UI:** Saad/Othman — Fall detection and Streamlit dashboard.
- **Memory & Scribe:** Saad/Othman — LangGraph orchestration and PDF reporting.

---

## 📋 Deployment

Designed for the **Kaggle MedGemma Impact Challenge**. Deadline: February 24, 2026.
