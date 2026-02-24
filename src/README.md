# MedGemma Sentinel - Source Code Architecture 🏗️

This directory contains the core modules of the MedGemma Sentinel system. Each submodule is responsible for a specific aspect of the clinical decision support pipeline.

## 📚 Module Overview

### 1. **mcp_architecture/** - Model Context Protocol & Orchestration
Real-time medical decision support through MCP-based ReAct loops with tool orchestration.

- **Core Components:**
  - `medical_mcp_server.py` - Registers clinical tools (cardiology analysis, patient history, audio analysis)
  - `medical_mcp_client_2.py` - ReAct loop client for autonomous clinical reasoning
  - `cardiology_sentinel.py` - Cardiology-specific inference and prompt steering

- **Key Features:**
  - Tool-based clinical reasoning (ReAct pattern)
  - Activation steering for model personality switching
  - Real-time vitals stream processing with windowing

👉 **Details:** See [mcp_architecture/README.md](mcp_architecture/README.md)

---

### 2. **night_cardiology_sentinel/** - Vitals Monitoring & Windowing
Standalone cardiac vitals processing with local inference pipeline.

- **Core Components:**
  - `data_parser.py` - Parse vitals from text files (multiple formats)
  - `inference.py` - Local inference using llama-cpp-python or Ollama

- **Key Features:**
  - Support for multiple vitals formats (HR, SpO2, RESP, PULSE)
  - Windowing strategies (15-min temporal, 10-row chunk)
  - Baseline comparison and anomaly detection

👉 **Details:** See [night_cardiology_sentinel/README.md](night_cardiology_sentinel/README.md)

---

### 3. **reporting_model/** - Full Clinical Pipeline
End-to-end system for comprehensive clinical analysis, memory management, and PDF report generation.

- **Core Components:**
  - `orchestration/` - LangGraph state machine and workflow nodes
  - `memory/` - GraphRAG integration for patient longitudinal records
  - `reporting/` - PDF generation and clinical visualizations
  - `guardrails/` - NeMo Guardrails for safety checks
  - `models/` - Pydantic data models for patients, vitals, events

- **Key Features:**
  - Hierarchical state machine (Reflex + Cognitive layers)
  - GraphRAG-based memory for longitudinal analysis
  - Automated PDF reporting with clinical insights
  - Input/output guardrails for safety

👉 **Details:** See [reporting_model/README.md](reporting_model/README.md)

---

### 4. **longitudinal_src/** - Multi-Report Clinical Evolution Analysis
Analyzes multiple patient reports over time to detect trends and generate evolution insights.

- **Core Components:**
  - `longitudinal_analysis.py` - PDF extraction, evolution tracking, visualization

- **Key Features:**
  - Extract structured medical data from PDF reports
  - Compare and track symptom/diagnosis evolution
  - Generate clinical recommendations and risk assessments
  - Visualize trends over time

👉 **Details:** See [longitudinal_src/README.md](longitudinal_src/README.md)

---

### 5. **overall_report_generation/** - Comprehensive Night/Day Surveillance
Full-stack clinical surveillance with dual-mode dashboard, intelligent report generation, and clinical decision support.

- **Core Components:**
  - `orchestration/` - LangGraph state machines for Night/Day workflows
  - `memory/` - Patient store, event management, and longitudinal memory
  - `reporting/` - PDF generators (SBAR, Shift Handover, RAP2 Differential)
  - `guardrails/` - Clinical safety validation and output steering
  - `models/` - Pydantic models for Patient, Vitals, Events, Reports
  - `ui/app.py` - Streamlit dashboard with Night/Day modes (moved to root as `app_overall_report_generation.py`)

- **Key Features:**
  - **Night Mode:** Autonomous vitals monitoring with auto-escalation and SBAR report generation
  - **Day Mode:** Doctor-assisted clinical assessment with RAP2 differential diagnosis
  - **Clinical Scoring:** NEWS2 calculation, reflex escalation rules, threshold alerts
  - **Report Generation:** Professional PDFs with clinical plots and structured narratives
  - **Smart Memory:** Event persistence and patient history tracking

👉 **Details:** See [overall_report_generation/README.md](overall_report_generation/README.md)

---

## 🔄 System Integration Flow

```
┌─────────────────────────────────────────────────────────┐
│                   User Applications                      │
├─────────────────────────────────────────────────────────┤
│ app_night_cardiology_sentinal.py                        │
│ app_mcp_cardiology.py                                   │
│ app_longitudinal_analysis.py                            │
│ app_overall_report_generation.py (NEW)                 │
└──────────────┬──────────────────────────────────────────┘
               │
       ┌───────┴───────────────────────────┬──────────────┐
       │                                   │              │
   ┌───▼──────────┐    ┌──────────────┐  │  ┌───────────▼───┐
   │   Night      │    │     MCP      │  │  │  Longitudinal │
   │ Cardiology   │    │ Architecture │  │  │     Source    │
   │  Sentinel    │    │              │  │  │               │
   └──────────────┘    └──────────────┘  │  └───────────────┘
        │                    │           │         │
        └────────────┬───────┴───────────┼─────────┘
                     │                   │
              ┌──────▼──────────────────▼─────┐
              │ Overall Report Generation    │
              │ ┌──────────┐  ┌────────────┐ │
              │ │Night Mode│  │ Day Mode   │ │
              │ │  (Auto)  │  │ (Manual)   │ │
              │ └──────────┘  └────────────┘ │
              └──────────┬────────────────────┘
                         │
   ┌────────────────────▼───────────────────────┐
   │    Reporting Model System (Shared)         │
   ├──────────────────────────────────────────┤
   │ ┌──────────────┐  ┌──────────┐  ┌─────────────┐ │
   │ │Orchestration │  │ Memory   │  │  Reporting  │ │
   │ │(LangGraph)   │  │(GraphRAG)│  │(PDF/Report) │ │
   │ └──────────────┘  └──────────┘  └─────────────┘ │
   └──────────────────────────────────────────────────┘
```

## 📦 Dependency Structure

| Module | Dependencies | Primary Use |
|--------|--------------|-------------|
| **night_cardiology_sentinel** | llama-cpp-python, ollama | Real-time vitals monitoring |
| **mcp_architecture** | fastmcp, ollama, python-dotenv | Interactive ReAct loops |
| **reporting_model** | langgraph, llama-index, reportlab | Full clinical pipeline |
| **longitudinal_src** | pdfplumber, plotly, reportlab | Report analysis & evolution tracking |
| **overall_report_generation** | langgraph, llama-index, reportlab, weasyprint | Night/Day surveillance & reporting |

## 🚀 Getting Started

### For Vitals Monitoring Only
```bash
pip install -r night_cardiology_sentinel/requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
streamlit run app_night_cardiology_sentinal.py
```

Add your Hugging Face token in `night_cardiology_sentinel/.env`:
```
HF_TOKEN=your_hugging_face_token
```

Optional local model download:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Ismailea04/medgemma-night-sentinel --local-dir ./models
```

### For MCP-Based Clinical Reasoning
```bash
pip install -r mcp_architecture/requirements.txt
ollama pull amsaravi/medgemma-4b-it:q6
streamlit run app_mcp_cardiology.py
```

Install Ollama from https://ollama.ai before pulling the model.

Add your Hugging Face token in `mcp_architecture/.env`:
```
HF_TOKEN=your_hugging_face_token
```

### For Longitudinal Report Analysis
```bash
pip install -r longitudinal_src/requirements.txt
streamlit run app_longitudinal_analysis.py
```

Add your Hugging Face token in `longitudinal_src/.env`:
```
HF_TOKEN=your_hugging_face_token
```

Download the local model:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download hmzBen/medgemma-1.5-medical-q4km --local-dir ./models
```

### For Full System
```bash
pip install -r ../requirements.txt  # Install consolidated deps
streamlit run app_mcp_cardiology.py
```

### For Overall Report Generation (Night/Day Surveillance)

Comprehensive Night-to-Day workflow with unified AI-powered clinical intelligence, SBAR reporting, and RAP2 differential diagnosis:

**Prerequisites:**
- Python >= 3.10
- System dependencies:
  - **macOS:** `cairo`, `pango`, `gdk-pixbuf` (install via `brew install cairo pango gdk-pixbuf`)
  - **Linux:** `libcairo2-dev`, `libpango`, `libpango-1.0-0`, `libpango-gobject-0` (install via `apt-get install`)
  - **Windows:** GTK+ Runtime (if using weasyprint)

**Installation:**
```bash
# Install module dependencies
pip install -r overall_report_generation/requirements.txt

# Install llama-cpp-python (choose one):
# For CPU:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# For CUDA (NVIDIA GPU):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For Metal (Apple Silicon):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

**Model Setup (choose one):**

Option 1 - Ollama (Recommended):
```bash
ollama pull amsaravi/medgemma-4b-it:q6
# App will auto-connect to Ollama at localhost:11434
```

Option 2 - Local GGUF Model:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Ismailea04/medgemma-night-sentinel --local-dir ./models
# Configure in app UI: models/medgemma-night-sentinel-Q4_K_M.gguf
```

**Environment Configuration (Optional):**

If using Hugging Face model loader, create `.env` in the workspace root:
```
HF_TOKEN=your_hugging_face_token
OLLAMA_BASE_URL=http://localhost:11434
```

**Running the App:**

```bash
streamlit run app_overall_report_generation.py
```

Then in the Streamlit UI:
1. **Create/Select Patient:** Enter demographics or select existing patient profile
2. **Night Mode:** 
   - Input vital signs (HR, SpO2, respiratory rate)
   - System auto-generates SBAR critical incident reports
   - Automatic NEWS2 scoring and threshold escalations
   - Save event logs to local persistent storage
3. **Day Mode:**
   - Review 24-hour event timeline
   - Doctor-assisted clinical assessment
   - Generate RAP2 differential diagnosis PDFs
   - Export shift handover summaries

**Architecture:**

See [overall_report_generation/README.md](overall_report_generation/README.md) for:
- **6 Core Modules:** orchestration, memory, reporting, guardrails, models, ui
- **Data Flow Diagrams:** Night Mode and Day Mode workflows
- **Demo Workflow:** 4-step end-to-end example
- **Tech Stack:** Complete dependency documentation

## For Full System
```bash
pip install -r ../requirements.txt  # Install consolidated deps
streamlit run app_mcp_cardiology.py
```

## 📖 Module Details

- **[mcp_architecture/](mcp_architecture/)** - Real-time ReAct-based clinical reasoning
- **[night_cardiology_sentinel/](night_cardiology_sentinel/)** - Lightweight vitals processing
- **[reporting_model/](reporting_model/)** - Full orchestration & memory system
- **[longitudinal_src/](longitudinal_src/)** - Multi-report evolution analysis
- **[overall_report_generation/](overall_report_generation/)** - Comprehensive Night/Day clinical surveillance

---

**Last Updated:** February 24, 2026  
**Version:** 1.0
