# MedGemma Sentinel - Complete System Documentation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [The Model](#3-the-model)
4. [Project Structure](#4-project-structure)
5. [Dependencies & Versions](#5-dependencies--versions)
6. [Installation Guide (Any Machine)](#6-installation-guide-any-machine)
7. [How to Run the System](#7-how-to-run-the-system)
8. [Built Functionalities](#8-built-functionalities)
9. [MedGemma Engine (Inference Modes)](#9-medgemma-engine-inference-modes)
10. [Steering Prompts System](#10-steering-prompts-system)
11. [Report Generation Pipeline](#11-report-generation-pipeline)
12. [LangGraph Orchestration](#12-langgraph-orchestration)
13. [GraphRAG Memory System](#13-graphrag-memory-system)
14. [PDF Report Generator](#14-pdf-report-generator)
15. [Testing](#15-testing)
16. [Troubleshooting](#16-troubleshooting)
17. [API Reference](#17-api-reference)

---

## 1. System Overview

**MedGemma Sentinel** is an autonomous medical intelligence infrastructure that fuses night surveillance, specialized consultation assistance, and multimodal analysis into a single AI engine. It acts as an embedded clinical brain capable of watching, assisting, understanding, and documenting human health -- without internet, without cloud, and at low cost.

### Core Capabilities

- **Night Surveillance (RAP1):** Continuous monitoring of patients via SpO2, heart rate, temperature, audio (breathing, cough), and IR vision (posture, falls). Generates structured night reports.
- **Day Consultation (RAP2):** Assists generalist staff as a virtual specialist (cardiology, dermatology, ophthalmology). Generates consultation reports with differential diagnoses.
- **Offline-First:** Runs entirely on local hardware using a quantized MedGemma model (2.4 GB RAM). No internet required after setup.
- **Structured Clinical Documentation:** AI-generated Markdown and PDF reports following medical standards.

---

## 2. Architecture

```
+-------------------------------------------------------------------+
|                     MedGemma Sentinel System                       |
+-------------------------------------------------------------------+
|                                                                     |
|  +------------------+    +------------------+    +--------------+   |
|  |   LangGraph      |    |   GraphRAG       |    |  MedGemma    |   |
|  |   Orchestration   |--->|   Memory         |--->|  Engine      |   |
|  |   (State Machine) |    |   (Patient Graph)|    |  (LLM)      |   |
|  +------------------+    +------------------+    +--------------+   |
|          |                        |                      |          |
|          v                        v                      v          |
|  +------------------+    +------------------+    +--------------+   |
|  |   Night Node     |    |   Patient Graph  |    |  Steering    |   |
|  |   Day Node       |    |   Retriever      |    |  Prompts     |   |
|  |   RAP1 Node      |    |   Graph Store    |    |  (Night/Day) |   |
|  |   RAP2 Node      |    |                  |    +--------------+   |
|  +------------------+    +------------------+           |          |
|                                                          v          |
|                                                  +--------------+   |
|                                                  |  PDF Report  |   |
|                                                  |  Generator   |   |
|                                                  |  (ReportLab) |   |
|                                                  +--------------+   |
+-------------------------------------------------------------------+
```

### Data Flow

1. **Patient data** enters via the LangGraph state machine
2. **Night surveillance** detects events (desaturation, fever, agitation, apnea)
3. **GraphRAG memory** stores and retrieves patient context
4. **MedGemma engine** generates clinical reports using steering prompts
5. **PDF generator** produces professional medical documents

---

## 3. The Model

### Model Identity

| Property | Value |
|----------|-------|
| **Base Model** | `google/medgemma-1.5-4b-it` (Google's HAI-DEF, Instruction-Tuned) |
| **Quantized Version** | `medgemma-1.5-medical-Q4_K_M.gguf` |
| **HuggingFace Repo** | [hmzBen/medgemma-1.5-medical-q4km](https://huggingface.co/hmzBen/medgemma-1.5-medical-q4km) |
| **Architecture** | Gemma 3 (4B parameters) |
| **Format** | GGUF (llama.cpp compatible) |
| **Quantization** | Q4_K_M (Mixed-Precision, I-Matrix calibrated) |
| **File Size** | ~2.4 GB (down from 8.5 GB F16) |
| **Context Window** | 131,072 tokens (model max), 4,096 tokens (default config) |
| **RAM Requirement** | ~3-4 GB during inference |

### Quantization Details

The model was quantized using a SOTA I-Matrix calibration process:

1. **Calibration Dataset** (~40k tokens): 30% Doctor-Patient Dialogue (ChatDoctor), 30% Medical Facts (MedAlpaca), 40% Diagnostic Logic (USMLE Board Exams)
2. **I-Matrix Generation:** Created a "heatmap" of which neurons are medically critical
3. **Q4_K_M Compression:** Medical logic neurons kept at 6-bit precision, grammar neurons compressed to 4-bit
4. **Result:** 8.5 GB --> 2.4 GB while preserving clinical reasoning accuracy

### Model Location

The quantized model file must be placed at:
```
models/medgemma-1.5-medical-Q4_K_M.gguf
```

The engine auto-searches these paths (in order):
- `models/`
- `./` (project root)
- `../models/`

Any file matching `*medgemma*.gguf` is detected, with preference for Q4_K_M variants.

---

## 4. Project Structure

```
medgemma_project/
|-- models/                              # Model files (GGUF)
|   |-- medgemma-1.5-medical-Q4_K_M.gguf  # Quantized model (2.4 GB)
|   +-- medgemma-1.5-4b-it-F16.gguf       # Full precision (7.8 GB, optional)
|
|-- src/                                 # Source code
|   |-- __init__.py
|   |-- models/                          # Data models (Pydantic)
|   |   |-- patient.py                     # Patient, Condition, Medication, Allergy
|   |   |-- vitals.py                      # SpO2, HeartRate, Temperature, BP readings
|   |   |-- events.py                      # ClinicalEvent, NightEvent, DayEvent
|   |   +-- __init__.py
|   |
|   |-- orchestration/                   # LangGraph workflow
|   |   |-- graph.py                       # MedGemmaSentinelGraph (state machine)
|   |   |-- nodes.py                       # NightNode, DayNode, Rap1Node, Rap2Node
|   |   |-- state.py                       # SentinelState, NightData, DayData
|   |   +-- __init__.py
|   |
|   |-- memory/                          # GraphRAG knowledge base
|   |   |-- patient_graph.py               # PatientGraphRAG (graph operations)
|   |   |-- graph_store.py                 # LocalGraphStore (persistence)
|   |   |-- retriever.py                   # GraphRetriever (context retrieval)
|   |   +-- __init__.py
|   |
|   +-- reporting/                       # Report generation
|       |-- medgemma_engine.py             # MedGemmaEngine (LLM inference)
|       |-- prompts.py                     # Steering prompts + MedGemmaReportGenerator
|       |-- pdf_generator.py               # PDFReportGenerator (ReportLab)
|       |-- templates.py                   # NightReportTemplate, DayReportTemplate
|       +-- __init__.py
|
|-- tests/                               # Test suite (97 tests)
|   |-- test_medgemma_integration.py       # 11 MedGemma-specific tests
|   |-- test_memory.py                     # GraphRAG memory tests
|   |-- test_models.py                     # Data model tests
|   |-- test_orchestration.py              # LangGraph workflow tests
|   |-- test_reporting.py                  # Report generation tests
|   +-- __init__.py
|
|-- examples/                            # Demo scripts
|   |-- demo_workflow.py                   # Complete workflow demo (main)
|   +-- demo_with_medgemma.py              # AI inference demo
|
|-- data/                                # Data directory
|   |-- reports/                           # Generated reports output
|   +-- synthetic/                         # Synthetic test data
|
|-- scripts/
|   +-- download_medgemma.py               # Model download script
|
|-- llama-cpp/                           # llama.cpp binaries (optional)
|   |-- llama-server.exe                   # Local inference server
|   |-- llama-quantize.exe                 # Quantization tool
|   +-- *.dll                              # Required DLLs
|
|-- pytest.ini                           # Pytest configuration
|-- requirements.txt                     # Python dependencies
|-- MEMORY.md                            # Project memory/notes
+-- REPORTING.md                         # This file
```

---

## 5. Dependencies & Versions

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| `llama-cpp-python` | 0.3.16 | Local model inference (loads GGUF files) |
| `pydantic` | 2.11.7 | Data validation and models |
| `reportlab` | 4.4.4 | PDF report generation |
| `networkx` | 3.6.1 | Graph data structure for GraphRAG |
| `langgraph` | 1.0.8 | State machine orchestration |
| `langchain` | 1.2.10 | LLM framework (used by LangGraph) |
| `jinja2` | 3.1.6 | Template rendering |
| `numpy` | 2.2.6 | Numerical operations |
| `requests` | 2.32.5 | HTTP client (HF API, server mode) |
| `huggingface-hub` | 0.36.2 | Model download from HuggingFace |
| `python-dateutil` | 2.9.0 | Date/time utilities |
| `pytest` | 8.3.5 | Test framework |

### Build Dependencies (for llama-cpp-python)

| Tool | Purpose |
|------|---------|
| **MSYS2** with `mingw-w64-x86_64-gcc` | C++ compiler (Windows) |
| **CMake** (via MSYS2 or standalone) | Build system |
| Or **Visual Studio Build Tools** | Alternative C++ compiler (Windows) |
| Or **gcc/g++** | C++ compiler (Linux/macOS) |

### Python Version

- **Minimum:** Python 3.10
- **Tested on:** Python 3.12.6 (Windows 64-bit)

---

## 6. Installation Guide (Any Machine)

### Step 1: Clone or Copy the Project

```bash
# Copy the project folder to the target machine
# Ensure the models/ directory contains the GGUF file
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install C++ Compiler (for llama-cpp-python)

#### Windows (MSYS2 -- Recommended, lightweight)
```powershell
# Install MSYS2
winget install MSYS2.MSYS2

# Install GCC toolchain inside MSYS2
C:\msys64\usr\bin\bash.exe -lc "pacman -S --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make"

# Add to PATH and install llama-cpp-python
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
$env:CMAKE_GENERATOR = "MinGW Makefiles"
pip install llama-cpp-python
```

#### Windows (Visual Studio Build Tools -- Alternative)
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
pip install llama-cpp-python
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt install build-essential cmake
pip install llama-cpp-python
```

#### macOS
```bash
xcode-select --install
pip install llama-cpp-python
```

### Step 4: Get the Model File

**Option A: Download from HuggingFace (if Q4_K_M is uploaded)**
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="hmzBen/medgemma-1.5-medical-q4km",
    filename="medgemma-1.5-medical-Q4_K_M.gguf",
    local_dir="models"
)
```

**Option B: Copy from a teammate**
```bash
# Copy the file to models/medgemma-1.5-medical-Q4_K_M.gguf
```

**Option C: Download F16 and quantize yourself (if you have llama-quantize)**
```bash
# Download F16
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='hmzBen/medgemma-1.5-medical-q4km', filename='medgemma-1.5-4b-it-F16.gguf', local_dir='models')"

# Quantize to Q4_K_M
./llama-cpp/llama-quantize models/medgemma-1.5-4b-it-F16.gguf models/medgemma-1.5-medical-Q4_K_M.gguf Q4_K_M
```

### Step 5: Verify Installation

```bash
python -c "
from src.reporting.medgemma_engine import MedGemmaEngine
engine = MedGemmaEngine()
print(engine.get_status())
"
```

Expected output:
```
[...] Loading model via llama-cpp-python: medgemma-1.5-medical-Q4_K_M.gguf
[OK] Model loaded via llama-cpp-python (medgemma-1.5-medical-Q4_K_M.gguf)
{'loaded': True, 'model_path': 'models\\medgemma-1.5-medical-Q4_K_M.gguf', 'server_url': None, 'temperature': 0.3, 'max_tokens': None, 'mode': 'llama-cpp-python'}
```

---

## 7. How to Run the System

### Run the Structured Demo (Parsing & Report Formatting)

```bash
cd medgemma_project
set PYTHONIOENCODING=utf-8
python -m examples.demo_workflow
```

This is the **main demo** that demonstrates the full workflow with structured report parsing:

1. **LangGraph Workflow Visualization** -- ASCII graph of Night -> RAP1 -> Day -> RAP2 pipeline
2. **GraphRAG Patient Memory** -- Creates patient (Jean Camara, 62yo), stores conditions/medications, retrieves context
3. **MedGemma Steering Prompts** -- Lists all 11 prompt types, shows night/day/cardio prompt configurations
4. **Full Workflow Execution** -- Runs the LangGraph state machine end-to-end with synthetic data (32 vitals, 3 audio, 2 vision events)
5. **Structured Report Generation** -- Parses workflow output into structured sections, renders Markdown via `NightReportTemplate.render_markdown()` and `DayReportTemplate.render_markdown()`, generates PDFs via `PDFReportGenerator`
6. **Context Retrieval** -- Semantic search over the patient graph

**Output** (generated in `data/reports/`):
- `rap1_night_DEMO001.pdf` / `.md` -- Night surveillance report
- `rap2_day_DEMO001.pdf` / `.md` -- Day consultation report

**Key parsing methods used:**
- `_parse_markdown_to_elements()` -- Parses AI-generated markdown into PDF elements
- `_build_alerts_section()` -- Structures alert data by severity
- `_build_vitals_section()` -- Formats vital signs into tables
- `_build_sleep_section()` -- Structures sleep quality analysis
- `_build_recommendations_section()` -- Formats clinical recommendations
- `_build_complaint_section()` -- Structures chief complaint for day reports
- `_build_exam_section()` -- Formats examination findings
- `_build_ai_analysis_section()` -- Structures MedGemma AI output
- `render_markdown()` / `render_html()` -- Template-based report rendering

### Run the Complete Demo (Main Workflow)

```bash
cd medgemma_project
set PYTHONIOENCODING=utf-8
python -m examples.demo_workflow
```

This executes the full workflow with **all components**:
1. Visualizes LangGraph workflow structure
2. Creates a demo patient (Jean Camara, 62yo, cardiac/DM2) in GraphRAG
3. Demonstrates MedGemma steering prompts (11 prompt types)
4. Runs Night -> Rap1 -> Day -> Rap2 LangGraph workflow
5. Generates 6 clinical matplotlib plots (vitals trends, events, severity)
6. Generates Night + Day reports as PDF (with embedded plots) and Markdown
7. Demonstrates GraphRAG context retrieval and semantic search

### Run Tests

```bash
python -m pytest tests/ -v
```

All 97 tests should pass (including 11 MedGemma integration tests).

### Use the Engine in Your Code

```python
from src.reporting.medgemma_engine import MedGemmaEngine

# Engine auto-detects the model in models/ folder
engine = MedGemmaEngine()

# Generate a night report
report = engine.generate_night_report(
    patient_context="Jean Camara, 73yo male, COPD stage II",
    night_summary="Night with 4 detected events",
    events=[
        {"type": "desaturation", "severity": "high", "description": "SpO2: 86%"},
        {"type": "apnea", "severity": "critical", "description": "Apnea 11s"},
    ]
)
print(report)
```

### Use the Report Generator (Higher-Level API)

```python
from src.reporting.prompts import MedGemmaReportGenerator

gen = MedGemmaReportGenerator()

# Night report
report = gen.generate_night_report(
    patient_context={"name": "Jean Camara", "id": "P001"},
    night_data={"events": [...], "total_events": 4}
)

# Day report
report = gen.generate_day_report(
    patient_context={"name": "Jean Camara", "id": "P001"},
    night_context="Night with desaturation events",
    day_data={"symptoms": ["chest pain"], "consultation_mode": "cardio"},
    specialty="cardio"
)
```

---

## 8. Built Functionalities

### 8.1 Data Models (`src/models/`)

| Class | File | Description |
|-------|------|-------------|
| `Patient` | patient.py | Full patient record (demographics, conditions, medications, allergies, risk factors) |
| `Condition` | patient.py | Medical condition with ICD code and status |
| `Medication` | patient.py | Medication with dosage, frequency, route |
| `Allergy` | patient.py | Allergy with substance, severity, reaction |
| `SpO2Reading` | vitals.py | Oxygen saturation with automatic status classification (normal/low/critical) |
| `HeartRateReading` | vitals.py | Heart rate with bradycardia/tachycardia detection |
| `TemperatureReading` | vitals.py | Temperature with fever detection |
| `BloodPressureReading` | vitals.py | BP with hypertension classification + MAP calculation |
| `ClinicalEvent` | events.py | Base clinical event with type, severity, acknowledgement |
| `NightEvent` | events.py | Night-specific event with multimodal data sources |
| `DayEvent` | events.py | Day consultation event |
| `AlertLevel` | events.py | Enum: CRITICAL, HIGH, MEDIUM, LOW |
| `EventType` | events.py | Enum: DESATURATION, FEVER, AGITATION, APNEA, FALL, etc. |

### 8.2 MedGemma LLM Engine (`src/reporting/medgemma_engine.py`)

The core inference engine. See [Section 9](#9-medgemma-engine-inference-modes) for details.

**Key Methods:**
- `generate_night_report()` -- AI-generated night surveillance report
- `generate_day_report()` -- AI-generated consultation report
- `analyze_symptoms()` -- Differential diagnosis from symptom list
- `get_status()` -- Current engine mode and configuration

### 8.3 Steering Prompts (`src/reporting/prompts.py`)

Dynamic prompt system that "steers" MedGemma's behavior for different clinical scenarios. See [Section 10](#10-steering-prompts-system).

**Available Prompt Types:**
- `NIGHT_SURVEILLANCE` -- Real-time night monitoring analysis
- `NIGHT_REPORT_GENERATION` -- Structured RAP1 report
- `DAY_CONSULTATION` -- General/specialty consultation
- `DAY_REPORT_GENERATION` -- Structured RAP2 report
- `TRIAGE_ASSESSMENT` -- Emergency triage evaluation
- `LONGITUDINAL_ANALYSIS` -- Multi-day trend analysis
- `CARDIO_ANALYSIS` -- Cardiology specialty mode
- `DERMATO_ANALYSIS` -- Dermatology specialty mode
- `OPHTALMO_ANALYSIS` -- Ophthalmology specialty mode

### 8.4 LangGraph Orchestration (`src/orchestration/`)

State machine workflow that sequences the clinical pipeline. See [Section 12](#12-langgraph-orchestration).

**Nodes:** NightNode --> Rap1Node --> DayNode --> Rap2Node

### 8.5 GraphRAG Memory (`src/memory/`)

Patient knowledge graph for context storage and retrieval. See [Section 13](#13-graphrag-memory-system).

### 8.6 PDF Report Generator (`src/reporting/pdf_generator.py`)

Professional medical PDF output using ReportLab. See [Section 14](#14-pdf-report-generator).

### 8.7 Report Templates (`src/reporting/templates.py`)

HTML/Markdown templates for night and day reports with proper medical formatting.

---

## 9. MedGemma Engine (Inference Modes)

The engine supports **4 inference modes**, tried in priority order:

### Mode 1: Local llama.cpp Server (`server`)

```bash
# Start the server (from llama-cpp/ folder)
./llama-cpp/llama-server.exe -m models/medgemma-1.5-medical-Q4_K_M.gguf --port 8080 -c 4096
```

The engine auto-detects a running server at `http://localhost:8080` and uses the OpenAI-compatible `/v1/chat/completions` endpoint.

**Pros:** Fastest, GPU offloading possible (`-ngl 20`), shared across multiple Python processes.

### Mode 2: llama-cpp-python Direct (`llama-cpp-python`)

The engine loads the GGUF model directly into Python memory using `llama-cpp-python`.

**Pros:** No separate server process, fully self-contained.
**Cons:** Model stays loaded in RAM for the process lifetime, slower startup.

This is the **current default mode** when the model file exists in `models/`.

### Mode 3: HuggingFace Inference API (`huggingface`)

Remote inference via HuggingFace (requires internet + `HF_TOKEN` env var).

```bash
set HF_TOKEN=hf_your_token_here
python -m examples.demo_workflow
```

**Note:** As of February 2026, no inference provider hosts MedGemma remotely. This mode is future-proofed for when providers add support.

### Mode 4: Simulation (`simulation`)

Generates template-based reports without the model. Used for development and testing when no model is available.

### Mode Selection Logic

```
Engine Init
    |
    +--> Check localhost:8080/health --> [OK] --> "server" mode
    |                                   [FAIL]
    +--> Find GGUF in models/ --> Load via llama-cpp-python --> [OK] --> "llama-cpp-python" mode
    |                                                          [FAIL]
    +--> Check HF_TOKEN env var --> Verify API --> [OK] --> "huggingface" mode
    |                                             [FAIL]
    +--> Fallback --> "simulation" mode
```

### Engine Configuration

```python
engine = MedGemmaEngine(
    model_path="models/medgemma-1.5-medical-Q4_K_M.gguf",  # Explicit path (optional)
    temperature=0.3,          # Lower = more deterministic (0.0-1.0)
    max_tokens=-1,            # -1 = unlimited output
    n_ctx=4096,               # Context window size
    n_gpu_layers=0,           # GPU layers (0 = CPU only, 20 = partial GPU)
    server_url="http://localhost:8080",  # Custom server URL
    hf_token="hf_xxx",       # HuggingFace token
)
```

---

## 10. Steering Prompts System

The steering system dynamically configures MedGemma's behavior for each clinical scenario using structured `SteeringPrompt` objects.

### How It Works

1. Each prompt type has a **system prompt** (defines role and behavior) and a **user prompt template** (provides patient data)
2. The system prompt "steers" MedGemma into the correct mode (night guard, cardiologist, triage nurse, etc.)
3. Templates use `$variable` placeholders filled at runtime
4. Output sections define the expected report structure

### Example: Night Surveillance Prompt

```python
from src.reporting.prompts import MedGemmaPrompts, PromptType

prompt = MedGemmaPrompts.get_prompt(PromptType.NIGHT_SURVEILLANCE)

# Fill with patient data
formatted = prompt.format_user_prompt(
    patient_id="P001",
    patient_context="73yo male, COPD",
    time_window="60",
    vitals_data="SpO2: 86%, HR: 110",
    audio_events="Coughing detected",
    vision_events="Agitation detected"
)
```

### Alert Levels in Night Mode

| Level | Criteria | Action |
|-------|----------|--------|
| **CRITICAL** | SpO2 < 85%, HR < 40 or > 150, prolonged apnea | Immediate intervention |
| **HIGH** | SpO2 85-90%, respiratory anomalies | Urgent evaluation |
| **MEDIUM** | Moderate anomalies | Enhanced surveillance |
| **LOW** | Minor findings | Note in report |

### Specialty Modes

The day consultation prompt adapts to specialty:
- **General:** Structured interview, differential diagnosis, investigations
- **Cardio:** ECG analysis, cardiovascular risk stratification, ASCVD scoring
- **Dermato:** ABCDE lesion analysis, morphology classification
- **Ophtalmo:** Fundoscopy, diabetic retinopathy, glaucoma screening

---

## 11. Report Generation Pipeline

### Night Report (RAP1) Flow

```
Patient Data + Night Events
        |
        v
MedGemmaReportGenerator.generate_night_report()
        |
        v
MedGemmaEngine.generate_night_report()
        |
        +--> Build prompt (patient context + events)
        +--> Send to model (chat completion)
        +--> Strip thinking/reasoning from output
        +--> Return Markdown report
        |
        v
Save as .md file + Generate PDF via PDFReportGenerator
```

### Night Report Sections

1. **Executive Summary** -- 2-3 sentence overview
2. **Critical Alerts** -- Timestamped severe events
3. **Sleep Quality** -- Score and observations
4. **Vital Signs Evolution** -- Trends and anomalies
5. **Interventions** -- Actions taken during the night
6. **Recommendations** -- Action items for the day team

### Day Report (RAP2) Flow

```
Patient Data + Night Context + Consultation Data
        |
        v
MedGemmaReportGenerator.generate_day_report()
        |
        v
MedGemmaEngine.generate_day_report()
        |
        +--> Build prompt (patient + night summary + consultation)
        +--> Send to model with specialty context
        +--> Strip thinking/reasoning from output
        +--> Return Markdown report
        |
        v
Save as .md file + Generate PDF via PDFReportGenerator
```

### Day Report Sections

1. **Patient Identification** -- ID, demographics
2. **Night Summary** -- Carried from RAP1
3. **Chief Complaint** -- Presenting symptoms
4. **Examination Findings** -- Physical exam results
5. **Differential Diagnoses** -- Ranked with reasoning
6. **Investigations** -- Recommended tests
7. **Treatment Plan** -- Evidence-based management

---

## 12. LangGraph Orchestration

### State Machine

The `MedGemmaSentinelGraph` implements a LangGraph state machine with 4 nodes:

```
[START] --> NightNode --> Rap1Node --> DayNode --> Rap2Node --> [END]
```

### State Model (`SentinelState`)

```python
class SentinelState:
    patient_id: str
    phase: WorkflowPhase      # NIGHT, RAP1, DAY, RAP2
    night_data: NightData      # Events, alerts, vitals
    day_data: DayData          # Consultation data
    report_data: ReportData    # Generated reports
    steering_mode: SteeringMode  # Current prompt mode
```

### Workflow Phases

| Phase | Node | Description |
|-------|------|-------------|
| `NIGHT` | NightNode | Process night surveillance events |
| `RAP1` | Rap1Node | Generate night report |
| `DAY` | DayNode | Process day consultation data |
| `RAP2` | Rap2Node | Generate day report |

### Usage

```python
from src.orchestration.graph import MedGemmaSentinelGraph

graph = MedGemmaSentinelGraph()
result = graph.run(
    patient_id="P001",
    night_events=[...],
    day_data={...}
)
```

---

## 13. GraphRAG Memory System

### Components

| Class | File | Purpose |
|-------|------|---------|
| `PatientGraphRAG` | patient_graph.py | High-level graph operations (add patient, add event, query) |
| `LocalGraphStore` | graph_store.py | Persistent graph storage using NetworkX |
| `GraphRetriever` | retriever.py | Context retrieval for report generation |

### How It Works

1. **Patient Registration:** Creates a graph node with demographics, conditions, medications, allergies
2. **Event Recording:** Adds clinical event nodes connected to the patient
3. **Context Retrieval:** Queries the graph for relevant patient context before report generation

### Example

```python
from src.memory.patient_graph import PatientGraphRAG
from src.memory.retriever import GraphRetriever

# Initialize
memory = PatientGraphRAG()
retriever = GraphRetriever(memory)

# Add patient
memory.add_patient(
    patient_id="P001",
    name="Jean Camara",
    age=73,
    conditions=["COPD", "Hypertension", "Diabetes T2"],
    medications=["Amlodipine", "Metformine", "Spiriva"],
    allergies=["Penicilline"],
    risk_factors=["Age > 65", "Obesity"],
    room="500"
)

# Add clinical event
memory.add_clinical_event(
    patient_id="P001",
    event_type="desaturation",
    description="SpO2 dropped to 86%",
    severity="high"
)

# Retrieve context for night report
context = retriever.get_patient_context_for_night("P001")
```

---

## 14. PDF Report Generator

### Features

- **Professional medical layout** using ReportLab
- **A4 format** with headers, footers, page numbers
- **Color-coded alerts** (red for critical, orange for high)
- **Structured sections** matching clinical report standards
- **Auto-generated filenames** with patient ID and timestamp

### Report Styles

| Style | Description |
|-------|-------------|
| `CLINICAL` | Professional medical format (default) |
| `SUMMARY` | Condensed single-page summary |
| `DETAILED` | Full detailed report with all data |

### Output Location

Reports are saved to `data/reports/`:
```
data/reports/
|-- rap1_night_MEDGEMMA_001_20260218_1551.pdf
|-- rap2_day_MEDGEMMA_001_20260218_1553.pdf
|-- rap1_night_MEDGEMMA_001.md
+-- rap2_day_MEDGEMMA_001.md
```

### Usage

```python
from src.reporting.pdf_generator import PDFReportGenerator

pdf_gen = PDFReportGenerator()

# Night PDF
pdf_path = pdf_gen.generate_night_report({
    "patient_id": "P001",
    "patient_name": "Jean Camara",
    "room": "500",
    "summary": "Night with 4 events detected...",
    "events": [...],
    "night_data": {...},
    "vitals_summary": {"SpO2": {"min": 86, "max": 98, "avg": 94}},
    "recommendations": ["Surveillance SpO2", "Clinical reassessment"]
})

# Day PDF
pdf_path = pdf_gen.generate_day_report({
    "patient_id": "P001",
    "patient_name": "Jean Camara",
    "room": "500",
    "summary": "Cardiology consultation...",
    "day_data": {...},
    "recommendations": ["ECG", "Troponin levels"]
})
```

---

## 15. Testing

### Test Suite Overview

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_medgemma_integration.py` | 11 | MedGemma engine, report generator, full workflow |
| `test_memory.py` | 16 | GraphRAG, graph store, retriever |
| `test_models.py` | 28 | Patient, vitals, events, alert levels |
| `test_orchestration.py` | 20 | LangGraph nodes, state, graph |
| `test_reporting.py` | 22 | Prompts, templates, PDF generator |
| **Total** | **97** | **All passing** |

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_medgemma_integration.py -v

# Single test
python -m pytest tests/test_medgemma_integration.py::TestMedGemmaEngine::test_engine_initialization -v
```

### Key Test Classes

- `TestMedGemmaEngine` -- Engine init, status, night/day report generation, symptom analysis
- `TestMedGemmaReportGenerator` -- Higher-level generator with patient context
- `TestMedGemmaIntegration` -- End-to-end workflow with demo patient

---

## 16. Troubleshooting

### "llama-cpp-python build failed: CMAKE_C_COMPILER not set"

You need a C++ compiler. Install MSYS2 on Windows:
```powershell
winget install MSYS2.MSYS2
C:\msys64\usr\bin\bash.exe -lc "pacman -S --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make"
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
$env:CMAKE_GENERATOR = "MinGW Makefiles"
pip install llama-cpp-python
```

### "No GGUF model file found"

Place the model file in `models/medgemma-1.5-medical-Q4_K_M.gguf`. The engine searches `models/`, `./`, and `../models/` for any file matching `*medgemma*.gguf`.

### "Engine mode: simulation" (model not loading)

Check:
1. Model file exists: `ls models/*.gguf`
2. `llama-cpp-python` is installed: `pip show llama-cpp-python`
3. File is not corrupted: verify file size is ~2.4 GB for Q4_K_M

### "n_ctx_per_seq < n_ctx_train" warning

This is normal. The model supports up to 131,072 tokens but we use 4,096 for efficiency. To increase:
```python
engine = MedGemmaEngine(n_ctx=8192)
```

### "UnicodeEncodeError: charmap codec can't encode character"

Windows terminal encoding issue. Set UTF-8:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### PDF generation warning: "NoneType comparison"

Non-blocking edge case in night PDF template when some vitals data is None. Reports still generate correctly.

### Tests take a long time

When the real model is loaded, each test that creates an `MedGemmaEngine` instance loads the model (~10-15 seconds). Total suite runs in ~55 minutes with model loaded. Without model (simulation mode), tests complete in seconds.

---

## 17. API Reference

### MedGemmaEngine

```python
class MedGemmaEngine:
    def __init__(self, hf_token=None, server_url=None, model_path=None,
                 temperature=0.3, max_tokens=-1, n_ctx=4096, n_gpu_layers=0)

    def generate_night_report(self, patient_context: str, night_summary: str,
                               events: List[Dict]) -> str

    def generate_day_report(self, patient_context: str, night_context: str,
                             consultation_data: Dict, specialty: str = "general") -> str

    def analyze_symptoms(self, symptoms: List[str], patient_context: str) -> Dict

    def get_status(self) -> Dict[str, Any]
```

### MedGemmaReportGenerator

```python
class MedGemmaReportGenerator:
    def __init__(self)

    def generate_night_report(self, patient_context: Dict, night_data: Any) -> str

    def generate_day_report(self, patient_context: Dict, night_context: str,
                             day_data: Dict, specialty: str = "general") -> str

    def get_engine_status(self) -> Dict[str, Any]
```

### MedGemmaPrompts

```python
class MedGemmaPrompts:
    @classmethod
    def get_prompt(cls, prompt_type: PromptType, **kwargs) -> SteeringPrompt

    @classmethod
    def list_prompts(cls) -> List[Dict[str, str]]

    @classmethod
    def get_night_surveillance_prompt(cls) -> SteeringPrompt
    def get_night_report_prompt(cls) -> SteeringPrompt
    def get_day_consultation_prompt(cls, specialty="general") -> SteeringPrompt
    def get_day_report_prompt(cls) -> SteeringPrompt
    def get_triage_prompt(cls) -> SteeringPrompt
    def get_longitudinal_prompt(cls) -> SteeringPrompt
```

### PatientGraphRAG

```python
class PatientGraphRAG:
    def add_patient(self, patient_id, name, age, conditions, medications,
                    allergies, risk_factors, room) -> None

    def get_patient_context(self, patient_id: str) -> Dict

    def get_patient_summary(self, patient_id: str) -> str

    def add_clinical_event(self, patient_id, event_type, description, severity) -> None

    def add_consultation(self, patient_id, specialty, findings, recommendations) -> None

    def get_statistics(self) -> Dict
```

### PDFReportGenerator

```python
class PDFReportGenerator:
    def generate_night_report(self, data: Dict) -> str  # Returns PDF file path

    def generate_day_report(self, data: Dict) -> str     # Returns PDF file path
```

---

*Documentation generated: February 18, 2026*
*System version: MedGemma Sentinel v1.0*
*Model: medgemma-1.5-medical-Q4_K_M.gguf (2.4 GB, Q4_K_M quantized)*
*Tests: 97/97 passing*
