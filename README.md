# MedGemma Sentinel: The Autonomous Edge-AI Guardian 🏥🛡️

### adds-on : the project schema

**MedGemma Sentinel** is a multimodal AI agent designed for offline rural clinics (e.g., the Atlas Mountains) to bridge the "Vigilance Gap" in healthcare.  It operates on low-cost hardware and switches between two specialized roles to support medical staff 24/7.

## 🌟 Key Features

* **Night Cardiology Sentinel:** Continuous vitals monitoring with windowed analysis and structured clinical summaries.
* **MCP Orchestration:** ReAct loop with tool use for cardiology analysis, patient history, and real-time context.
* **Edge-Ready Inference:** GGUF + I-Matrix quantized MedGemma for low-memory, offline deployment.
* **Longitudinal Memory:** Local patient timeline and report generation with GraphRAG.

## 🏗️ System Architecture

MedGemma Sentinel is a **Hierarchical Agent** managed by a State Machine:

1. **Reflex Layer:** Edge sensing (vision + audio) triggers safety alerts and sends events upstream.
2. **Cognitive Layer:** MedGemma-based reasoning with ReAct tool orchestration via MCP.
3. **Memory + Reporting:** GraphRAG links daily reports, vitals trends, and generates longitudinal summaries.
4. **Clinical Delivery:** Actionable instructions and escalation criteria for the night nurse.

## 📋 Technical Stack

* **LLM:** MedGemma 2 (4B) in GGUF (I-Matrix quantized)
* **Orchestration:** MCP, ReAct loop, LangGraph
* **Memory:** LlamaIndex + GraphRAG
* **Runtime/UI:** Streamlit, llama.cpp, Ollama (local inference)
* **Vision/Audio (Reflex):** HeAR, YamNet, Faster-Whisper

## 💓 Night Cardiology Sentinel Apps

Two Streamlit apps are available:

- **Standalone Analyzer:** Upload vitals and run windowed clinical analysis.
- **MCP Cardiology Agent:** Interactive ReAct loop with tool-calling and actionable instructions.

### Features

- **Patient Selection:** Upload patient demographics via JSON or enter manually
- **Vitals Processing:** Parse heart rate and vital signs from text files (supports two formats)
- **Windowing Analysis:** Process data in 15-minute time windows or 10-row chunks
- **Clinical Insights:** Generate structured clinical analysis with:
  - **Comparison:** Current data vs patient baseline
  - **Detection:** Identification of clinical anomalies
  - **Interpretation:** Short clinical assessment
- **Flexible Model Loading:** Use local GGUF files or download from Hugging Face

### Installation

1. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Install Ollama and pull the model:**

   ```bash
   ollama pull amsaravi/medgemma-4b-it:q6
   ```
4. **Add your Hugging Face token:**

   - Create a `.env` file in [src/mcp_architecture](src/mcp_architecture) using `.env.example` as a template.
5. **Optional (edge device): Download the model locally:**

   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli download Ismailea04/medgemma-night-sentinel --local-dir ./models
   ```
6. **Optional: Install llama-cpp-python (if not already installed):**

   ```bash
   # For CPU
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

   # For CUDA (if you have NVIDIA GPU)
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```

### Usage

1. **Start the app:**

   ```bash
   # Standalone analyzer
   streamlit run app_night_cardiology_sentinal.py

   # MCP cardiology agent
   streamlit run app_mcp_cardiology.py
   ```
2. **In the web interface:**

   - **Step 1:** Upload `subjects_info.json` (example: `data/processed/hr_adolescent/subjects_info.json`) or enter patient details manually
   - **Step 2:** Upload vitals text file (example: `data/processed/hr_adolescent/subject_903_hr.txt`)
   - **Step 3:** Choose windowing mode (15-minute or 10-row)
   - **Step 4:** Configure model source:
     - **Local:** `models/medgemma-night-sentinel-Q4_K_M.gguf`
     - **Hugging Face:** `Ismailea04/medgemma-night-sentinel`
   - **Step 5:** Click **"🚀 Run analysis"**
3. **View results:**

   - Each window displays summary statistics and clinical analysis
   - Export results as needed

### Supported Vitals Formats

**Format 1: Simple Heart Rate**

```
Time: 00:00 - heart rate [#/min]: 64
Time: 00:01 - heart rate [#/min]: 67
```

**Format 2: Multi-parameter Key-Value**

```
Time: 2s - HR: 69.92, PULSE: 68.02, RESP: 18.97, %SpO2: 97.92
Time: 4s - HR: 70.15, PULSE: 68.50, RESP: 19.02, %SpO2: 98.01
```

### Project Structure

```
medgemma-sentinel/
├── app_night_cardiology_sentinal.py    # Standalone analyzer
├── app_mcp_cardiology.py               # MCP cardiology app
├── src/
│   └── night_cardiology_sentinel/
│       ├── __init__.py
│       ├── data_parser.py              # Vitals parsing & windowing
│       └── inference.py                # Model inference
├── models/
│   └── medgemma-night-sentinel-Q4_K_M.gguf
├── data/
│   └── processed/
│       └── hr_adolescent/
│           ├── subjects_info.json
│           └── subject_*.txt
└── figures/
    ├── logo2.png
    ├── card_medgemma_sentinal.png
    └── workflow.png
```

### Example Use Case

**Scenario:** Night shift monitoring of a 65-year-old male post-operative patient

1. Upload patient profile (baseline HR: 70 bpm)
2. Monitor continuous vitals stream
3. System detects: HR spike to 135 bpm, irregular rhythm, SpO2 92%
4. **Night Sentinel Report:**
   - **Comparison:** Significant deviation from baseline (65 bpm increase)
   - **Detection:** Tachycardia with arrhythmia + mild hypoxemia
   - **Interpretation:** Possible atrial fibrillation or early sepsis - requires immediate clinical assessment

## 👥 The "Special Forces" Team

* **Infrastructure:** Hamza — Model quantization and API deployment.
* **Steering:** Ismail — Activation engineering and vector injection.
* **Audio:** Youssra — Event detection and speech-to-text.
* **Vision & UI:** Saad/Othman — Fall detection and Streamlit dashboard.
* **Memory & Scribe:** Saad/Othman — LangGraph orchestration and PDF reporting.

## 🚀 Deployment

Designed for the **Kaggle MedGemma Impact Challenge**. Deadline: February 24, 2026.
