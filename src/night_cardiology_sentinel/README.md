# Night Cardiology Sentinel - Vitals Monitoring & Windowing 💓

Lightweight module for processing cardiac vital signs with local inference and anomaly detection.

## 📋 Overview

This module provides real-time vital signs monitoring with structured analysis:
- **Data Parsing:** Extract vitals from multiple text formats
- **Windowing:** Organize vitals into temporal or chunk-based windows
- **Local Inference:** Run MedGemma model locally for low-latency analysis
- **Clinical Analytics:** Compare current vitals to baseline and detect anomalies

## 📁 Core Files

### `data_parser.py`
Handles all vitals data extraction and windowing logic.

**Key Functions:**
- `parse_vitals_lines(lines)` - Parse vitals from text format into structured rows
  - Supports two formats:
    - Simple: `Time: HH:MM - heart rate [#/min]: VALUE`
    - Complex: `Time: Xs - HR: X, PULSE: Y, RESP: Z, %SpO2: W`
  - Returns: List of dicts with `{time, type, value}` or `{time, hr, pulse, resp, spo2}`

- `chunk_rows(rows, mode)` - Divide vitals into windows
  - `"15-minute windows"` - Group by temporal 15-min blocks
  - `"10-row windows"` - Group by fixed count (10 rows per window)
  - Returns: List of window lists

- `summarize_window(rows)` - Extract statistics from vitals window
  - Calculates: min, max, mean, median for HR, SpO2, RESP, PULSE
  - Detects anomalies: tachycardia, hypoxemia, tachypnea
  - Returns: Structured summary string

**Usage Example:**
```python
from data_parser import parse_vitals_lines, chunk_rows, summarize_window

# Read vitals file
with open("vitals.txt") as f:
    lines = f.readlines()

# Parse into structure
rows = parse_vitals_lines(lines)
# [{'time': '00:00', 'hr': 72, 'spo2': 97, ...}, ...]

# Create windows
windows = chunk_rows(rows, "15-minute windows")
# [[row1, row2, ...], [row5, row6, ...], ...]

# Summarize each window
for window in windows:
    summary = summarize_window(window)
    print(summary)
```

---

### `inference.py`
Local model loading and inference engine.

**Key Classes:**
- `SentinelInference` - Wrapper for local MedGemma GGUF model
  - `__init__(model_path, n_ctx)` - Load GGUF model with context size
  - `predict(prompt, max_tokens)` - Run inference and return response
  - `resolve_model_path()` - Handle local/HuggingFace model paths

**Key Functions:**
- `build_prompt(subject_info, window_summary)` - Create cardiology-focused prompt
  - Injects: patient demographics, baseline HR, current vitals summary
  - Template: "You are Night Cardiology Sentinel analyzing cardiac data..."

- `chunk_rows(rows, mode)` - (Re-exported from data_parser)

- `parse_vitals_lines(lines)` - (Re-exported from data_parser)

- `summarize_window(rows)` - (Re-exported from data_parser)

**Model Loading Example:**
```python
from inference import SentinelInference

# Load from local GGUF
sentinel = SentinelInference("./models/medgemma-4b.gguf", n_ctx=2048)

# Generate response
response = sentinel.predict(
    prompt="Patient 65yo with HR spike to 135...",
    max_tokens=256
)
print(response)
```

---

## 🧠 Prompt Engineering

The Night Sentinel is activated through prompt steering (no fine-tuning needed).

**Base Personality Prompt:**
```
You are the Night Cardiology Sentinel - a specialized AI guardian 
focused on continuous cardiac monitoring and early warning detection.

Your role:
1. Monitor vital trends and anomalies
2. Compare current values against patient baseline
3. Detect clinical red flags (tachycardia, hypoxemia, arrhythmia)
4. Provide actionable clinical assessment
5. Recommend immediate interventions if needed

Format your response:
## Comparison
[How current vitals compare to baseline]

## Detection
[Identified anomalies and severity]

## Interpretation
[Clinical assessment and recommendations]
```

---

## 📊 Supported Vitals Parameters

| Parameter | Abbreviation | Normal Range | Alert Threshold |
|-----------|--------------|--------------|-----------------|
| Heart Rate | HR | 60-100 bpm | >120 or <50 |
| Oxygen Saturation | SpO2 | >95% | <92% |
| Respiratory Rate | RESP | 12-20 /min | >24 or <10 |
| Pulse | PULSE | 60-100 bpm | >120 or <50 |
| Blood Pressure | BP | <120/80 | >140/90 or <90/60 |

---

## 🔄 Data Flow

```
Text Vitals File
       │
       ▼
parse_vitals_lines()  ─────→  Structured Rows
       │                       [{time, hr, spo2, resp, pulse}, ...]
       │
       ▼
chunk_rows()  ────────────→  Windows
       │                      [[row1, row2, ...], [row5, row6, ...]]
       │
       ▼
summarize_window()  ──────→  Summaries
       │                      "HR range: 72-85 bpm, mean 78"
       │
       ▼
build_prompt()  ───────────→  Prompt with Context
       │                       "Patient 65yo baseline HR 70..."
       │
       ▼
SentinelInference.predict()  → Clinical Response
                               "## Comparison: Normal baseline..."
```

---

## 🎯 Quick Start

### Basic Usage
```bash
# 1. Start the Streamlit app
streamlit run app_night_cardiology_sentinal.py

# 2. In the web interface:
# - Upload subjects_info.json
# - Upload vitals.txt
# - Choose windowing mode
# - Select model source
# - Click "🚀 Run analysis"
```

### Programmatic Usage
```python
from night_cardiology_sentinel.data_parser import parse_vitals_lines, chunk_rows, summarize_window
from night_cardiology_sentinel.inference import SentinelInference, build_prompt, SubjectInfo

# Read vitals
with open("patient_vitals.txt") as f:
    lines = f.readlines()

# Parse
rows = parse_vitals_lines(lines)
windows = chunk_rows(rows, "15-minute windows")

# Create subject
subject = SubjectInfo(
    subject_code=402,
    age_years=65,
    gender="M",
    weight_kg=80,
    length_cm=175
)

# Load model
sentinel = SentinelInference("models/medgemma-night-sentinel-Q4_K_M.gguf", n_ctx=2048)

# Analyze each window
for i, window in enumerate(windows):
    summary = summarize_window(window)
    prompt = build_prompt(subject, summary)
    response = sentinel.predict(prompt, max_tokens=256)
    print(f"Window {i+1}:\n{response}\n")
```

---

## 📦 Dependencies

- `llama-cpp-python` - Local GGUF inference
- `huggingface_hub` - Model downloading
- `numpy` - Numerical operations
- `streamlit` - Web interface (for app)

Install with:
```bash
pip install llama-cpp-python huggingface-hub numpy streamlit
```

---

## 🧪 Testing

```bash
# Test vitals parsing
python -m pytest tests/test_data_parser.py -v

# Test inference
python -m pytest tests/test_inference.py -v
```

---

## 📈 Clinical Features

### Baseline Comparison
- Stores patient baseline HR from subject info
- Calculates deviation: `current HR - baseline HR`
- Flags significant changes (>20% deviation)

### Anomaly Detection
- **Tachycardia:** HR > 100 bpm
- **Bradycardia:** HR < 60 bpm
- **Hypoxemia:** SpO2 < 92%
- **Tachypnea:** RR > 24 /min
- **Arrhythmia:** Detected via HR variability and patterns

### Statistical Summaries
Per window:
- Minimum, maximum, mean, median for each parameter
- Standard deviation for HR and SpO2
- Trend direction (rising, falling, stable)

---

## 🚀 Integration Points

This module integrates with:
- **MCP Architecture:** Via cardiology_sentinel.py for ReAct-based reasoning
- **Reporting Model:** Via orchestration/nodes.py for full pipeline
- **Streamlit Apps:** Via app_night_cardiology_sentinal.py for web interface

---

## 📚 Model Information

**MedGemma Night Sentinel (Q4_K_M):**
- Base: Google MedGemma 2 (4B)
- Quantization: Q4_K_M (GGUF format)
- Context: 2048 tokens default
- VRAM: ~3-4GB (optimized for edge devices)
- Latency: ~100-200ms per inference

Download:
```bash
huggingface-cli download Ismailea04/medgemma-night-sentinel --local-dir ./models
```

---

**Last Updated:** February 24, 2026  
**Version:** 1.5

### Format 1: Simple HR
```
Time: 00:00 - heart rate [#/min]: 64
Time: 00:01 - heart rate [#/min]: 67
```

### Format 2: Key-Value Pairs
```
Time: 2s - HR: 69.92, PULSE: 68.02, RESP: 18.97, %SpO2: 97.92
Time: 4s - HR: 70.15, PULSE: 68.50, RESP: 19.02, %SpO2: 98.01
```

## Model Sources

- **Local**: Point to a `.gguf` file on disk (default: `models/medgemma-night-sentinel-Q4_K_M.gguf`)
- **Hugging Face**: Download from a HF repo (requires `huggingface-cli login`)

## Example Files

- Subject info: `data/processed/hr_adolescent/subjects_info.json`
- Vitals data: `data/processed/hr_adolescent/subject_903_hr.txt`
