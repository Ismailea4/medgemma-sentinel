# MCP Architecture - Real-Time Clinical Reasoning 🧠

This module implements the **Model Context Protocol (MCP)** server and client for autonomous clinical decision-making using a ReAct (Reasoning + Acting) loop with tool orchestration.

## 📋 Overview

The MCP architecture enables real-time medical reasoning by:
1. **Registering clinical tools** on the MCP server
2. **Executing ReAct loops** where an LLM reasons about patient data and calls tools
3. **Processing observations** to update clinical understanding
4. **Generating actionable instructions** for medical staff

## 📁 Core Files

### `medical_mcp_server.py`
Implements the MCP server that exposes clinical analysis tools.

**Key Functions:**
- `initialize_server()` - Register all available tools
- `analyze_cardiology_sentinel(patient_id, vitals_text)` - Analyze cardiac vitals using MedGemma
- `get_patient_vitals(patient_id)` - Retrieve current vital signs
- `get_patient_history(patient_id)` - Get longitudinal patient records
- `get_current_time()` - Provide temporal context
- `analyze_room_audio(audio_data)` - Process audio from monitoring station

**Tool Registration:**
```python
# Register tools with their schemas
tools = [
    Tool(name="analyze_cardiology_sentinel", ...),
    Tool(name="get_patient_vitals", ...),
    Tool(name="get_patient_history", ...),
    Tool(name="get_current_time", ...),
    Tool(name="analyze_room_audio", ...),
]
```

---

### `medical_mcp_client_2.py`
Implements the ReAct loop client that orchestrates the reasoning + acting cycle.

**Key Functions:**
- `run_react_loop(user_query, max_iterations=8)` - Execute autonomous reasoning
- `parse_action(llm_output)` - Extract Action and Action Input from LLM
- `call_tool(tool_name, arguments)` - Invoke MCP tools
- `update_prompt_context(observation)` - Incorporate observations into reasoning

**ReAct Loop Pattern:**
```
Question: User's medical query
├─ Thought: LLM reasons about next step
├─ Action: Select tool to use
├─ Action Input: JSON parameters
├─ Observation: Tool result
├─ Thought: Process observation
├─ Action: Next tool or final answer
└─ Final Answer: Clinical conclusion
```

---

### `cardiology_sentinel.py`
Cardiology-specific inference and model steering for the Night Sentinel.

**Key Classes:**
- `SubjectInfo` - Patient demographics (age, weight, height, gender)
- `SentinelInference` - Load and run local MedGemma model
- `CardioPrompt` - Build cardiology-specific prompts with context steering

**Key Functions:**
- `predict(prompt, max_tokens)` - Run inference on local GGUF model
- `build_prompt(subject_info, vitals_summary)` - Create steered prompt
- `summarize_window(vitals_rows)` - Extract vitals statistics
- `parse_vitals_lines(lines)` - Parse text vitals into structured data

**Prompt Steering Example:**
```python
# Activate "Night Sentinel" personality without fine-tuning
prompt = """<system>You are the Night Cardiology Sentinel...
Patient: 65yo male, baseline HR 70 bpm

Current Window (15 min):
HR range: 115-135 bpm
SpO2: 86-92%
RR: 24-28

Provide:
1. Comparison to baseline
2. Anomaly detection
3. Clinical interpretation
</system>"""
```

---

## 🔧 Installation & Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install Ollama locally:**
   ```bash
   # Download from https://ollama.ai
   ollama pull amsaravi/medgemma-4b-it:q6
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Add your Hugging Face token to .env
   export HF_TOKEN="your_token_here"
   ```

5. **Run the client:**
   ```bash
   python medical_mcp_client_2.py
   ```

---

## 🎯 Quick Start

### Run ReAct Loop for Cardiology Scenario
```python
from medical_mcp_client_2 import MCPClient

client = MCPClient()
scenario = """
Patient 402's ECG shows tachycardia:
Time: 00:00 - HR: 115, SPO2: 88, RESP: 24
Time: 00:05 - HR: 125, SPO2: 86, RESP: 28

Analyze and recommend interventions.
"""

result = asyncio.run(client.run_react_loop(scenario, max_iterations=8))
print(result["final_answer"])
```

### Output Example
```
Thought: Patient has significant tachycardia with SpO2 drop...
Action: analyze_cardiology_sentinel
Action Input: {"patient_id": "402", "vitals_text": "..."}
Observation: Tachycardia with mild hypoxemia - requires immediate assessment
Thought: Need to check history for context
Action: get_patient_history
...
Final Answer: 
- Immediate intervention: Assess for arrhythmia
- Monitor SpO2 closely, target >94%
- Consider oxygen supplementation
```

---

## 🛠️ Tool Schemas

### `analyze_cardiology_sentinel`
Analyzes cardiac vitals and generates clinical assessment.

**Input:**
```json
{
  "patient_id": "string (patient identifier)",
  "vitals_text": "string (vitals in text format: 'Time: HH:MM - HR: X, SpO2: Y, RESP: Z')"
}
```

**Output:**
```json
{
  "comparison": "Assessment vs baseline",
  "detection": "Identified anomalies",
  "interpretation": "Clinical assessment"
}
```

---

### `get_patient_vitals`
Retrieves current vital signs for a patient.

**Input:**
```json
{
  "patient_id": "string"
}
```

**Output:**
```json
{
  "patient_id": "string",
  "timestamp": "ISO 8601",
  "vitals": {
    "heart_rate": "number",
    "spo2": "number",
    "respiratory_rate": "number",
    "blood_pressure": "string"
  }
}
```

---

### `get_patient_history`
Retrieves longitudinal patient records for context.

**Input:**
```json
{
  "patient_id": "string",
  "days_back": "number (optional, default 7)"
}
```

**Output:**
```json
{
  "patient_id": "string",
  "visits": [
    {
      "date": "ISO 8601",
      "chief_complaint": "string",
      "diagnoses": ["string"],
      "summary": "string"
    }
  ]
}
```

---

## 🔄 Architecture Diagram

```
┌──────────────────────────────────────┐
│  Streamlit App (app_mcp_cardiology)  │
└────────────┬─────────────────────────┘
             │ User Query
             ▼
┌──────────────────────────────────────┐
│  MCP Client (medical_mcp_client_2)   │
│  - ReAct Loop Controller              │
│  - LLM Prompt Management              │
└────────────┬─────────────────────────┘
             │ Thought/Action/Input
             ▼
┌──────────────────────────────────────┐
│  Ollama/LLM (amsaravi/medgemma-4b)   │
│  - Activated for clinical reasoning   │
└────────────┬─────────────────────────┘
             │ Tool Calls
             ▼
┌──────────────────────────────────────┐
│  MCP Server (medical_mcp_server)      │
│  - Tool Registry                      │
│  - Tool Execution                     │
└────────────┬─────────────────────────┘
             │ Observations
        ┌────┼────┬──────────┐
        ▼    ▼    ▼          ▼
    Cardio Patient Audio  Current
    Analysis  History  Analysis  Time
    (GGUF)    (DB)    (YamNet)  (Clock)
```

---

## 📊 Supported Scenarios

1. **Cardiac Emergency Detection**
   - Tachycardia with hypoxemia
   - Arrhythmia patterns
   - Possible sepsis indicators

2. **Respiratory Distress**
   - Tachypnea with SpO2 drop
   - Accessory muscle use indicators
   - Escalation recommendations

3. **Patient Assessment**
   - Multi-parameter evaluation
   - Baseline comparison
   - Risk stratification

4. **Longitudinal Monitoring**
   - Trend analysis across visits
   - Intervention effectiveness
   - Predictive alerts

---

## 🧪 Testing

```bash
# Run cardiology scenario test
python cardiology_scenario_test.py

# Run tool integration test
python test_cardiology_tool.py
```

---

## 📚 Dependencies

- `fastmcp` - MCP protocol implementation
- `ollama` - Local LLM inference
- `python-dotenv` - Environment configuration
- `llama-cpp-python` - Optional: Local GGUF inference
- `httpx` - Async HTTP client

See [requirements.txt](requirements.txt) for full list.

---

## 🔐 Environment Variables

Create `.env` file with:
```
HF_TOKEN=your_hugging_face_token
OLLAMA_MODEL=amsaravi/medgemma-4b-it:q6
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 🚀 Next Steps

- Integrate with [../night_cardiology_sentinel/](../night_cardiology_sentinel/) for vitals windowing
- Connect to [../reporting_model/](../reporting_model/) for full orchestration
- Add custom tools for facility-specific protocols
- Implement caching for frequently accessed patient data

---

**Last Updated:** February 24, 2026  
**Version:** 2.0
