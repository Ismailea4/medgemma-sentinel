# 🎬 MedGemma Sentinel — Demo Videos

Comprehensive video demonstrations of the MedGemma Sentinel clinical AI system, showcasing autonomous surveillance, intelligent reporting, and longitudinal patient analysis workflows.

---

## 🏥 Complete System Demos

| Video | Feature | Description |
|-------|---------|-------------|
| [MedGemma_Sentinel_Full_Demo.mov](MedGemma_Sentinel_Full_Demo.mov) | **Full Night-to-Day Workflow** | End-to-end demonstration: Night Mode autonomous surveillance detects a critical respiratory event (SpO2 desaturation, tachycardia, audio distress), saves it to patient memory, then Day Mode loads the night context, accepts the doctor's morning assessment, and generates an AI-powered differential diagnosis correlating night and day findings. |
| [mcp_app.mp4](mcp_app.mp4) | **🩺 MCP + Night Cardiology Sentinel** | Demonstrates two complementary modes: **(1) Standalone Analyzer** – Upload or manually input patient demographics and vital signs (heart rate, respiration rate, blood pressure, etc.). The system analyzes these metrics using a cardiology-calibrated, quantized MedGemma model to provide real-time clinical insights. **(2) MCP Agent Mode** – An autonomous agent leverages the Model Context Protocol (MCP) to dynamically select and invoke specialized medical tools. Based on the patient's condition, it generates priority alerts with detailed clinical descriptions for emergency cases. |
| [MedGemma Longitudinal Analysis.mp4](MedGemma%20Longitudinal%20Analysis.mp4) | **📊 Longitudinal Patient Tracking** | Illustrates the longitudinal analysis workflow: Upload multiple medical reports from different time points for the same patient. The system compares findings across reports using a quantized general-purpose MedGemma model, generates a comprehensive longitudinal summary that highlights clinical trends, changes in condition, and disease progression over time, providing clinicians with a holistic view of the patient's medical trajectory. |
| [Demo_MedGemma_Reporting_Saâd.mp4](Demo_MedGemma_Reporting_Sa%C3%A2d.mp4) | **🏥 Intelligent Medical Reporting** | Showcases the end-to-end intelligent report generation pipeline: Aggregates patient data from both night shift and day shift monitoring, uses the orchestration system (LangGraph + GraphRAG memory) to synthesize cross-shift observations, applies medical guardrails to ensure clinical accuracy and safety, and generates a comprehensive medical report that captures the complete 24-hour patient status, enabling more informed clinical decision-making. |

---

## 📄 Generated Clinical Report Demos

| Video | Report Document | Description |
|-------|-----------------|-------------|
| [RAP1_SBAR_Critical_Incident_Report.mp4](reports/RAP1_SBAR_Critical_Incident_Report.mp4) | **Document A — SBAR Critical Incident Report** | PDF generated during Night Mode when a critical event is detected. Includes patient vitals table with red-highlighted critical values, NEWS2 clinical score, structured SBAR narrative (Situation, Background, Assessment, Recommendation), and actionable recommendation checklist. |
| [Shift_Handover_Report.mp4](reports/Shift_Handover_Report.mp4) | **Document B — Shift Handover Summary** | PDF generated at the Night-to-Day transition. Contains a hospital-style flow sheet table (Time / Event Type / Key Vitals) with zebra-striped rows, clinical impression summary, overnight pattern analysis, and prioritized recommendations for the incoming day team. |
| [RAP2_Differential_Diagnosis_Report.mp4](reports/RAP2_Differential_Diagnosis_Report.mp4) | **Document C — RAP2 Differential Diagnosis** | PDF generated in Day Mode after the doctor enters morning exam findings. Features the doctor's clinical input in a boxed section, top 3 differential diagnoses ranked by probability with reasoning bullets, red-highlighted recommended investigations, and a correlated night surveillance context section linking overnight events to each diagnosis. |

---

## 🏗️ System Architecture

```
Night Mode (Autonomous)          Day Mode (Doctor-Assisted)
┌─────────────────────┐          ┌─────────────────────────┐
│ Vitals Monitoring    │          │ Night Event Timeline     │
│ Audio Surveillance   │──save──▶│ Doctor's Clinical Input  │
│ AI SBAR Analysis     │  events │ AI Differential Diagnosis│
│ Document A (PDF)     │         │ Document B & C (PDF)     │
└─────────────────────┘          └─────────────────────────┘
```

## 🛠️ Tech Stack

- **AI Engine:** MedGemma 2 (4B params, Q4_K_M quantized)
- **UI Framework:** Streamlit with custom CSS dashboard
- **PDF Generation:** ReportLab with professional medical templates
- **Memory:** File-based patient event storage (GraphRAG-ready)
- **Orchestration:** MCP (Model Context Protocol) + LangGraph state machine
- **Clinical Safety:** NeMo Guardrails for medical accuracy validation

---

## 📝 Viewing Notes

- All videos are available for direct viewing and download from this repository
- GitHub and most markdown renderers support inline video playback for `.mp4` and `.mov` files
- For optimal viewing experience, download videos if streaming is slow
