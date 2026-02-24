# MedGemma Sentinel — Demo Videos

Video walkthroughs of the MedGemma Sentinel clinical AI system, demonstrating the complete Night-to-Day autonomous surveillance workflow and each generated clinical report.

---

## Full Workflow Demo

| Video | Description |
|-------|-------------|
| [MedGemma_Sentinel_Full_Demo.mov](MedGemma_Sentinel_Full_Demo.mov) | End-to-end demonstration: Night Mode autonomous surveillance detects a critical respiratory event (SpO2 desaturation, tachycardia, audio distress), saves it to patient memory, then Day Mode loads the night context, accepts the doctor's morning assessment, and generates an AI-powered differential diagnosis correlating night and day findings. |

---

## Individual Report Demos

| Video | Report | Description |
|-------|--------|-------------|
| [RAP1_SBAR_Critical_Incident_Report.mp4](reports/RAP1_SBAR_Critical_Incident_Report.mp4) | **Document A — SBAR Critical Incident Report** | PDF generated during Night Mode when a critical event is detected. Includes patient vitals table with red-highlighted critical values, NEWS2 clinical score, structured SBAR narrative (Situation, Background, Assessment, Recommendation), and actionable recommendation checklist. |
| [Shift_Handover_Report.mp4](reports/Shift_Handover_Report.mp4) | **Document B — Shift Handover Summary** | PDF generated at the Night-to-Day transition. Contains a hospital-style flow sheet table (Time / Event Type / Key Vitals) with zebra-striped rows, clinical impression summary, overnight pattern analysis, and prioritized recommendations for the incoming day team. |
| [RAP2_Differential_Diagnosis_Report.mp4](reports/RAP2_Differential_Diagnosis_Report.mp4) | **Document C — RAP2 Differential Diagnosis** | PDF generated in Day Mode after the doctor enters morning exam findings. Features the doctor's clinical input in a boxed section, top 3 differential diagnoses ranked by probability with reasoning bullets, red-highlighted recommended investigations, and a correlated night surveillance context section linking overnight events to each diagnosis. |

---

## System Architecture

```
Night Mode (Autonomous)          Day Mode (Doctor-Assisted)
┌─────────────────────┐          ┌─────────────────────────┐
│ Vitals Monitoring    │          │ Night Event Timeline     │
│ Audio Surveillance   │──save──▶│ Doctor's Clinical Input  │
│ AI SBAR Analysis     │  events │ AI Differential Diagnosis│
│ Document A (PDF)     │         │ Document B & C (PDF)     │
└─────────────────────┘          └─────────────────────────┘
```

## Tech Stack

- **AI Engine:** MedGemma 2 (4B params, Q4_K_M quantized)
- **UI Framework:** Streamlit with custom CSS dashboard
- **PDF Generation:** ReportLab with professional medical templates
- **Memory:** File-based patient event storage (GraphRAG-ready)
