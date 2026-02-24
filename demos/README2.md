# 🎬 Demos

This folder contains demonstration videos showcasing the main workflows of the MedGemma Sentinel system.

---

## 🩺 MCP + Night Cardiology Sentinel

https://github.com/user-attachments/assets/mcp_app.mp4

This demonstration features two complementary modes:

1. **Standalone Analyzer**: Upload or manually input patient demographics and vital signs (heart rate, respiration rate, blood pressure, etc.). The system analyzes these metrics using a cardiology-calibrated, quantized MedGemma model to provide real-time clinical insights.

2. **MCP Agent Mode**: An autonomous agent leverages the Model Context Protocol (MCP) to dynamically select and invoke specialized medical tools. Based on the patient's condition, it generates priority alerts with detailed clinical descriptions for emergency cases.

---

## 📊 Longitudinal Analysis

https://github.com/user-attachments/assets/MedGemma%20Longitudinal%20Analysis.mp4

This demo illustrates the longitudinal patient tracking workflow:

- Upload multiple medical reports from different time points for the same patient
- The system compares findings across reports using a quantized general-purpose MedGemma model
- Generates a comprehensive longitudinal summary that highlights clinical trends, changes in condition, and disease progression over time
- Provides clinicians with a holistic view of the patient's medical trajectory

---

## 🏥 Intelligent Medical Reporting

https://github.com/user-attachments/assets/Demo_MedGemma_Reporting_Sa%C3%A2d.mp4

This demonstration showcases the end-to-end intelligent report generation pipeline:

- Aggregates patient data from both night shift and day shift monitoring
- Uses the orchestration system (LangGraph + GraphRAG memory) to synthesize cross-shift observations
- Applies medical guardrails to ensure clinical accuracy and safety
- Generates a comprehensive medical report that captures the complete 24-hour patient status, enabling more informed clinical decision-making

---

## 📝 Viewing Notes

- GitHub natively renders `.mp4` video files as playable media players when linked with `https://github.com/user-attachments/assets/` URLs
- Click the play button on each video preview above to watch the demonstrations
- Videos can also be downloaded directly from this repository for offline viewing
