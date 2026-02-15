# MedGemma Sentinel: The Autonomous Edge-AI Guardian 🏥🛡️

**MedGemma Sentinel** is a multimodal AI agent designed for offline rural clinics (e.g., the Atlas Mountains) to bridge the "Vigilance Gap" in healthcare.  It operates on low-cost hardware and switches between two specialized roles to support medical staff 24/7.

## 🌟 Key Features

* **The Night Sentinel:** Autonomous monitoring for falls (YOLOv10), distress, and bio-acoustic anomalies like coughing or wheezing (YamNet).
* **The Day Assistant:** Specialist diagnostic tool for imaging analysis (MedSigLIP) and automated doctor-patient consultation transcription (Faster-Whisper).
* **Activation Steering:** Instantly shifts the model's "personality" from safety watchdog to medical specialist without fine-tuning.
* **GraphRAG Integration:** Uses LlamaIndex and LangGraph to manage longitudinal patient history and answer complex clinical evolution questions.


## 🏗️ System Architecture

MedGemma Sentinel is a **Hierarchical Agent** managed by a State Machine:

1.  **Orchestration:** LangGraph manages the flow between the "Reflex Layer" (real-time sensing) and the "Cognitive Layer" (clinical reasoning).
2.  **Inference Engine:** Google MedGemma 2 (4B) optimized via GGUF and I-Matrix Quantization to run on 8GB RAM.
3.  **Memory:** A local, file-based storage system using GraphRAG to link daily reports and vital trends.

## 📋 Technical Stack

* **LLM:** MedGemma 2 (4B)
* **Vision:** YOLOv10-Nano (Reflex), MedSigLIP (Analytical)
* **Audio:** YamNet (Event Detection), Faster-Whisper (ASR), HeAR
* **Frameworks:** LangGraph, LlamaIndex, Streamlit, llama.cpp

## 👥 The "Special Forces" Team

* **Infrastructure:** Hamza — Model quantization and API deployment.
* **Steering:** Ismail — Activation engineering and vector injection.
* **Audio:** Youssra — Event detection and speech-to-text.
* **Vision & UI:** Saad/Othman — Fall detection and Streamlit dashboard.
* **Memory & Scribe:** Saad/Othman — LangGraph orchestration and PDF reporting.
## 🚀 Deployment

Designed for the **Kaggle MedGemma Impact Challenge**. Deadline: February 24, 2026.