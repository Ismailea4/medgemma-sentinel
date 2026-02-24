# MedGemma Sentinel: The Autonomous Edge-AI Guardian 🏥🛡️

**MedGemma Sentinel** is a multimodal AI agent designed for offline rural clinics (e.g., the Atlas Mountains) to bridge the "Vigilance Gap" in healthcare. It operates on low-cost hardware and switches between two specialized roles to support medical staff 24/7.

## 🌟 Key Features

- **The Night Sentinel:** Autonomous monitoring for falls (YOLOv10), distress, and bio-acoustic anomalies like coughing or wheezing (YamNet).
- **The Day Assistant:** Specialist diagnostic tool for imaging analysis (MedSigLIP) and automated doctor-patient consultation transcription (Faster-Whisper).
- **Activation Steering:** Instantly shifts the model's "personality" from safety watchdog to medical specialist without fine-tuning.
- **GraphRAG Integration:** Uses LlamaIndex and LangGraph to manage longitudinal patient history and answer complex clinical evolution questions.

## 📋 Tech Stack

- **LLM:** MedGemma 2 (4B)
- **Frameworks:** LangGraph, LlamaIndex, Streamlit, llama.cpp
- **Vision & Audio:** YOLOv10, YamNet, Faster-Whisper, MedSigLIP

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**

### System Dependencies (for PDF generation)

#### macOS
```bash
brew install cairo pango gdk-pixbuf libffi
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install python3-dev libcairo2-dev libpango1.0-dev libpangoft2-1.0-0
```

#### Windows
Most dependencies are included. If you encounter issues with `weasyprint`, install GTK+ from [here](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer).

### Installation

1. **Clone the repository:**
   ```bash
   git clone -b saad-reports-generation https://github.com/Ismailea4/medgemma-sentinel.git
   cd medgemma-sentinel
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-ui.txt
   ```

   Or use the setup script (macOS/Linux):
   ```bash
   chmod +x setup.sh && ./setup.sh
   ```

3. **Run the Streamlit UI:**
   ```bash
   streamlit run ui/app.py
   ```
   The dashboard opens at `http://localhost:8501`

### Running the UI

```bash
streamlit run ui/app.py
```

The dashboard will open at `http://localhost:8501`

### Running Tests

```bash
pytest tests/ -v
```

---

## 📖 Project Structure

```
src/
├── orchestration/      # LangGraph state machine & workflows
├── memory/             # Patient graphs & storage
├── reporting/          # Report generation & clinical analysis
└── models/             # Data models (Patient, Vitals, Events)

ui/                     # Streamlit application
simulators/             # Data generators for testing
examples/               # Demo workflows
tests/                  # Unit & integration tests
```

---

## 👥 Team

- **Infrastructure:** Hamza — Model quantization and API deployment.
- **Steering:** Ismail — Activation engineering and vector injection.
- **Audio:** Youssra — Event detection and speech-to-text.
- **Vision & UI:** Saad/Othman — Fall detection and Streamlit dashboard.
- **Memory & Scribe:** Saad/Othman — LangGraph orchestration and PDF reporting.

---

## 📋 Deployment

Designed for the **Kaggle MedGemma Impact Challenge**. Deadline: February 24, 2026.
