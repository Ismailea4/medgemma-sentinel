# Longitudinal Source - Multi-Report Clinical Evolution Analysis 📊

This module analyzes multiple patient medical reports over time to detect trends, track symptom/diagnosis evolution, and generate comprehensive longitudinal clinical assessments.

## 📋 Overview

The Longitudinal Source module enables:
1. **Report Extraction:** Parse structured medical data from PDF documents
2. **Evolution Tracking:** Monitor how symptoms, diagnoses, and severity change over time
3. **Trend Analysis:** Identify patterns and clinical progression
4. **Report Generation:** Create comprehensive evolution reports with recommendations and visualizations
5. **Risk Assessment:** Determine current risk level based on longitudinal patterns

## 📁 Core Files

### `longitudinal_analysis.py`
Main module containing all longitudinal analysis functions.

**Key Classes:**
- `SEVERITY_MAP` - Maps severity strings to numeric values for comparison
- `SEVERITY_COLORS` - Color coding for visualization (red=high, orange=medium, green=low)

**Key Functions:**

#### `extract_medical_data(pdf_path)`
Extracts structured medical data from PDF reports.

**Input:** PDF file path
**Output:** Dictionary with:
```python
{
    "patient_id": "str",
    "patient_name": "str", 
    "date": "DD/MM/YYYY",
    "time": "HH:MM",
    "room": "str",
    "main_complaint": "str",
    "symptoms": ["str"],
    "vital_signs": {"name": "value"},
    "physical_exam": {"finding": "value"},
    "diagnoses": ["str"],
    "severity": "Elevée/Moyenne/Basse",
    "management_plan": ["str"],
    "critical_alerts": bool,
    "raw_text": "str",
    "full_date": datetime,
    "filename": "str"
}
```

**Regular Expressions Used:**
- Patient name: `Patient:\s*([^\n]+)`
- Date: `Date:\s*(\d{2}/\d{2}/\d{4})`
- Symptoms: `Symptômes associés:\s*([\s\S]*?)(?:■|Examen|...)`
- Diagnoses: `Diagnostics différentiels:\s*([\s\S]*?)(?:Gravité|Plan|...)`
- Severity: `Gravité évaluée:\s*([^\n]+)`

---

#### `analyze_longitudinal_evolution(patient_reports, llm_model, analysis_type)`
Analyzes patient evolution using MedGemma.

**Parameters:**
- `patient_reports` - List of extracted report dicts
- `llm_model` - Loaded LLM model (from llama-cpp-python)
- `analysis_type` - "comparison" (2 reports) or "time_series" (multiple)

**Returns:** Dictionary with:
```python
{
    "summary": "Overall evolution summary",
    "severity_evolution": "Changes in severity",
    "symptom_changes": {
        "new_symptoms": ["str"],
        "resolved_symptoms": ["str"],
        "persistent_symptoms": ["str"],
        "worsened_symptoms": ["str"]
    },
    "diagnosis_evolution": {
        "new_diagnoses": ["str"],
        "excluded_diagnoses": ["str"],
        "prioritized_diagnoses": ["str"],
        "analysis": "str"
    },
    "management_changes": {
        "new_interventions": ["str"],
        "removed_interventions": ["str"],
        "modified_interventions": ["str"],
        "justification": "str"
    },
    "critical_alerts": ["str"],
    "clinical_recommendations": ["str"],
    "prognosis": "str",
    "monitoring_parameters": ["str"],
    "overall_assessment": "str",
    "risk_level": "Élevé/Moyen/Faible"
}
```

**LLM Prompt Template (for comparison):**
```
You are an expert medical analyst.
Compare these two patient consultations:

Consultation 1 (date1): symptoms, diagnoses, severity
Consultation 2 (date2): symptoms, diagnoses, severity

Provide JSON analysis with:
- symptom_changes (new/resolved/persistent/worsened)
- diagnosis_evolution (new/excluded/prioritized)
- severity_evolution
- clinical_recommendations
- risk_level
```

---

#### `generate_fallback_longitudinal_analysis(patient_reports, analysis_type)`
Provides hardcoded analysis if LLM fails.

**Logic:**
- Compares severity values numerically for trend
- Uses set operations to detect symptom changes: `new = current - previous`
- Flags high-severity dates for alert detection
- Returns structured dict matching LLM output format

---

#### `create_longitudinal_visualizations(patient_reports, analysis)`
Generates interactive Plotly charts for evolution trends.

**Visualizations (2x2 Grid):**
1. **Severity Over Time** - Line chart with severity zones (red/orange/green)
2. **Symptom Count** - Bar chart showing symptom count per date
3. **Diagnosis Count** - Bar chart showing diagnosis count per date
4. **Overall Trend Gauge** - Indicator showing improvement/stable/deterioration

**Return:** Plotly figure object

---

#### `generate_enhanced_report_with_longitudinal_analysis(original_report, longitudinal_analysis, previous_reports, output_path)`
Creates comprehensive PDF report with evolution analysis.

**Structure:**
1. Patient information table
2. Original report details
3. Longitudinal analysis section:
   - Summary and overall assessment
   - Severity evolution with color coding
   - Symptom changes with formatting
   - Diagnosis evolution
   - Management modifications
   - Critical alerts (warning box if present)
   - Clinical recommendations (bulleted list)
   - Prognosis and risk level
4. Footer with report metadata

**Technology:** ReportLab for PDF generation with styled tables

---

## 🔄 Data Processing Pipeline

```
PDF File 1    PDF File 2    ... PDF File N
     │            │              │
     └────────────┴──────────────┘
            │
            ▼
    extract_medical_data()
            │
            ▼
    [Structured Data Dicts]
            │
            ▼
    analyze_longitudinal_evolution()
    (with LLM or fallback)
            │
            ▼
    [Evolution Analysis Dict]
            │
            ├─→ create_longitudinal_visualizations()
            │   └─→ Plotly Charts
            │
            └─→ generate_enhanced_report_with_longitudinal_analysis()
                └─→ PDF Report
```

---

## 🎯 Analysis Modes

### Comparison Mode (2 Reports)
Compares baseline vs current state.

**Output Focus:**
- Changes from previous visit
- New vs resolved findings
- Intervention effectiveness
- Prognosis based on progression

**Use Case:** Follow-up visit assessment, effectiveness monitoring

---

### Time Series Mode (3+ Reports)
Analyzes trends across multiple dates.

**Output Focus:**
- Temporal patterns and milestones
- Critical periods identification
- Trend prediction
- Long-term management recommendations

**Use Case:** Chronic disease monitoring, longitudinal outcomes

---

## 📊 Severity Mapping

```python
SEVERITY_MAP = {
    "Elevee": 3,      "High": 3,        # Red zone
    "Moyenne": 2,     "Medium": 2,      # Orange zone
    "Basse": 1,       "Low": 1,         # Green zone
    "None": 0,        "": 0             # Baseline
}
```

**Trend Rules:**
- Severity increase → Deterioration (⬆️ RED)
- Severity decrease → Improvement (⬇️ GREEN)
- Severity stable → Monitoring (➡️ ORANGE)

---

## 🔐 PDF Format Assumptions

The module expects PDF with sections marked by:
- `Patient:` - Patient name
- `ID:` - Patient identifier
- `Date:` - Consultation date
- `Chambre:` - Patient room/location
- `Plainte principale:` - Chief complaint
- `Symptômes associés:` - Associated symptoms
- `Constantes:` - Vital signs (constantes)
- `Examen physique:` - Physical examination
- `Diagnostics différentiels:` - Differential diagnoses
- `Gravité évaluée:` - Severity assessment
- `Plan de Prise en Charge` - Management plan

---

## 🎨 Visualization Features

**Severity Chart:**
- Zone coloring: Red (2.0-3.5), Orange (1-2), Green (0-1)
- Text annotations showing exact values
- Marker size proportional to importance

**Trend Gauge:**
- Green arc (66-100) = Improvement
- Orange arc (33-66) = Stable
- Red arc (0-33) = Deterioration
- Current value needle shows status

---

## 🔧 Integration with Streamlit

The module is designed for use in `app_longitudinal_analysis.py`:

```python
# In Streamlit app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src" / "longitudinal_src"))

from longitudinal_analysis import (
    extract_medical_data,
    analyze_longitudinal_evolution,
    create_longitudinal_visualizations,
    generate_enhanced_report_with_longitudinal_analysis,
    load_medgemma_model
)

# Upload PDFs → Extract → Analyze → Visualize → Report
```

---

## 📦 Dependencies

- `pdfplumber>=0.10.3` - PDF text extraction
- `pandas>=2.2.0` - Data manipulation
- `numpy>=2.0.0` - Numerical operations
- `plotly>=5.18.0` - Interactive visualizations
- `reportlab>=4.4.0` - PDF generation
- `matplotlib>=3.9.0` - Additional plotting
- `llama-cpp-python>=0.3.0` - Optional: Local LLM inference
- `regex`, `datetime` - Standard library

---

## 🧪 Example Usage

### Basic Analysis
```python
from longitudinal_analysis import (
    extract_medical_data,
    analyze_longitudinal_evolution,
    create_longitudinal_visualizations
)

# Extract data
report1 = extract_medical_data("consultation1.pdf")
report2 = extract_medical_data("consultation2.pdf")

# Analyze evolution
analysis = analyze_longitudinal_evolution(
    [report1, report2],
    llm_model=None,  # Use fallback
    analysis_type="comparison"
)

# Display results
print(f"Assessment: {analysis['overall_assessment']}")
print(f"Risk Level: {analysis['risk_level']}")
print(f"New Symptoms: {analysis['symptom_changes']['new_symptoms']}")

# Create visualization
fig = create_longitudinal_visualizations([report1, report2], analysis)
fig.show()
```

### With LLM-Based Analysis
```python
from longitudinal_analysis import load_medgemma_model

# Load model
llm = load_medgemma_model("./models/medgemma-4b-it-fp16.gguf")

# Analyze with LLM
analysis = analyze_longitudinal_evolution(
    [report1, report2],
    llm_model=llm,
    analysis_type="comparison"
)

# Generate PDF
output_file = generate_enhanced_report_with_longitudinal_analysis(
    original_report=report2,
    longitudinal_analysis=analysis,
    previous_reports=[report1],
    output_path="patient_evolution_report.pdf"
)
```

---

## 🚀 Output Example

**Structured Analysis Output:**
```json
{
  "summary": "Patient shows overall improvement with reduction in symptoms",
  "severity_evolution": "Trend: amélioration (Gravité: 3 → 1)",
  "symptom_changes": {
    "new_symptoms": ["mild cough"],
    "resolved_symptoms": ["chest pain", "dyspnea"],
    "persistent_symptoms": ["fatigue"],
    "worsened_symptoms": []
  },
  "diagnosis_evolution": {
    "new_diagnoses": [],
    "excluded_diagnoses": ["acute MI"],
    "prioritized_diagnoses": ["stable angina", "hypertension"],
    "analysis": "Cardiac workup negative, diagnosis narrowed"
  },
  "clinical_recommendations": [
    "Continue current medications",
    "Reduce statin dose",
    "Schedule 2-week follow-up",
    "Monitor for recurrent symptoms"
  ],
  "risk_level": "Faible"
}
```

---

## 📚 Technical Notes

- **Temporal Sorting:** Uses `full_date` (datetime object) for accurate ordering
- **Set Operations:** For symptom/diagnosis comparison (case-insensitive)
- **Fallback System:** Always provides output even if LLM fails
- **French Support:** Full support for French medical terminology
- **Edge-Safe:** Works with or without LLM (fallback built-in)

---

**Last Updated:** February 24, 2026  
**Version:** 1.2
