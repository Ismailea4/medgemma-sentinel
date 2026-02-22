# Night Cardiology Sentinel - Streamlit App

Streamlit application for running clinical anomaly detection using the MedGemma Night Sentinel model.

## Features

- **Subject Selection**: Upload `subjects_info.json` or enter patient details manually
- **Vitals Upload**: Parse heart rate and vital signs from text files
- **Windowing**: Process vitals in 15-minute time windows or 10-row chunks
- **Model Inference**: Run the quantized MedGemma model (local GGUF or Hugging Face)
- **Clinical Insights**: Get structured analysis with comparison, detection, and interpretation

## Project Structure

```
medgemma-sentinel/
├── app_night_cardiology_sentinal.py    # Main Streamlit app (run this)
├── src/
│   └── night_cardiology_sentinel/
│       ├── __init__.py
│       ├── data_parser.py              # Vitals parsing and windowing
│       └── inference.py                # Model loading and inference
├── models/
│   └── medgemma-night-sentinel-Q4_K_M.gguf
└── data/
    └── processed/
        └── hr_adolescent/
            └── subjects_info.json
```

## Usage

1. **Install dependencies**:
   ```bash
   pip install streamlit llama-cpp-python huggingface-hub
   ```

2. **Run the app**:
   ```bash
   streamlit run app_night_cardiology_sentinal.py
   ```

3. **In the app**:
   - Upload `subjects_info.json` or enter patient details
   - Upload a vitals `.txt` file
   - Choose windowing mode (15-minute or 10-row)
   - Select model source (local or Hugging Face)
   - Click "Run analysis"

## Supported Vitals Formats

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
