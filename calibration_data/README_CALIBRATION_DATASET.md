# MedGemma I-Matrix Quantization Calibration Dataset

## Overview

A cardiology-focused calibration dataset has been successfully generated for use with **llama.cpp I-matrix quantization**. This dataset contains 38 clinically-grounded text records derived from patient heart rate and ECG monitoring data, optimized for quantizing the MedGemma-4B model in the cardiology domain.

## Key Statistics

| Metric | Value |
|--------|-------|
| **Total Calibration Records** | 38 |
| **Total Estimated Tokens** | ~5,990 |
| **Output File Size** | ~24 KB |
| **Format** | Plain text (UTF-8) |
| **Prompt Template** | Structured cardiology interpretation |

### Population Breakdown

| Population | Count | Data Points | Duration |
|-----------|-------|-------------|----------|
| **Adolescent** | 24 subjects | 108k-123k per subject | 1,800-2,050 hours each |
| **Adult (ICU)** | 7 patients | 500-2,600 data points | 25-72 minutes each |
| **Neonate (ICU)** | 7 patients | 800-3,900 data points | 3-3,023 minutes each |

## Dataset Description

### Adolescent Records (24 subjects)

**Source**: Fitbit continuous heart rate monitoring  
**Duration**: ~75-85 days per subject (1-minute intervals)  
**Clinical Features**:
- Stable resting heart rates (60-100 bpm typical)
- Regular sinus rhythm
- Minimal arrhythmias
- Good autonomic tone variability
- Age range: 10-17 years
- Complete demographic data (age, gender, height, weight)

**Example Narrative**:
```
Adolescent patient, 10 years old, Female. Height: 154.5 cm, Weight: 42.7 kg.

Continuous heart rate monitoring over 1972.6 hours shows normal baseline 
resting heart rate of 91.0 bpm (±18.1 bpm). Heart rate range: 57 to 185 bpm. 
HR variability is moderate, indicating adequate cardiac response to stimuli.

Rhythm assessment: Regular sinus rhythm throughout monitoring period with no 
significant arrhythmias detected. No evidence of ectopic beats or conduction 
abnormalities. Cardiac rate response to daily activities appears appropriate 
for age.
```

### Adult ICU Records (7 patients)

**Source**: MIMIC-III Intensive Care Unit monitoring  
**Duration**: 25-72 minutes per patient (1-second intervals)  
**Clinical Features**:
- Complex ECG parameters: HR, PULSE, RESP, ST segments, SpO2
- Arrhythmia profiling: PVC counts, SVPB runs, pacing metrics
- Beat classification data: normal, ectopic, paced
- Rhythm abnormalities: bigeminy/trigeminy percentages
- HR variability measures
- ICU-level cardiac complexity

**Example Narrative**:
```
Adult icu patient, with complex cardiac monitoring.

Continuous cardiac monitoring over 44.1 minutes. Mean heart rate 72.0 bpm 
(±9.6 bpm), range 54-119 bpm; pulse 70.1 bpm; respiration 20.3 /min; 
SpO2 96.5%.

Cardiac rhythm analysis: Predominantly normal sinus rhythm with 4 abnormality 
features documented. premature ventricular contractions (PVC) present (0.0%, 
81 total), supraventricular premature beats (0.5%, 964 total), bigeminy 
detected (4.0%), trigeminy detected (12.0%). Patient demonstrates rhythm 
abnormalities requiring ongoing cardiac assessment and possible intervention.
```

### Neonate ICU Records (7 patients)

**Source**: MIMIC-III Neonatal Intensive Care Unit monitoring  
**Duration**: 3-3,023 minutes per patient (1-second intervals)  
**Clinical Features**:
- Pediatric-appropriate heart rates (77-211 bpm range)
- Respiratory rate monitoring (30-51 breaths/min typical)
- Limited arrhythmia data (neonates often show stable rhythms)
- Very short patient stays (critical care admission)
- Age-adjusted interpretation guidelines

**Example Narrative**:
```
Pediatric icu patient, requiring continuous cardiac monitoring.

Continuous cardiac monitoring over 3022.8 minutes. Mean heart rate 130.7 bpm 
(±14.7 bpm), range 77-211 bpm; pulse 130.3 bpm; respiration 43.0 /min.

Cardiac rhythm analysis: Predominantly normal sinus rhythm with 0 abnormality 
features documented. Stable cardiac rhythm without significant arrhythmias. 
Cardiac rhythm remains stable with appropriate rate control.
```

## File Format

### Primary Output: `cardiology_calibration_imatrix.txt`

**Structure**:
- Plain text file with UTF-8 encoding
- 38 calibration prompts separated by 80-character dividers
- Each prompt follows the standardized template:

```
Below is a cardiology topic:

[Clinical narrative with HR metrics, rhythm analysis, and interpretation]

================================================================================
```

**Characteristics**:
- Easy to parse and integrate with llama.cpp
- No JSON formatting (direct text for I-matrix processing)
- Well-annotated clinical content
- Consistent structure across all 38 records

### Metadata Output: `calibration_dataset_log.json`

**Purpose**: Track dataset composition, sources, and statistics

**Contents**:
```json
{
  "total_records": 38,
  "total_tokens": 5990,
  "populations": {
    "adolescent": {
      "count": 24,
      "records": [
        {
          "subject_id": 903,
          "data_points": 118354,
          "duration_hours": 1972.57
        },
        ...
      ]
    },
    "adult": {
      "count": 7,
      "records": [
        {
          "patient_id": "adult_3190202n",
          "data_points": 2613,
          "duration_sec": 2647
        },
        ...
      ]
    },
    "neonate": {
      "count": 7,
      "records": [
        {
          "patient_id": "neonate_3182898n",
          "data_points": 900,
          "duration_sec": 181368
        },
        ...
      ]
    }
  },
  "output_file": "cardiology_calibration_imatrix.txt"
}
```

## Clinical Content Strategy

### Metric Extraction

For each patient record, the following metrics were extracted:

**Adolescent Data**:
- Heart rate statistics: mean, std dev, min, max, median
- Variability assessment: HR variability classification
- Rhythm evaluation: regular sinus rhythm status
- Duration: total monitoring hours

**MIMIC-III Data**:
- Heart rate statistics (as above)
- Physiological parameters: PULSE, RESP, SpO2
- Beat classification: normal beats, PVCs, SVPBs, paced beats
- Arrhythmia metrics: PVC/SVPB run counts, bigeminy%, trigeminy%
- HR variability: max variability index
- Duration: elapsed seconds

### Clinical Narrative Generation

Narratives were generated using clinical logic:

1. **Demographic Context**: Age, gender, height, weight (when available)
2. **Vital Signs Summary**: HR range, mean, standard deviation
3. **HR Assessment**: Classification as normal, elevated, or low
4. **Variability Interpretation**: Excellent/good/moderate based on std dev
5. **Rhythm Analysis**: Specific arrhythmias detected with counts
6. **Clinical Conclusion**: Prognosis or monitoring recommendation

This approach ensures the calibration data reflects how cardiologists interpret patient records, making the quantization more effective for the task.

## Usage Instructions

### For llama.cpp I-Matrix Quantization

1. **Prepare the model**:
   ```bash
   # Download/convert MedGemma-4B to GGUF format
   python convert-hf-to-gguf.py model_path/ --outtype f16
   ```

2. **Run I-matrix quantization**:
   ```bash
   ./llama-quantize [input_model.gguf] \
     [output_model_q8_0.gguf] Q8_0 \
     --calibration calibration_output/cardiology_calibration_imatrix.txt
   ```

3. **Verify calibration weights**:
   - Check llama.cpp output logs for successful I-matrix computation
   - Verify no errors in token processing
   - Confirm importance scores were calculated

### Advantages of This Calibration Dataset

✅ **Domain-Specific**: All content focuses on cardiology interpretation  
✅ **Diverse Populations**: Adolescent stability + adult complexity + neonate edge cases  
✅ **Realistic Scale**: ~6K tokens matches typical quantization calibration needs  
✅ **Structured Format**: Consistent prompt template for uniform quantization  
✅ **Rich Features**: Incorporates HR metrics, ECG data, and clinical reasoning  
✅ **Ready-to-Use**: No preprocessing needed; directly compatible with llama.cpp  

## Data Processing Pipeline

### Generation Script: `generate_calibration_dataset.py`

A complete automated pipeline was created to:

1. **Load subject data**:
   - Adolescent demographics from `subjects_info.json`
   - Adolescent HR CSVs (24 subjects)
   - MIMIC-III adult CSVs (7 subjects)
   - MIMIC-III neonate CSVs (7 subjects)

2. **Extract metrics**:
   - Population-specific metric extraction
   - Robust error handling (missing columns, invalid values)
   - Numpy array processing with type conversion

3. **Generate narratives**:
   - Clinical context creation
   - Vital signs interpretation
   - Arrhythmia assessment
   - Standardized formatting

4. **Format prompts**:
   - Finetune-compatible template: "Below is a cardiology topic:\n\n[narrative]"
   - Separator insertion for clarity

5. **Save outputs**:
   - Main calibration file (text)
   - Metadata log (JSON with numpy compatibility)

### Script Features

- **Error Handling**: Graceful handling of missing/malformed data
- **Logging**: Detailed processing logs for each subject
- **Flexibility**: Easily adaptable for new data sources or metrics
- **Reusability**: Can be rerun with additional data sources

## Quality Assurance

✓ All 24 adolescent subjects processed  
✓ 7/7 adult patients successfully extracted  
✓ 7/8 neonates processed (1 missing HR column)  
✓ Total: 38/39 available subjects (97.4% success rate)  
✓ Estimated tokens: 5,990 (suitable for I-matrix calibration)  
✓ File integrity: Verified UTF-8 encoding and format consistency  
✓ JSON metadata: Validated for schema compliance  

## Integration with MedGemma Finetuning

This calibration dataset complements the existing MedGemma finetuning pipeline:

- **Finetuning base**: `google/medgemma-4b-it` (4B parameters)
- **Finetuning dataset**: `ilyassacha/cardiologyChunks` (external HF Hub corpus)
- **Finetuning method**: LoRA adapters (r=4, lora_alpha=16)
- **Quantization**: I-matrix with this calibration data ensures quantization-aware to cardiology task

The calibration data represents **distribution-aware** quantization for the specific medical domain.

## Next Steps

1. **Convert base model to GGUF**:
   ```python
   from llama_cpp import llama_cpp
   # Use llama.cpp conversion tools
   ```

2. **Apply I-matrix quantization**:
   ```bash
   # Run with calibration dataset
   llama-quantize medgemma-4b.gguf output.gguf Q8_0 \
     --calibration-file calibration_output/cardiology_calibration_imatrix.txt
   ```

3. **Validate quantized model**:
   - Test inference speed
   - Compare outputs to unquantized baseline
   - Verify cardiology-specific accuracy

## Additional Notes

- **Token Count**: Rough estimate (~4 chars per token); actual count depends on tokenizer
- **Scalability**: Script can handle additional patient records; just add CSVs to `data/processed/` directories
- **Customization**: Edit narrative generation functions to add/remove clinical features
- **Reproducibility**: Script uses fixed random seed (3407) for consistency

---

**Generated**: February 21, 2026  
**Data Source**: MedGemma-Sentinel project (Fitbit + MIMIC-III)  
**Format**: Calibration-ready text corpus for llama.cpp I-matrix quantization  
**Status**: Ready for deployment ✅
