Here are the steps I followed for I-matrix quantization using a single text calibration file.

## 0. Prerequisites

- Windows PowerShell
- Python 3.10+
- 30+ GB free disk space
- `medgemma_ready_calibration.txt` (your calibration dataset)

Recommended working folder (example):

```
C:\MedGemma_Sentinel\
```

Inside this folder you will place:

- `medgemma-4b-it` (raw HF model)
- `medgemma-4b-it-fp16.gguf`
- `medgemma_ready_calibration.txt`
- `llama_bin\` (llama.cpp executables)

## 1. Download the raw model (MedGemma)

### 1. Log in to Hugging Face

```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

### 2. Download the raw MedGemma model

```powershell
huggingface-cli download google/medgemma-4b-it --local-dir .\medgemma-4b-it
```

### 3. Get the conversion script

```powershell
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r .\llama.cpp\requirements.txt
```

### 4. Convert to uncompressed GGUF (FP16)

```powershell
python .\llama.cpp\convert_hf_to_gguf.py .\medgemma-4b-it --outtype f16 --outfile medgemma-4b-it-fp16.gguf
```

## 2. Download `llama_bin`

### 1. Download the pre-compiled Windows executables

1. Go to the official `llama.cpp` releases page in your web browser:
   https://github.com/ggerganov/llama.cpp/releases
2. Scroll down to the Assets section of the latest release.
3. You need to download a `.zip` file. Which one you choose depends on your laptop:
   - If you have an Nvidia GPU (RTX, GTX): download the file containing `win-cuda-cu12.2-x64.zip` (or similar CUDA version).
   - If you only have a CPU or standard graphics: download the file containing `win-avx2-x64.zip`.

### 2. Extract and move the files

1. Open the `.zip` file you just downloaded.
2. Select all files and paste them into a folder named `llama_bin`. The most important files are:
   - `llama-imatrix.exe` (or `imatrix.exe` in some older versions)
   - `llama-quantize.exe` (or `quantize.exe`)
   - `llama-cli.exe` (or `main.exe`)
3. Place the `llama_bin` folder next to your `medgemma-4b-it-fp16.gguf` and `medgemma_ready_calibration.txt`.

### 3. Create the I-matrix file

```powershell
.\llama_bin\llama-imatrix.exe -m medgemma-4b-it-fp16.gguf -f medgemma_ready_calibration.txt -o night_sentinel_imatrix.dat --chunks 100
```

## 3. Execute the quantization

### Run quantization

```powershell
.\llama_bin\llama-quantize.exe --imatrix night_sentinel_imatrix.dat medgemma-4b-it-fp16.gguf medgemma-night-sentinel-Q4_K_M.gguf Q4_K_M
```

## 4. Test the model

### Test with llama.cpp CLI

```powershell
.\llama_bin\llama-cli.exe -m medgemma-night-sentinel-Q4_K_M.gguf -c 2048 -n 256 -p "<start_of_turn>user`n[NIGHT SENTINEL SYSTEM]`nAnalyze the following continuous cardiac monitoring data:`nPatient baseline HR is 70 bpm. Current HR is 135 bpm with irregular rhythm. SpO2 is 92%. What is your assessment?`n<end_of_turn>`n<start_of_turn>model`n"
```

## Notes

- If `llama-imatrix.exe` is missing, check you downloaded a release that includes I-matrix support.
- The calibration file can be a single text file with multiple samples (one after another).
- Larger calibration files improve quality but take longer to process.
