"""
Download Meta Llama Guard 3 (1B) from Hugging Face

Model: meta-llama/Llama-Guard-3-1B
Format: SafeTensors (HuggingFace transformers compatible)
Size:  ~3GB (1B parameters, float16)

This model is used as the safety guardrail layer in MedGemma Sentinel.
It classifies user inputs and bot outputs against the O1-O8 safety taxonomy.

Requirements:
  - huggingface_hub (pip install huggingface_hub)
  - A Hugging Face account with access to the gated model
  - HF token set via: huggingface-cli login
    OR environment variable: HF_TOKEN=hf_xxx

Usage:
  python scripts/download_llama_guard.py
"""

import os
import sys
from pathlib import Path

# Target directory
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_DIR = MODELS_DIR / "Llama-Guard-3-1B"

# Hugging Face repo
REPO_ID = "meta-llama/Llama-Guard-3-1B"

# Expected files in the repo
EXPECTED_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def get_hf_token():
    """Get Hugging Face token from environment or cached login."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Try to read from cached login (~/.huggingface/token or ~/.cache/huggingface/token)
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    return None


def download_with_huggingface_hub():
    """Download the full Llama Guard 3 1B repository using huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[...] huggingface_hub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    token = get_hf_token()

    if not token:
        print("=" * 60)
        print("  AUTHENTICATION REQUIRED")
        print("=" * 60)
        print()
        print(f"  {REPO_ID} is a gated model.")
        print("  You need a Hugging Face token with access granted.")
        print()
        print("  Steps:")
        print("  1. Go to https://huggingface.co/meta-llama/Llama-Guard-3-1B")
        print("  2. Accept the license agreement")
        print("  3. Create a token at https://huggingface.co/settings/tokens")
        print("  4. Set it via one of:")
        print("       export HF_TOKEN=hf_your_token_here")
        print("       huggingface-cli login")
        print("=" * 60)
        print()

        token = input("Paste your HF token (or press Enter to abort): ").strip()
        if not token:
            print("[ABORT] No token provided.")
            sys.exit(1)

    print(f"[...] Downloading {REPO_ID}...")
    print(f"      Target: {TARGET_DIR}")
    print("      This may take a few minutes (~3GB).\n")

    MODELS_DIR.mkdir(exist_ok=True)

    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
        token=token,
        ignore_patterns=["*.md", "*.txt", ".gitattributes", "original/*"],
    )

    print(f"\n[OK] Repository downloaded to: {downloaded_path}")
    return downloaded_path


def download_individual_files():
    """Fallback: download individual files with requests."""
    import requests

    token = get_hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    MODELS_DIR.mkdir(exist_ok=True)
    TARGET_DIR.mkdir(exist_ok=True)

    base_url = f"https://huggingface.co/{REPO_ID}/resolve/main"

    for filename in EXPECTED_FILES:
        url = f"{base_url}/{filename}"
        output_path = TARGET_DIR / filename

        if output_path.exists():
            print(f"  [SKIP] {filename} (already exists)")
            continue

        print(f"  [....] Downloading {filename}...")
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  [....] {filename}: {pct:.1f}%", end="", flush=True)

            size_mb = output_path.stat().st_size / (1024**2)
            print(f"\r  [OK]   {filename} ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"\r  [FAIL] {filename}: {e}")
            if output_path.exists():
                output_path.unlink()

    print(f"\n[OK] Files downloaded to: {TARGET_DIR}")
    return str(TARGET_DIR)


def verify_model(model_dir: str) -> bool:
    """Verify the downloaded model files."""
    model_path = Path(model_dir)

    if not model_path.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        return False

    print("\nVerifying downloaded files:")
    all_ok = True

    for filename in EXPECTED_FILES:
        filepath = model_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            print(f"  [OK] {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {filename}")
            all_ok = False

    # Check model.safetensors size (should be ~3GB)
    safetensors = model_path / "model.safetensors"
    if safetensors.exists():
        size_gb = safetensors.stat().st_size / (1024**3)
        if size_gb < 2.0:
            print(f"  [WARNING] model.safetensors ({size_gb:.2f} GB) seems too small")
            all_ok = False
        else:
            print(f"  [OK] model.safetensors size: {size_gb:.2f} GB")

    return all_ok


def test_model_loading(model_dir: str) -> bool:
    """Test loading the model with transformers."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("\n[...] Testing model loading with transformers...")

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        # Quick classification test
        test_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'User' messages in conversations
according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
O1: Violence and Hate.
O5: Regulated or Controlled Substances.
O8: Medical Ethics and Unethical Procedures.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: What is the recommended dosage of ibuprofen for an adult?

<END CONVERSATION>

Provide your safety assessment for User in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        print(f"  Test input:  'What is the recommended dosage of ibuprofen?'")
        print(f"  Model says:  '{response}'")

        if response.lower().startswith("safe"):
            print("  [OK] Model correctly classified safe input!")
        else:
            print(f"  [WARNING] Unexpected response: {response}")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except ImportError as e:
        print(f"[WARNING] Cannot test: {e}")
        print("  Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"[ERROR] Model loading test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  Llama Guard 3 (1B) - Model Downloader")
    print("  For MedGemma Sentinel Safety Layer")
    print("=" * 60)
    print()
    print(f"  Repository:  {REPO_ID}")
    print(f"  Target dir:  {TARGET_DIR}")
    print()

    # Check if model already exists
    safetensors = TARGET_DIR / "model.safetensors"
    if safetensors.exists() and safetensors.stat().st_size > 2 * (1024**3):
        print("[INFO] Llama Guard 3 1B already downloaded.")
        if verify_model(str(TARGET_DIR)):
            test_model_loading(str(TARGET_DIR))
            return str(TARGET_DIR)

    # Try downloading
    try:
        model_path = download_with_huggingface_hub()
    except Exception as e:
        print(f"\n[WARNING] snapshot_download failed: {e}")
        print("Trying individual file download...\n")
        model_path = download_individual_files()

    # Verify
    if verify_model(model_path):
        test_model_loading(model_path)

    print()
    print("=" * 60)
    print("  Download complete!")
    print()
    print("  The model is now ready for MedGemma Sentinel.")
    print("  It will be loaded automatically by SentinelGuard.")
    print()
    print("  To test:  python -c \"from src.guardrails import SentinelGuard; g = SentinelGuard(); print(g.get_status())\"")
    print("=" * 60)

    return model_path


if __name__ == "__main__":
    main()
