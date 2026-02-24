"""
Download MedGemma 1.5 Medical Q4_K_M model from Hugging Face

Model: hmzBen/medgemma-1.5-medical-q4km
Format: GGUF (llama.cpp compatible)
Size: ~2.5GB (Q4_K_M quantization)
"""

import os
import sys
from pathlib import Path


def download_with_huggingface_hub():
    """Download using huggingface_hub library"""
    try:
        from huggingface_hub import hf_hub_download
        
        print("Downloading MedGemma 1.5 Medical Q4_K_M...")
        print("Repository: hmzBen/medgemma-1.5-medical-q4km")
        print("This may take a few minutes depending on your connection.\n")
        
        # Create models directory
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Download the quantized model
        model_path = hf_hub_download(
            repo_id="hmzBen/medgemma-1.5-medical-q4km",
            filename="medgemma-1.5-medical-Q4_K_M.gguf",
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        
        print(f"\n[SUCCESS] Model downloaded to: {model_path}")
        print(f"Model size: {Path(model_path).stat().st_size / (1024**3):.2f} GB")
        
        return model_path
        
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        # Retry after install
        from huggingface_hub import hf_hub_download
        return download_with_huggingface_hub()


def download_with_requests():
    """Fallback: download using requests"""
    import requests
    from tqdm import tqdm
    
    url = "https://huggingface.co/hmzBen/medgemma-1.5-medical-q4km/resolve/main/medgemma-1.5-medical-Q4_K_M.gguf"
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / "medgemma-1.5-medical-Q4_K_M.gguf"
    
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}\n")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"\n[SUCCESS] Model downloaded to: {output_path}")
    return str(output_path)


def verify_model(model_path: str) -> bool:
    """Verify the downloaded model"""
    path = Path(model_path)
    
    if not path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    
    size_gb = path.stat().st_size / (1024**3)
    if size_gb < 2.0:
        print(f"[WARNING] Model size ({size_gb:.2f} GB) seems too small")
        return False
    
    print(f"[OK] Model verified: {path.name} ({size_gb:.2f} GB)")
    return True


def test_model_loading(model_path: str) -> bool:
    """Test loading the model with llama-cpp-python"""
    try:
        from llama_cpp import Llama
        
        print("\nTesting model loading...")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=0,  # CPU only for test
            n_ctx=512,
            verbose=False,
        )
        
        # Quick test
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
        
        print("[OK] Model loads and runs correctly!")
        return True
        
    except ImportError:
        print("[WARNING] llama-cpp-python not installed. Install with:")
        print("  pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return False


def main():
    print("=" * 60)
    print("MedGemma 1.5 Medical Q4_K_M - Model Downloader")
    print("=" * 60)
    print()
    
    # Check if model already exists
    models_dir = Path(__file__).parent.parent / "models"
    existing_model = models_dir / "medgemma-1.5-medical-Q4_K_M.gguf"
    
    if existing_model.exists():
        print(f"[INFO] Model already exists: {existing_model}")
        if verify_model(str(existing_model)):
            test_model_loading(str(existing_model))
            return str(existing_model)
    
    # Try downloading
    try:
        model_path = download_with_huggingface_hub()
    except Exception as e:
        print(f"huggingface_hub failed: {e}")
        print("Trying direct download...")
        model_path = download_with_requests()
    
    # Verify and test
    if verify_model(model_path):
        test_model_loading(model_path)
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Install llama-cpp-python: pip install llama-cpp-python")
    print("2. Run the demo: python examples/demo_with_medgemma.py")
    print("=" * 60)
    
    return model_path


if __name__ == "__main__":
    main()
