import os
import sys
import platform
import subprocess
import psutil


def find_llama_server_binary() -> str:
    """Find the llama-server binary in common build locations."""
    env_path = os.getenv("LLAMA_SERVER_PATH")
    if env_path and os.path.exists(env_path) and os.access(env_path, os.X_OK):
        return env_path

    candidates = [
        "./llama.cpp/llama-server",
        "./llama.cpp/build/bin/llama-server",
        "./llama.cpp/build/llama-server",
        "./llama.cpp/bin/llama-server",
    ]

    for path in candidates:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    print("Error: could not find 'llama-server' binary.")
    print("Build it first (see llama.cpp/docs/build.md) or set LLAMA_SERVER_PATH.")
    sys.exit(1)


def has_nvidia_gpu() -> bool:
    system = platform.system()
    if system not in ("Linux", "Windows"):
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_optimal_settings() -> tuple[int, int]:
    """Detect hardware and return (ngl, ctx)."""
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    system = platform.system()
    machine = platform.machine()

    print(f"Detected: {system} ({machine}) | RAM: {total_ram_gb:.1f} GB")

    ctx = 2048
    ngl = 0

    if system == "Darwin" and machine == "arm64":
        print("Apple Silicon detected (Metal build expected).")
        ngl = 99
        if total_ram_gb >= 16:
            ctx = 8192
        elif total_ram_gb >= 8:
            ctx = 4096
        else:
            ctx = 2048
    elif has_nvidia_gpu():
        print("NVIDIA GPU detected.")
        ngl = 99
        ctx = 8192 if total_ram_gb >= 12 else 4096
    else:
        if total_ram_gb >= 16:
            ctx = 4096
        else:
            ctx = 2048

    return ngl, ctx


def start() -> None:
    binary_path = find_llama_server_binary()
    ngl, ctx = get_optimal_settings()

    model_path = "outputs/medgemma-1.5-medical-Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        sys.exit(1)

    cmd = [
        binary_path,
        "-m",
        model_path,
        "--port",
        "8080",
        "--host",
        "0.0.0.0",
        "-c",
        str(ctx),
        "-ngl",
        str(ngl),
        "--alias",
        "medgemma",
    ]

    print("Launching:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    start()
