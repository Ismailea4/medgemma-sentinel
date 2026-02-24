"""
MedGemma Sentinel - Launcher
Starts the llama.cpp server with the quantized model and runs the demo.

Usage:
    python launch.py              # Start server + run demo
    python launch.py --server     # Start server only
    python launch.py --demo       # Run demo only (server must be running)
"""

import subprocess
import sys
import time
import os
import signal
import requests
from pathlib import Path

# Configuration matching setup from (from "The MedGemma Engine.pdf")
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "medgemma-1.5-medical-Q4_K_M.gguf"
LLAMA_SERVER = PROJECT_DIR / "llama-cpp" / "llama-server.exe"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
CONTEXT_SIZE = 4096  


def check_server_health(timeout=5):
    """Check if llama-server is running and healthy"""
    try:
        r = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def start_server():
    """Start llama-server with the Q4_K_M model"""
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Download it from: https://huggingface.co/hmzBen/medgemma-1.5-medical-q4km")
        sys.exit(1)

    if not LLAMA_SERVER.exists():
        print(f"[ERROR] llama-server not found: {LLAMA_SERVER}")
        sys.exit(1)

    if check_server_health():
        print(f"[OK] Server already running on port {SERVER_PORT}")
        return None

    print(f"[STARTING] llama-server on port {SERVER_PORT}...")
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  Context: {CONTEXT_SIZE} tokens")

    cmd = [
        str(LLAMA_SERVER),
        "-m", str(MODEL_PATH),
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "-c", str(CONTEXT_SIZE),
        "--alias", "medgemma",
    ]

    # Start server as subprocess
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
    )

    # Wait for server to be ready
    print("[WAITING] Server loading model...", end="", flush=True)
    for i in range(120):  # Wait up to 120 seconds
        time.sleep(1)
        if check_server_health(timeout=2):
            print(f"\n[OK] Server ready! (took {i+1}s)")
            return server_process
        if server_process.poll() is not None:
            stderr = server_process.stderr.read().decode(errors='replace')
            print(f"\n[ERROR] Server crashed:\n{stderr[:500]}")
            sys.exit(1)
        print(".", end="", flush=True)

    print("\n[ERROR] Server did not start within 120 seconds")
    server_process.terminate()
    sys.exit(1)


def run_demo():
    """Run the MedGemma demo workflow"""
    if not check_server_health():
        print("[ERROR] Server is not running. Start it first with: python launch.py --server")
        sys.exit(1)

    print("\n[DEMO] Running MedGemma Sentinel workflow...")
    from examples.demo_with_medgemma import run_complete_workflow_with_medgemma
    run_complete_workflow_with_medgemma()


def stop_server(process):
    """Stop the llama-server"""
    if process and process.poll() is None:
        print("\n[STOPPING] Shutting down server...")
        if os.name == 'nt':
            process.terminate()
        else:
            process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        print("[OK] Server stopped")


def main():
    args = sys.argv[1:]
    server_only = "--server" in args
    demo_only = "--demo" in args

    if demo_only:
        run_demo()
        return

    # Start server
    server_process = start_server()

    if server_only:
        print(f"\n[INFO] Server running at http://{SERVER_HOST}:{SERVER_PORT}")
        print("[INFO] Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_server(server_process)
        return

    # Start server + run demo
    try:
        run_demo()
    finally:
        stop_server(server_process)


if __name__ == "__main__":
    main()
