from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Make sure your team runs `huggingface-cli login` in their terminal first!

# 1. Automatically fetch the model from your Private Hugging Face Repo
# (It downloads once and caches it locally on their machine)
print("📥 Fetching Night Sentinel from Hugging Face...")
model_path = hf_hub_download(
    repo_id="Ismailea04/medgemma-night-sentinel", # Replace with your repo
    filename="medgemma-night-sentinel-Q4_K_M.gguf"
)

# 2. Load the GGUF model into local RAM/CPU
print("🧠 Booting Clinical Brain...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,       # Memory window for the prompt
    verbose=False     # Set to True if you want to see C++ debug logs
)

# 3. The Function Saad can use for his Agent
def trigger_night_sentinel(patient_profile, vital_signs_anomaly):
    """
    Feeds an anomaly to the steered MedGemma model.
    """
    # Using the strict MedGemma Instruction-Tuning Template
    prompt = (
        f"<start_of_turn>user\n"
        f"[NIGHT SENTINEL SYSTEM]\n"
        f"Patient Profile: {patient_profile}\n"
        f"Event/Anomaly: {vital_signs_anomaly}\n"
        f"    TASK:
    1. Compare the current data to the patient's specific baseline (Name it : ##Comparaison).
    2. Identify any clinical anomalies (Name it : ##Detection).
    3. Provide a short clinical interpretation of what might be happening (Name it : ##Interpretation).<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    
    response = llm(
        prompt,
        max_tokens=256,
        stop=["<end_of_turn>"], # Stop generating when the thought is finished
        echo=False
    )
    
    return response['choices'][0]['text'].strip()

# --- TEST EXECUTION ---
if __name__ == "__main__":
    profile = "65yo male, post-operative recovery. Baseline HR 70."
    anomaly = "Sudden HR spike to 135 bpm with irregular rhythm, SpO2 92%."
    
    print("\n🚨 Triggering Alert Analysis...")
    diagnosis = trigger_night_sentinel(profile, anomaly)
    print(f"\n[SENTINEL REPORT]\n{diagnosis}")