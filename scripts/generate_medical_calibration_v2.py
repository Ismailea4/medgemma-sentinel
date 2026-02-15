import os
import random
from datasets import load_dataset

# Configuration
output_path = "/Users/abdul/Downloads/GemmaQuantiazed/calibration_data/medical_calibration_sota_v3.txt"
total_target_words = 40000  # Approx 52k tokens (Safe buffer)

# Ratios
ratios = {
    "dialogue": 0.30,  # 30% Doctor-Patient Chat
    "facts": 0.30,     # 30% NIH Medical Facts
    "reasoning": 0.40  # 40% USMLE Hard Logic
}

targets = {k: int(total_target_words * v) for k, v in ratios.items()}
collected = {"dialogue": [], "facts": [], "reasoning": []}
counts = {"dialogue": 0, "facts": 0, "reasoning": 0}

print(f"🚀 Starting SOTA Calibration Gen V3 (Stable)...")
print(f"🎯 Targets: {targets} words")

# ---------------------------------------------------------
# 1. DIALOGUE (30%) - ChatDoctor
# ---------------------------------------------------------
print("1️⃣  Fetching Dialogue (ChatDoctor)...")
try:
    ds = load_dataset("Lavita/ChatDoctor-HealthCareMagic-100k", split="train", streaming=True)
    for row in ds:
        if counts["dialogue"] >= targets["dialogue"]: break
        text = f"Patient: {row['input']}\nDoctor: {row['output']}\n"
        collected["dialogue"].append(text)
        counts["dialogue"] += len(text.split())
except Exception as e:
    print(f"⚠️ Error loading ChatDoctor: {e}")

# ---------------------------------------------------------
# 2. FACTS (30%) - Multiple Fallbacks (NIH/Medical Data)
# ---------------------------------------------------------
print("2️⃣  Fetching Medical Facts (MedAlpaca)...")
try:
    # This dataset is extremely stable and high quality
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train", streaming=True)
    
    for row in ds:
        if counts["facts"] >= targets["facts"]: break
        
        # Format: Input (Question) -> Output (Answer)
        text = f"Fact Query: {row['input']}\nFact Answer: {row['output']}\n"
        collected["facts"].append(text)
        counts["facts"] += len(text.split())
        
except Exception as e:
    print(f"⚠️ Error loading MedAlpaca: {e}")
# ---------------------------------------------------------
# 3. REASONING (40%) - USMLE (Medical Board Exams)
# ---------------------------------------------------------
print("3️⃣  Fetching Reasoning (MedQA-USMLE)...")
try:
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", streaming=True)
    for row in ds:
        if counts["reasoning"] >= targets["reasoning"]: break
        
        question = row['question']
        answer = row['answer'] 
        text = f"Clinical Question: {question}\nCorrect Diagnosis: {answer}\n"
        
        collected["reasoning"].append(text)
        counts["reasoning"] += len(text.split())
except Exception as e:
    print(f"⚠️ Error loading USMLE: {e}")

# ---------------------------------------------------------
# 4. MIX & SHUFFLE (Crucial for I-Matrix Stability)
# ---------------------------------------------------------
print("🔄 Interleaving Data Sources...")
final_lines = []

# We mix them so the model sees [Chat -> Fact -> Logic -> Chat...]
# This prevents the quantizer from overfitting to one style early on.
max_len = max(len(collected["dialogue"]), len(collected["facts"]), len(collected["reasoning"]))

for i in range(max_len):
    if i < len(collected["dialogue"]): final_lines.append(collected["dialogue"][i])
    if i < len(collected["facts"]): final_lines.append(collected["facts"][i])
    if i < len(collected["reasoning"]): final_lines.append(collected["reasoning"][i])

full_text = "\n".join(final_lines)

# Write to file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\n✅ DONE. File saved: {output_path}")
print(f"📊 Final Word Count: {len(full_text.split())}")
print(f"✓ Dialogue collected: {counts['dialogue']} words")
print(f"✓ Facts collected: {counts['facts']} words")
print(f"✓ Reasoning collected: {counts['reasoning']} words")
print("🧪 Mix: 30% Chat / 30% Science / 40% Hard Logic.")
