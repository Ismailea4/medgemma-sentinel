"""
Custom Medical Sound Classifier using Google HeAR
Designed to be imported as a tool/module for an AI Agent.
"""

import os
import glob
import numpy as np
import librosa
import warnings
import getpass
import joblib
import sys
from dotenv import load_dotenv

load_dotenv()

# CRITICAL FOR MCP: Silence TensorFlow so it doesn't break the JSON pipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
# CRITICAL FOR MCP: Disable HuggingFace progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from huggingface_hub import from_pretrained_keras, login
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="soundfile")
warnings.filterwarnings("ignore", module="librosa")

# --- 1. CONFIGURATION ---
DATASET_DIR = "./dataset/"  
MODEL_SAVE_PATH = "medical_sound_rf_model.joblib"

# HeAR Parameters (Do not change these, the model expects them)
SAMPLE_RATE = 16000
CLIP_DURATION = 2
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION
CLIP_OVERLAP_PERCENT = 10
CLIP_IGNORE_SILENT_CLIPS = True
SILENCE_RMS_THRESHOLD_DB = -50

# def authenticate_huggingface():
#     token = os.environ.get("HF_TOKEN")
#     if not token:
#         # We CANNOT ask for manual input here. It will freeze the MCP server.
#         raise ValueError("CRITICAL ERROR: 'HF_TOKEN' is missing from the environment. Please add it to your .env file.")
    
#     print("Using HF_TOKEN from environment variables.", file=sys.stderr)
#     login(token=token)

def load_hear_model():
    print("Loading Google HeAR model...", file=sys.stderr)
    try:
        loaded_model = from_pretrained_keras("google/hear")
        infer = loaded_model.signatures["serving_default"]
        print("HeAR Model loaded successfully.", file=sys.stderr)
        return infer
    except Exception as e:
        print("Failed to load model. Did you authenticate with Hugging Face?", file=sys.stderr)
        raise e

def extract_embeddings_from_file(file_path, infer_fn):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None

    clip_batch = []
    overlap_samples = int(CLIP_LENGTH * (CLIP_OVERLAP_PERCENT / 100))
    step_size = CLIP_LENGTH - overlap_samples
    num_clips = max(1, (len(audio) - overlap_samples) // step_size)

    for i in range(num_clips):
        start_sample = i * step_size
        end_sample = start_sample + CLIP_LENGTH
        clip = audio[start_sample:end_sample]
        
        if end_sample > len(audio):
            clip = np.pad(clip, (0, CLIP_LENGTH - len(clip)), 'constant')
            
        rms_loudness = round(20 * np.log10(np.sqrt(np.mean(clip**2) + 1e-10)))
        if CLIP_IGNORE_SILENT_CLIPS and rms_loudness < SILENCE_RMS_THRESHOLD_DB:
            continue
            
        clip_batch.append(clip)

    if not clip_batch:
        return None

    clip_batch = np.asarray(clip_batch)
    embedding_batch = infer_fn(x=clip_batch)['output_0'].numpy()
    return embedding_batch

def process_dataset(data_dir, infer_fn):
    all_embeddings = []
    all_labels = []
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found classes: {classes}", file=sys.stderr)
    
    if not classes:
        raise ValueError(f"No subfolders found in {data_dir}.")

    for class_label in classes:
        class_dir = os.path.join(data_dir, class_label)
        audio_files = glob.glob(os.path.join(class_dir, "*.wav")) + glob.glob(os.path.join(class_dir, "*.ogg"))
        
        for file_path in audio_files:
            embeddings = extract_embeddings_from_file(file_path, infer_fn)
            if embeddings is not None:
                for emb in embeddings:
                    all_embeddings.append(emb)
                    all_labels.append(class_label)
                    
    return np.array(all_embeddings), np.array(all_labels)

def train(save_path=MODEL_SAVE_PATH):
    # authenticate_huggingface()
    infer_fn = load_hear_model()
    
    print("\n--- Extracting Embeddings ---", file=sys.stderr)
    X, y = process_dataset(DATASET_DIR, infer_fn)
    
    if len(X) == 0:
        print("No valid audio segments found. Check your dataset directory.", file=sys.stderr)
        return

    unique_classes, class_counts = np.unique(y, return_counts=True)
    if class_counts.min() < 2:
        stratify_param = None
    else:
        stratify_param = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_param)
    print("\n--- Training Random Forest Classifier ---", file=sys.stderr)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"[Random Forest] Accuracy: {acc*100:.2f}%", file=sys.stderr)

    joblib.dump(model, save_path)
    print(f"\nModel successfully saved to: {save_path}", file=sys.stderr)
    return save_path

def classify_new_audio(file_path, model, infer_fn):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    embeddings = extract_embeddings_from_file(file_path, infer_fn)
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Could not extract embeddings. The file might be invalid or pure silence.")
        
    predictions = model.predict(embeddings)
    unique_classes, counts = np.unique(predictions, return_counts=True)
    final_prediction = unique_classes[np.argmax(counts)]
    return str(final_prediction)

def inference(file_path, saved_model_path=MODEL_SAVE_PATH):
    print(f"\n--- Running Agent Inference on: {file_path} ---", file=sys.stderr)
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"Trained model not found at {saved_model_path}. Please run train() first.")
        
    #authenticate_huggingface()
    model = joblib.load(saved_model_path)
    infer_fn = load_hear_model()
    diagnosis = classify_new_audio(file_path, model, infer_fn)
    
    print(f">> PATIENT DIAGNOSIS: {diagnosis.upper()} <<", file=sys.stderr)
    return diagnosis

if __name__ == "__main__":
    DO_TRAIN = False 
    if DO_TRAIN:
        train(MODEL_SAVE_PATH)

    AUDIO_TO_CLASSIFY = "./Testing_data/Coughing_actresses_153.wav"
    if AUDIO_TO_CLASSIFY and os.path.exists(AUDIO_TO_CLASSIFY):
        result = inference(AUDIO_TO_CLASSIFY, MODEL_SAVE_PATH)
        print(f"Agent received variable: {result}", file=sys.stderr)