"""
FastMCP Server for Night Watch Medical Tools.
This exposes our audio classifier and database access to ANY MCP-compatible client.
"""

import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# Import your audio inference tool
try:
    from classification_sound_2 import inference
except ImportError:
    print("Warning: 'classification_sound.py' not found. Audio tool will fail.")

# Import night cardiology sentinel (local module)
try:
    from cardiology_sentinel import (
        SubjectInfo, 
        parse_vitals_lines, 
        summarize_window, 
        SentinelInference, 
        build_prompt
    )
    SENTINEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 'cardiology_sentinel' module not found. Error: {e}")
    SENTINEL_AVAILABLE = False

# Initialize the FastMCP Server
mcp = FastMCP("NightWatchServer")


@mcp.tool()
def analyze_room_audio(room_number: str) -> str:
    """
    Analyzes the current acoustic environment of a room. 
    Use this to listen to the patient's breathing/coughing.
    """
    # In a real app, you'd fetch the live stream for the specific room_number
    audio_file = "./Testing_data/Coughing_actresses_153.wav" 
    model_file = "medical_sound_rf_model.joblib"
    try:
        diagnosis = inference(audio_file, saved_model_path=model_file)
        return f"Acoustic AI detected: {diagnosis.upper()} in Room {room_number}"
    except Exception as e:
        return f"Audio analysis failed. Error: {e}"

@mcp.tool()
def get_patient_vitals(patient_id: str) -> str:
    """
    Fetches real-time IoT vital signs (Heart Rate, SpO2, Blood Pressure, etc.).
    """
    db = {
        "101": {"HR": 72, "SpO2": 98, "BP": "120/80", "Temp": 36.8, "ECG": "Normal Sinus"},
        "402": {"HR": 115, "SpO2": 88, "BP": "145/95", "Temp": 37.9, "ECG": "Tachycardia"}
    }
    vitals = db.get(patient_id, "Patient ID not found in Vitals Database.")
    return json.dumps(vitals)

@mcp.tool()
def get_patient_history(patient_id: str) -> str:
    """
    Fetches the patient's medical history and EHR profile.
    """
    db = {
        "101": "45yo Male. No chronic conditions. Admitted for observation.",
        "402": "68yo Male. Severe COPD, Hypertension. High risk for respiratory failure."
    }
    return db.get(patient_id, "EHR not found.")


@mcp.tool()
def get_current_time() -> str:
    """We used to get the current time"""
    # Get the current date and time
    current_datetime = datetime.now()

    # Print the full datetime object (e.g., 2026-02-21 15:50:00.123456)
    result = current_datetime

    return str(result)


@mcp.tool()
def analyze_cardiology_sentinel(patient_id: str, vitals_text: str) -> str:
    """
    Analyzes patient cardiology data using the Night Cardiology Sentinel model.
    Use this when you need to assess cardiac anomalies or heart-related conditions.
    
    Args:
        patient_id: The patient's ID (e.g., "402")
        vitals_text: Multi-line text with vitals in format:
                     "Time: 00:00 - HR: 115, SPO2: 88, RESP: 24\\nTime: 00:01 - HR: 118, SPO2: 87"
    
    Returns detailed cardiac analysis with comparison, detection, and clinical interpretation.
    """
    if not SENTINEL_AVAILABLE:
        return "Error: Night Cardiology Sentinel module not available."
    
    # Get patient info from database
    patient_db = {
        "101": {"code": 101, "gender": "M", "age": 45, "height": 175, "weight": 80},
        "402": {"code": 402, "gender": "M", "age": 68, "height": 170, "weight": 85}
    }
    
    patient_data = patient_db.get(patient_id)
    if not patient_data:
        return f"Error: Patient {patient_id} not found in database."
    
    # Build subject info
    subject = SubjectInfo(
        subject_code=patient_data["code"],
        gender=patient_data["gender"],
        age_years=patient_data["age"],
        length_cm=patient_data["height"],
        weight_kg=patient_data["weight"]
    )
    
    try:
        # Parse vitals from text
        lines = vitals_text.split('\n')
        rows = parse_vitals_lines(lines)
        
        if not rows:
            return "Error: No valid vitals data could be parsed. Expected format: 'Time: X - HR: Y, SPO2: Z'"
        
        # Create a summary of the vitals window
        window_summary = summarize_window(rows)
        
        # Build prompt for the sentinel model
        prompt = build_prompt(subject, window_summary)
        
        # Load model and run inference
        model_path = str(Path(__file__).parent.parent.parent / "models" / "medgemma-night-sentinel-Q4_K_M.gguf")
        
        sentinel = SentinelInference(model_path=model_path, n_ctx=2048)
        result = sentinel.predict(prompt, max_tokens=256)
        
        return f"CARDIOLOGY SENTINEL ANALYSIS for Patient {patient_id}:\n\n{result}"
        
    except Exception as e:
        return f"Error during cardiology analysis: {str(e)}"


if __name__ == "__main__":
    # Run the FastMCP server on stdio
    mcp.run()