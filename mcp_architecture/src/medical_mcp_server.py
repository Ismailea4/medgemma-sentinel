"""
FastMCP Server for Night Watch Medical Tools.
This exposes our audio classifier and database access to ANY MCP-compatible client.
"""

import json
from mcp.server.fastmcp import FastMCP
from datetime import datetime


# Import your audio inference tool
try:
    from classification_sound_2 import inference
except ImportError:
    print("Warning: 'classification_sound.py' not found. Audio tool will fail.")

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

if __name__ == "__main__":
    # Run the FastMCP server on stdio
    mcp.run()