"""
Local Storage Manager - File-based patient memory
Handles persistent storage of night events and alerts for agentic memory.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def safe_load_json(path):
    """Safely load JSON with fallback to empty list."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def safe_write_json(path, data):
    """Safely write JSON with automatic directory creation."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        print(f"Warning: Failed to save {path}: {e}")
        return False


class LocalStorage:
    """
    File-based patient memory storage.
    Organizes patient data by ID with support for profiles, night logs, and history.
    """

    def __init__(self, base_path="data/patients"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _get_patient_folder(self, patient_id):
        """Get or create patient folder."""
        folder = os.path.join(self.base_path, patient_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    def save_night_event(self, patient_id, event_data, ai_reasoning):
        """
        Save a critical event to the patient's night log.
        
        Args:
            patient_id: Unique patient identifier
            event_data: Dict with vitals snapshot (hr, spo2, audio_db, trend)
            ai_reasoning: String explanation from MedGemma
            
        Returns:
            int: Total number of events logged for this patient tonight
        """
        folder = self._get_patient_folder(patient_id)
        file_path = os.path.join(folder, "night_log.json")

        # Structure the event entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "CRITICAL_RESPIRATORY_DISTRESS",  # Standardized event type
            "event_data": event_data,  # {"hr": 140, "spo2": 88, "audio_db": 85, "trend": "..."}
            "ai_reasoning": ai_reasoning,
            "status": "UNRESOLVED"
        }

        # Load existing history
        history = safe_load_json(file_path)

        # Append new event
        history.append(entry)

        # Write back
        if safe_write_json(file_path, history):
            return len(history)
        return -1

    def get_night_events(self, patient_id):
        """
        Get ALL events recorded for patient last night.
        Used by Day Mode to show full timeline.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            list: All events from night_log.json or empty list
        """
        folder = self._get_patient_folder(patient_id)
        file_path = os.path.join(folder, "night_log.json")
        return safe_load_json(file_path)

    def get_night_summary(self, patient_id):
        """
        Get the MOST RECENT event for quick display.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            dict: Last event or None
        """
        events = self.get_night_events(patient_id)
        return events[-1] if events else None

    def save_patient_profile(self, patient_id, profile_data):
        """
        Save static patient profile info (name, age, conditions, etc).
        
        Args:
            patient_id: Patient identifier
            profile_data: Dict with patient metadata
            
        Returns:
            bool: Success status
        """
        folder = self._get_patient_folder(patient_id)
        file_path = os.path.join(folder, "profile.json")
        profile_data["created_at"] = datetime.now().isoformat()
        return safe_write_json(file_path, profile_data)

    def get_patient_profile(self, patient_id):
        """
        Retrieve patient profile info.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            dict: Patient profile or empty dict
        """
        folder = self._get_patient_folder(patient_id)
        file_path = os.path.join(folder, "profile.json")
        return safe_load_json(file_path) if os.path.exists(file_path) else {}

    def clear_night_log(self, patient_id):
        """
        Clear the night log for a patient (useful for testing/resetting).
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            bool: Success status
        """
        folder = self._get_patient_folder(patient_id)
        file_path = os.path.join(folder, "night_log.json")
        if os.path.exists(file_path):
            return safe_write_json(file_path, [])
        return True
