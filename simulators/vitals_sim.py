import random
from datetime import datetime, timedelta

class VitalsSimulator:
    """
    Generates synthetic JSON payloads for MedGemma Sentinel's Night Mode.
    Simulates both Vitals and Audio (Whisper) events.
    """
    
    def __init__(self, patient_case="General Admission"):
        self.patient_case = patient_case
        self.base_time = datetime.strptime("22:00", "%H:%M")

    def _format_time(self, minutes_added):
        target = self.base_time + timedelta(minutes=minutes_added)
        return target.strftime("%I:%M %p")

    def generate_night_events(self) -> list:
        """Returns a list of event dictionaries simulating a night shift."""
        
        # 1. Baseline: Normal events that happen for everyone
        events = [
            {
                "timestamp": self._format_time(0), 
                "type": "vitals_check", 
                "spo2": 98, "hr": 72, 
                "severity": "low", 
                "description": "Baseline vitals established. Patient resting."
            },
            {
                "timestamp": self._format_time(120), 
                "type": "audio_event", 
                "audio_class": "quiet", 
                "severity": "low", 
                "description": "Ambient room noise normal. No distress detected."
            }
        ]

        # 2. The Anomaly: Injected based on the UI's selected Patient Profile
        if self.patient_case == "Cardiac History":
            events.append({
                "timestamp": self._format_time(245), # ~02:05 AM
                "type": "vitals_anomaly", 
                "spo2": 95, "hr": 155, 
                "severity": "critical", 
                "description": "Sudden tachycardia detected. Heart rate exceeded safe threshold."
            })
            
        elif self.patient_case == "Respiratory Risk":
            events.append({
                "timestamp": self._format_time(254), # 02:14 AM
                "type": "vitals_anomaly", 
                "spo2": 88, "hr": 92, 
                "severity": "critical", 
                "description": "Rapid SpO2 desaturation."
            })
            events.append({
                "timestamp": self._format_time(255), # 02:15 AM
                "type": "audio_event", 
                "audio_class": "labored_breathing", 
                "severity": "high", 
                "description": "Acoustic pattern matches labored breathing / stridor."
            })
            
        else: # General Admission
            events.append({
                "timestamp": self._format_time(300), # 03:00 AM
                "type": "vitals_anomaly", 
                "spo2": 94, "hr": 110, 
                "severity": "medium", 
                "description": "Mild fever onset detected via thermal sensor."
            })

        return events

# Quick test if run directly
if __name__ == "__main__":
    sim = VitalsSimulator("Respiratory Risk")
    print(sim.generate_night_events())