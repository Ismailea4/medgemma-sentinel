"""
Clinical Events Models - Night surveillance and Day consultation events
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"           # Informational, no action needed
    LOW = "low"             # Minor concern, monitor
    MEDIUM = "medium"       # Moderate concern, assess soon
    HIGH = "high"           # Urgent, immediate assessment needed
    CRITICAL = "critical"   # Emergency, immediate intervention required


class EventType(str, Enum):
    """Types of clinical events"""
    # Night events
    APNEA = "apnea"
    DESATURATION = "desaturation"
    TACHYCARDIA = "tachycardia"
    BRADYCARDIA = "bradycardia"
    AGITATION = "agitation"
    FALL_RISK = "fall_risk"
    FEVER = "fever"
    HYPOTHERMIA = "hypothermia"
    ABNORMAL_BREATHING = "abnormal_breathing"
    VOCAL_DISTRESS = "vocal_distress"
    
    # Day events
    CONSULTATION = "consultation"
    EXAMINATION = "examination"
    MEDICATION_GIVEN = "medication_given"
    PROCEDURE = "procedure"
    VITAL_CHECK = "vital_check"
    TRIAGE = "triage"


class DataSource(str, Enum):
    """Source of event detection"""
    SENSOR_SPO2 = "sensor_spo2"
    SENSOR_ECG = "sensor_ecg"
    SENSOR_TEMP = "sensor_temperature"
    CAMERA_IR = "camera_ir"
    AUDIO_ANALYSIS = "audio_analysis"
    MANUAL_INPUT = "manual_input"
    AI_INFERENCE = "ai_inference"


class ClinicalEvent(BaseModel):
    """
    Base clinical event model
    Represents any significant clinical occurrence
    """
    id: str = Field(..., description="Unique event identifier")
    patient_id: str = Field(..., description="Associated patient ID")
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Classification
    alert_level: AlertLevel = Field(default=AlertLevel.INFO)
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Priority 0-1")
    
    # Context
    description: str = Field(..., description="Human-readable description")
    data_sources: List[DataSource] = Field(default_factory=list)
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    
    # AI analysis
    ai_analysis: Optional[str] = Field(None, description="MedGemma analysis")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Response
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    action_taken: Optional[str] = None
    
    # Relationships
    related_events: List[str] = Field(default_factory=list, description="Related event IDs")
    
    def acknowledge(self, by: str, action: Optional[str] = None) -> None:
        """Mark event as acknowledged"""
        self.acknowledged = True
        self.acknowledged_by = by
        self.acknowledged_at = datetime.now()
        self.action_taken = action
    
    def to_report_entry(self) -> Dict[str, Any]:
        """Convert to report-friendly format"""
        return {
            "time": self.timestamp.strftime("%H:%M"),
            "type": self.event_type.value,
            "level": self.alert_level.value,
            "description": self.description,
            "action": self.action_taken or "En attente"
        }


class NightEvent(ClinicalEvent):
    """
    Night surveillance event
    Detected during autonomous monitoring (Guard Night mode)
    """
    # Night-specific fields
    sleep_stage: Optional[str] = Field(None, description="Sleep stage if known")
    duration_seconds: Optional[int] = Field(None, description="Event duration")
    recurrence_count: int = Field(default=1, description="Number of occurrences")
    
    # Multimodal data
    audio_detected: bool = Field(default=False, description="Audio anomaly detected")
    audio_type: Optional[str] = Field(None, description="Type of audio: stridor, wheeze, snoring, etc.")
    
    vision_detected: bool = Field(default=False, description="Vision anomaly detected")
    vision_type: Optional[str] = Field(None, description="Type: movement, posture, fall, etc.")
    
    sensor_detected: bool = Field(default=False, description="Sensor anomaly detected")
    
    # Patient interaction
    patient_responded: Optional[bool] = Field(None, description="Did patient respond to voice check")
    response_quality: Optional[str] = Field(None, description="Response quality")
    
    # Fusion analysis
    fusion_score: float = Field(default=0.0, description="Multimodal fusion confidence")
    fusion_reasoning: Optional[str] = Field(None, description="Why sources were combined")
    
    def get_multimodal_summary(self) -> str:
        """Get summary of all data sources involved"""
        sources = []
        if self.audio_detected:
            sources.append(f"Audio ({self.audio_type})")
        if self.vision_detected:
            sources.append(f"Vision ({self.vision_type})")
        if self.sensor_detected:
            sources.append("Capteurs")
        return " + ".join(sources) if sources else "Aucune source"


class DayEvent(ClinicalEvent):
    """
    Day consultation/assistance event
    Occurs during medical staff interaction (Medi-Atlas mode)
    """
    # Consultation context
    consultation_mode: str = Field(default="general", description="Specialty mode: cardio, dermato, etc.")
    presenting_complaint: Optional[str] = Field(None, description="Main symptom/complaint")
    
    # Clinical assessment
    symptoms: List[str] = Field(default_factory=list)
    physical_exam: Dict[str, str] = Field(default_factory=dict)
    
    # Images/Media analyzed
    images_analyzed: List[str] = Field(default_factory=list, description="Image file references")
    audio_analyzed: List[str] = Field(default_factory=list, description="Audio file references")
    
    # AI assistance
    differential_diagnosis: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    severity_assessment: Optional[str] = Field(None)
    
    # Outcome
    final_diagnosis: Optional[str] = None
    treatment_plan: Optional[str] = None
    follow_up_required: bool = Field(default=False)
    referral_needed: Optional[str] = Field(None, description="Specialty referral if needed")
    
    def get_clinical_summary(self) -> str:
        """Generate clinical summary for reporting"""
        summary_parts = []
        
        if self.presenting_complaint:
            summary_parts.append(f"Motif: {self.presenting_complaint}")
        
        if self.symptoms:
            summary_parts.append(f"Symptômes: {', '.join(self.symptoms)}")
        
        if self.differential_diagnosis:
            summary_parts.append(f"Diagnostics différentiels: {', '.join(self.differential_diagnosis[:3])}")
        
        if self.severity_assessment:
            summary_parts.append(f"Gravité: {self.severity_assessment}")
        
        if self.recommended_actions:
            summary_parts.append(f"Actions recommandées: {', '.join(self.recommended_actions[:3])}")
        
        return "\n".join(summary_parts)


class EventTimeline(BaseModel):
    """
    Timeline of events for a patient
    Used for longitudinal analysis and reporting
    """
    patient_id: str
    start_time: datetime
    end_time: datetime
    events: List[ClinicalEvent] = Field(default_factory=list)
    
    # Aggregated statistics
    total_events: int = Field(default=0)
    critical_count: int = Field(default=0)
    high_count: int = Field(default=0)
    
    def add_event(self, event: ClinicalEvent) -> None:
        """Add event to timeline and update statistics"""
        self.events.append(event)
        self.total_events += 1
        if event.alert_level == AlertLevel.CRITICAL:
            self.critical_count += 1
        elif event.alert_level == AlertLevel.HIGH:
            self.high_count += 1
    
    def get_events_by_level(self, level: AlertLevel) -> List[ClinicalEvent]:
        """Filter events by alert level"""
        return [e for e in self.events if e.alert_level == level]
    
    def get_events_by_type(self, event_type: EventType) -> List[ClinicalEvent]:
        """Filter events by type"""
        return [e for e in self.events if e.event_type == event_type]
    
    @property
    def night_events(self) -> List[NightEvent]:
        """Get all night events"""
        return [e for e in self.events if isinstance(e, NightEvent)]
    
    @property
    def day_events(self) -> List[DayEvent]:
        """Get all day events"""
        return [e for e in self.events if isinstance(e, DayEvent)]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for reporting"""
        return {
            "period": f"{self.start_time.strftime('%d/%m/%Y')} - {self.end_time.strftime('%d/%m/%Y')}",
            "total_events": self.total_events,
            "critical": self.critical_count,
            "high": self.high_count,
            "night_events": len(self.night_events),
            "day_events": len(self.day_events),
            "event_types": {
                t.value: len([e for e in self.events if e.event_type == t])
                for t in EventType
                if any(e.event_type == t for e in self.events)
            }
        }
