"""
Sentinel State - Shared state for LangGraph workflow
Defines the state that flows through Night -> Rap1 -> Day -> Rap2
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field
from enum import Enum
import operator


class WorkflowPhase(str, Enum):
    """Current phase of the Sentinel workflow"""
    IDLE = "idle"
    NIGHT = "night"           # Surveillance nocturne
    RAP1 = "rap1"             # Génération rapport de nuit
    DAY = "day"               # Assistance médicale jour
    RAP2 = "rap2"             # Génération rapport de consultation
    COMPLETED = "completed"


class SteeringMode(str, Enum):
    """MedGemma steering mode (context-aware personality)"""
    NIGHT_SURVEILLANCE = "night_surveillance"    # Détection apnées, agitation, hypoxie
    SPECIALIST_VIRTUAL = "specialist_virtual"    # Aide au diagnostic
    TRIAGE_PRIORITY = "triage_priority"          # Évaluation gravité immédiate
    LONGITUDINAL = "longitudinal"                # Analyse sur 7-30 jours


class NightData(BaseModel):
    """Data collected during night surveillance"""
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Vital signs timeline
    vitals_readings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Detected events
    events: List[Dict[str, Any]] = Field(default_factory=list)
    alerts_triggered: int = Field(default=0)
    critical_alerts: int = Field(default=0)
    
    # Multimodal data
    audio_events: List[Dict[str, Any]] = Field(default_factory=list)
    vision_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Sleep quality metrics
    sleep_quality_score: Optional[float] = None
    total_sleep_time_hours: Optional[float] = None
    apnea_hypopnea_index: Optional[float] = None
    
    # Patient interactions
    voice_checks_performed: int = Field(default=0)
    patient_responses: List[Dict[str, Any]] = Field(default_factory=list)


class DayData(BaseModel):
    """Data collected during day consultation"""
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Consultation info
    consultation_mode: str = Field(default="general")
    presenting_complaint: Optional[str] = None
    
    # Clinical data
    symptoms: List[str] = Field(default_factory=list)
    physical_exam: Dict[str, str] = Field(default_factory=dict)
    vitals: Optional[Dict[str, Any]] = None
    
    # Images analyzed
    images: List[Dict[str, Any]] = Field(default_factory=list)
    
    # AI Analysis results
    differential_diagnosis: List[str] = Field(default_factory=list)
    severity_assessment: Optional[str] = None
    recommended_actions: List[str] = Field(default_factory=list)
    
    # Outcome
    final_diagnosis: Optional[str] = None
    treatment_plan: Optional[str] = None
    referral_needed: Optional[str] = None


class ReportData(BaseModel):
    """Generated report data"""
    report_type: str = Field(default="night")  # night, consultation, triage
    generated_at: datetime = Field(default_factory=datetime.now)
    
    # Content
    title: str = Field(default="Rapport Clinique")
    summary: str = Field(default="")
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    patient_summary: Optional[str] = None
    period_covered: Optional[str] = None
    
    # Output
    markdown_content: str = Field(default="")
    pdf_path: Optional[str] = None


class SentinelState(BaseModel):
    """
    Central state for MedGemma Sentinel workflow
    This state flows through all nodes: Night -> Rap1 -> Day -> Rap2
    
    LangGraph uses this as the shared state between all nodes.
    """
    # === Identification ===
    session_id: str = Field(default="", description="Unique session identifier")
    patient_id: str = Field(default="", description="Current patient ID")
    
    # === Workflow Control ===
    phase: WorkflowPhase = Field(default=WorkflowPhase.IDLE)
    steering_mode: SteeringMode = Field(default=SteeringMode.NIGHT_SURVEILLANCE)
    
    # === Timestamps ===
    workflow_start: datetime = Field(default_factory=datetime.now)
    workflow_end: Optional[datetime] = None
    current_phase_start: Optional[datetime] = None
    
    # === Patient Context (from GraphRAG) ===
    patient_context: Dict[str, Any] = Field(default_factory=dict)
    patient_history_summary: str = Field(default="")
    risk_factors: List[str] = Field(default_factory=list)
    
    # === Phase Data ===
    night_data: Optional[NightData] = None
    day_data: Optional[DayData] = None
    
    # === Reports ===
    rap1_report: Optional[ReportData] = None  # Night surveillance report
    rap2_report: Optional[ReportData] = None  # Day consultation report
    
    # === Messages for LLM (LangGraph pattern) ===
    messages: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
    
    # === Error Handling ===
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # === Metrics ===
    total_events_processed: int = Field(default=0)
    total_alerts: int = Field(default=0)
    processing_time_seconds: float = Field(default=0.0)
    
    def transition_to(self, new_phase: WorkflowPhase) -> None:
        """Transition to a new workflow phase"""
        self.phase = new_phase
        self.current_phase_start = datetime.now()
        
        # Update steering mode based on phase
        if new_phase == WorkflowPhase.NIGHT:
            self.steering_mode = SteeringMode.NIGHT_SURVEILLANCE
        elif new_phase == WorkflowPhase.DAY:
            self.steering_mode = SteeringMode.SPECIALIST_VIRTUAL
        elif new_phase in [WorkflowPhase.RAP1, WorkflowPhase.RAP2]:
            self.steering_mode = SteeringMode.LONGITUDINAL
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_error(self, error: str) -> None:
        """Record an error"""
        self.errors.append(f"[{datetime.now().isoformat()}] {error}")
    
    def add_warning(self, warning: str) -> None:
        """Record a warning"""
        self.warnings.append(f"[{datetime.now().isoformat()}] {warning}")
    
    def get_state_summary(self) -> str:
        """Get a summary of current state for logging"""
        return f"""
=== MedGemma Sentinel State ===
Session: {self.session_id}
Patient: {self.patient_id}
Phase: {self.phase.value}
Steering Mode: {self.steering_mode.value}
Events Processed: {self.total_events_processed}
Alerts: {self.total_alerts}
Errors: {len(self.errors)}
Warnings: {len(self.warnings)}
"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "phase": self.phase.value,
            "steering_mode": self.steering_mode.value,
            "workflow_start": self.workflow_start.isoformat(),
            "total_events": self.total_events_processed,
            "total_alerts": self.total_alerts,
            "has_night_data": self.night_data is not None,
            "has_day_data": self.day_data is not None,
            "has_rap1": self.rap1_report is not None,
            "has_rap2": self.rap2_report is not None,
            "errors": len(self.errors),
            "warnings": len(self.warnings)
        }


# Type alias for LangGraph
StateType = Dict[str, Any]


def create_initial_state(patient_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Create initial state dictionary for LangGraph"""
    import uuid
    
    state = SentinelState(
        session_id=session_id or str(uuid.uuid4()),
        patient_id=patient_id,
        phase=WorkflowPhase.IDLE,
        workflow_start=datetime.now()
    )
    
    return state.model_dump()
