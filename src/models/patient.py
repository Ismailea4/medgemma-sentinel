"""
Patient Model - Core patient data structures for MedGemma Sentinel
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class Gender(str, Enum):
    """Patient gender enumeration"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class BloodType(str, Enum):
    """Blood type enumeration"""
    A_POS = "A+"
    A_NEG = "A-"
    B_POS = "B+"
    B_NEG = "B-"
    AB_POS = "AB+"
    AB_NEG = "AB-"
    O_POS = "O+"
    O_NEG = "O-"


class Allergy(BaseModel):
    """Allergy information"""
    substance: str = Field(..., description="Allergenic substance")
    severity: str = Field(default="moderate", description="Severity: mild, moderate, severe")
    reaction: Optional[str] = Field(None, description="Type of reaction")
    confirmed: bool = Field(default=True)


class Medication(BaseModel):
    """Current medication information"""
    name: str = Field(..., description="Medication name")
    dosage: str = Field(..., description="Dosage (e.g., '500mg')")
    frequency: str = Field(..., description="Frequency (e.g., '2x/day')")
    route: str = Field(default="oral", description="Route of administration")
    start_date: Optional[date] = None
    prescriber: Optional[str] = None


class Condition(BaseModel):
    """Medical condition/diagnosis"""
    name: str = Field(..., description="Condition name")
    icd_code: Optional[str] = Field(None, description="ICD-10 code")
    diagnosed_date: Optional[date] = None
    status: str = Field(default="active", description="active, resolved, chronic")
    severity: Optional[str] = None


class Patient(BaseModel):
    """
    Core Patient model for MedGemma Sentinel
    Contains all essential patient information for clinical decision support
    """
    # Identification
    id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., description="Full name")
    date_of_birth: date = Field(..., description="Date of birth")
    gender: Gender = Field(..., description="Patient gender")
    
    # Physical characteristics
    height_cm: Optional[float] = Field(None, description="Height in centimeters")
    weight_kg: Optional[float] = Field(None, description="Weight in kilograms")
    blood_type: Optional[BloodType] = None
    
    # Contact & Location
    room: Optional[str] = Field(None, description="Hospital room number")
    bed: Optional[str] = Field(None, description="Bed identifier")
    
    # Medical history
    conditions: List[Condition] = Field(default_factory=list)
    allergies: List[Allergy] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    
    # Clinical context
    admission_date: Optional[datetime] = None
    admission_reason: Optional[str] = None
    attending_physician: Optional[str] = None
    
    # Risk factors
    risk_factors: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def age(self) -> int:
        """Calculate patient age"""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    @property
    def bmi(self) -> Optional[float]:
        """Calculate BMI if height and weight are available"""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m ** 2), 1)
        return None
    
    @property
    def active_conditions(self) -> List[Condition]:
        """Get list of active medical conditions"""
        return [c for c in self.conditions if c.status in ["active", "chronic"]]
    
    @property
    def severe_allergies(self) -> List[Allergy]:
        """Get list of severe allergies"""
        return [a for a in self.allergies if a.severity == "severe"]
    
    def get_summary(self) -> str:
        """Generate a clinical summary of the patient"""
        conditions_str = ", ".join([c.name for c in self.active_conditions]) or "Aucune"
        allergies_str = ", ".join([a.substance for a in self.allergies]) or "Aucune connue"
        meds_str = ", ".join([m.name for m in self.medications]) or "Aucun"
        
        return f"""
Patient: {self.name} ({self.gender.value}, {self.age} ans)
ID: {self.id} | Chambre: {self.room or 'N/A'} | Lit: {self.bed or 'N/A'}
Conditions actives: {conditions_str}
Allergies: {allergies_str}
Traitements en cours: {meds_str}
IMC: {self.bmi or 'N/A'}
Facteurs de risque: {', '.join(self.risk_factors) or 'Aucun'}
"""


class PatientHistory(BaseModel):
    """
    Patient history for GraphRAG storage
    Represents the complete medical timeline of a patient
    """
    patient_id: str
    entries: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Timeline markers
    first_visit: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    # Graph relationships
    related_patients: List[str] = Field(default_factory=list, description="Family/contacts")
    care_team: List[str] = Field(default_factory=list, description="Healthcare providers")
    
    def add_entry(self, entry_type: str, data: Dict[str, Any]) -> None:
        """Add a new entry to patient history"""
        entry = {
            "type": entry_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.entries.append(entry)
        self.last_update = datetime.now()
        if not self.first_visit:
            self.first_visit = datetime.now()
    
    def get_entries_by_type(self, entry_type: str) -> List[Dict[str, Any]]:
        """Get all entries of a specific type"""
        return [e for e in self.entries if e["type"] == entry_type]
    
    def get_recent_entries(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get entries from the last N days"""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        return [
            e for e in self.entries 
            if datetime.fromisoformat(e["timestamp"]).timestamp() > cutoff
        ]
