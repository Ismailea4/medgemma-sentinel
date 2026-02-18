"""
Vital Signs Models - Real-time patient monitoring data
"""

from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class VitalStatus(str, Enum):
    """Status of vital sign reading"""
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"


class VitalReading(BaseModel):
    """
    Single vital sign reading with metadata
    """
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = Field(default="sensor", description="Data source: sensor, manual, derived")
    quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Signal quality 0-1")
    status: VitalStatus = Field(default=VitalStatus.NORMAL)
    
    def is_critical(self) -> bool:
        """Check if reading is in critical range"""
        return self.status in [VitalStatus.CRITICAL_LOW, VitalStatus.CRITICAL_HIGH]


class SpO2Reading(VitalReading):
    """Oxygen saturation reading"""
    unit: str = Field(default="%")
    perfusion_index: Optional[float] = Field(None, description="PI value")
    
    @field_validator('status', mode='before')
    @classmethod
    def determine_status(cls, v, info):
        if 'value' in info.data:
            value = info.data['value']
            if value >= 95:
                return VitalStatus.NORMAL
            elif value >= 90:
                return VitalStatus.LOW
            elif value >= 85:
                return VitalStatus.CRITICAL_LOW
            else:
                return VitalStatus.CRITICAL_LOW
        return v


class HeartRateReading(VitalReading):
    """Heart rate reading"""
    unit: str = Field(default="bpm")
    rhythm: str = Field(default="regular", description="regular, irregular, arrhythmia")
    
    @field_validator('status', mode='before')
    @classmethod
    def determine_status(cls, v, info):
        if 'value' in info.data:
            value = info.data['value']
            if 60 <= value <= 100:
                return VitalStatus.NORMAL
            elif 50 <= value < 60 or 100 < value <= 110:
                return VitalStatus.LOW if value < 60 else VitalStatus.HIGH
            elif value < 50:
                return VitalStatus.CRITICAL_LOW
            else:
                return VitalStatus.CRITICAL_HIGH
        return v


class TemperatureReading(VitalReading):
    """Body temperature reading"""
    unit: str = Field(default="°C")
    site: str = Field(default="tympanic", description="Measurement site")
    
    @field_validator('status', mode='before')
    @classmethod
    def determine_status(cls, v, info):
        if 'value' in info.data:
            value = info.data['value']
            if 36.1 <= value <= 37.2:
                return VitalStatus.NORMAL
            elif 35.0 <= value < 36.1:
                return VitalStatus.LOW
            elif 37.2 < value <= 38.0:
                return VitalStatus.HIGH
            elif value < 35.0:
                return VitalStatus.CRITICAL_LOW
            else:
                return VitalStatus.CRITICAL_HIGH
        return v


class BloodPressureReading(BaseModel):
    """Blood pressure reading (systolic/diastolic)"""
    systolic: float = Field(..., description="Systolic pressure mmHg")
    diastolic: float = Field(..., description="Diastolic pressure mmHg")
    unit: str = Field(default="mmHg")
    timestamp: datetime = Field(default_factory=datetime.now)
    position: str = Field(default="sitting", description="Patient position during measurement")
    arm: str = Field(default="left", description="Measurement arm")
    
    @property
    def mean_arterial_pressure(self) -> float:
        """Calculate Mean Arterial Pressure (MAP)"""
        return round(self.diastolic + (self.systolic - self.diastolic) / 3, 1)
    
    @property
    def pulse_pressure(self) -> float:
        """Calculate pulse pressure"""
        return self.systolic - self.diastolic
    
    @property
    def status(self) -> VitalStatus:
        """Determine BP status based on clinical guidelines"""
        if self.systolic < 90 or self.diastolic < 60:
            return VitalStatus.CRITICAL_LOW
        elif self.systolic < 120 and self.diastolic < 80:
            return VitalStatus.NORMAL
        elif self.systolic < 140 or self.diastolic < 90:
            return VitalStatus.HIGH
        else:
            return VitalStatus.CRITICAL_HIGH
    
    def __str__(self) -> str:
        return f"{int(self.systolic)}/{int(self.diastolic)} mmHg"


class RespiratoryReading(VitalReading):
    """Respiratory rate reading"""
    unit: str = Field(default="breaths/min")
    pattern: str = Field(default="regular", description="Breathing pattern")
    effort: str = Field(default="normal", description="Respiratory effort: normal, labored, shallow")
    
    @field_validator('status', mode='before')
    @classmethod
    def determine_status(cls, v, info):
        if 'value' in info.data:
            value = info.data['value']
            if 12 <= value <= 20:
                return VitalStatus.NORMAL
            elif 10 <= value < 12 or 20 < value <= 24:
                return VitalStatus.LOW if value < 12 else VitalStatus.HIGH
            elif value < 10:
                return VitalStatus.CRITICAL_LOW
            else:
                return VitalStatus.CRITICAL_HIGH
        return v


class VitalSigns(BaseModel):
    """
    Complete vital signs snapshot for a patient
    Aggregates all vital parameters at a given moment
    """
    patient_id: str = Field(..., description="Patient identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Core vitals
    spo2: Optional[SpO2Reading] = None
    heart_rate: Optional[HeartRateReading] = None
    temperature: Optional[TemperatureReading] = None
    blood_pressure: Optional[BloodPressureReading] = None
    respiratory_rate: Optional[RespiratoryReading] = None
    
    # Additional parameters
    consciousness_level: Optional[str] = Field(None, description="AVPU scale: Alert, Voice, Pain, Unresponsive")
    pain_score: Optional[int] = Field(None, ge=0, le=10, description="Pain scale 0-10")
    
    # Metadata
    recorded_by: str = Field(default="system", description="Who recorded the vitals")
    notes: Optional[str] = None
    
    @property
    def has_critical(self) -> bool:
        """Check if any vital is in critical range"""
        vitals = [self.spo2, self.heart_rate, self.temperature, self.respiratory_rate]
        for v in vitals:
            if v and v.is_critical():
                return True
        if self.blood_pressure and self.blood_pressure.status in [
            VitalStatus.CRITICAL_LOW, VitalStatus.CRITICAL_HIGH
        ]:
            return True
        return False
    
    @property
    def critical_vitals(self) -> List[str]:
        """Get list of vitals in critical range"""
        critical = []
        if self.spo2 and self.spo2.is_critical():
            critical.append(f"SpO2: {self.spo2.value}%")
        if self.heart_rate and self.heart_rate.is_critical():
            critical.append(f"FC: {self.heart_rate.value} bpm")
        if self.temperature and self.temperature.is_critical():
            critical.append(f"T°: {self.temperature.value}°C")
        if self.respiratory_rate and self.respiratory_rate.is_critical():
            critical.append(f"FR: {self.respiratory_rate.value}/min")
        if self.blood_pressure and self.blood_pressure.status in [
            VitalStatus.CRITICAL_LOW, VitalStatus.CRITICAL_HIGH
        ]:
            critical.append(f"PA: {self.blood_pressure}")
        return critical
    
    def get_summary(self) -> str:
        """Generate a text summary of vital signs"""
        parts = [f"Constantes vitales - {self.timestamp.strftime('%d/%m/%Y %H:%M')}"]
        
        if self.spo2:
            parts.append(f"  SpO2: {self.spo2.value}% ({self.spo2.status.value})")
        if self.heart_rate:
            parts.append(f"  FC: {self.heart_rate.value} bpm ({self.heart_rate.status.value})")
        if self.blood_pressure:
            parts.append(f"  PA: {self.blood_pressure} ({self.blood_pressure.status.value})")
        if self.temperature:
            parts.append(f"  T°: {self.temperature.value}°C ({self.temperature.status.value})")
        if self.respiratory_rate:
            parts.append(f"  FR: {self.respiratory_rate.value}/min ({self.respiratory_rate.status.value})")
        if self.consciousness_level:
            parts.append(f"  Conscience: {self.consciousness_level}")
        if self.pain_score is not None:
            parts.append(f"  Douleur: {self.pain_score}/10")
            
        if self.has_critical:
            parts.append(f"  ⚠️ CRITIQUE: {', '.join(self.critical_vitals)}")
        
        return "\n".join(parts)
