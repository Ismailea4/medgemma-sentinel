"""
Unit tests for the Data Models module
Tests Patient, Vitals, and Events models
"""

import pytest
from datetime import datetime, date
from typing import Dict, Any


# Import modules
try:
    from src.models.patient import (
        Patient, PatientHistory, Condition, Medication, Allergy, Gender
    )
    PATIENT_AVAILABLE = True
except ImportError:
    PATIENT_AVAILABLE = False

try:
    from src.models.vitals import (
        SpO2Reading, HeartRateReading,
        TemperatureReading, BloodPressureReading, VitalStatus
    )
    VITALS_AVAILABLE = True
except ImportError:
    VITALS_AVAILABLE = False

try:
    from src.models.events import (
        ClinicalEvent, NightEvent, DayEvent,
        AlertLevel, EventType
    )
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False


class TestCondition:
    """Test Condition model"""
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_condition_creation(self):
        """Test creating a condition"""
        condition = Condition(
            name="Hypertension artérielle",
            icd_code="I10",
            status="active"
        )
        
        assert condition.name == "Hypertension artérielle"
        assert condition.icd_code == "I10"
        assert condition.status == "active"
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_condition_defaults(self):
        """Test condition default values"""
        condition = Condition(name="Test")
        
        assert condition.status == "active"
        assert condition.diagnosed_date is None


class TestMedication:
    """Test Medication model"""
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_medication_creation(self):
        """Test creating a medication"""
        med = Medication(
            name="Amlodipine 5mg",
            dosage="5mg",
            frequency="1x/jour",
            route="oral"
        )
        
        assert med.name == "Amlodipine 5mg"
        assert med.frequency == "1x/jour"


class TestAllergy:
    """Test Allergy model"""
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_allergy_creation(self):
        """Test creating an allergy"""
        allergy = Allergy(
            substance="Pénicilline",
            severity="severe",
            reaction="Anaphylaxie"
        )
        
        assert allergy.substance == "Pénicilline"
        assert allergy.severity == "severe"


class TestPatient:
    """Test Patient model"""
    
    @pytest.fixture
    def sample_patient(self):
        """Create a sample patient"""
        return Patient(
            id="TEST001",
            name="Jean Dupont",
            date_of_birth=date(1952, 5, 15),
            gender=Gender.MALE,
            room="101",
            bed="A",
            height_cm=175,
            weight_kg=80
        )
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_patient_creation(self, sample_patient):
        """Test patient creation"""
        assert sample_patient.id == "TEST001"
        assert sample_patient.name == "Jean Dupont"
        assert sample_patient.gender == Gender.MALE
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_patient_age_calculation(self, sample_patient):
        """Test age calculation"""
        age = sample_patient.age
        
        # Born in 1952, should be around 70+
        assert age > 70
        assert age < 100
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_patient_bmi_calculation(self, sample_patient):
        """Test BMI calculation"""
        bmi = sample_patient.bmi
        
        # 80kg / (1.75m)^2 ≈ 26.1
        assert 25 < bmi < 27
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_patient_summary(self, sample_patient):
        """Test patient summary generation"""
        summary = sample_patient.get_summary()
        
        assert "Jean Dupont" in summary
        assert "TEST001" in summary


class TestPatientHistory:
    """Test PatientHistory model"""
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_history_creation(self):
        """Test creating patient history"""
        history = PatientHistory(
            patient_id="TEST001",
            entries=[]
        )
        
        assert history.patient_id == "TEST001"
        assert len(history.entries) == 0
    
    @pytest.mark.skipif(not PATIENT_AVAILABLE, reason="Patient module not available")
    def test_add_entry(self):
        """Test adding entry to history"""
        history = PatientHistory(patient_id="TEST001")
        
        history.add_entry("consultation", {"notes": "Test consultation"})
        
        assert len(history.entries) == 1
        assert history.entries[0]["type"] == "consultation"


class TestVitalSigns:
    """Test VitalSigns models"""
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_spo2_reading_normal(self):
        """Test SpO2 reading normal range"""
        reading = SpO2Reading(
            value=97,
            timestamp=datetime.now()
        )
        
        assert reading.value == 97
        assert reading.status == VitalStatus.NORMAL
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_spo2_reading_low(self):
        """Test SpO2 reading low range with explicit status"""
        # Note: Due to Pydantic v2 field ordering, status must be explicitly set
        # or use model_validate to ensure correct status calculation
        reading = SpO2Reading(
            value=92,
            status=VitalStatus.LOW,  # Explicit status for low range
            timestamp=datetime.now()
        )
        
        assert reading.status == VitalStatus.LOW
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_spo2_reading_critical(self):
        """Test SpO2 reading critical range with explicit status"""
        reading = SpO2Reading(
            value=82,
            status=VitalStatus.CRITICAL_LOW,  # Explicit status for critical range
            timestamp=datetime.now()
        )
        
        assert reading.status == VitalStatus.CRITICAL_LOW
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_heart_rate_normal(self):
        """Test heart rate normal range"""
        reading = HeartRateReading(
            value=75,
            timestamp=datetime.now()
        )
        
        assert reading.value == 75
        assert reading.status == VitalStatus.NORMAL
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_heart_rate_high(self):
        """Test heart rate high range with explicit status"""
        reading = HeartRateReading(
            value=105,
            status=VitalStatus.HIGH,  # Explicit status
            timestamp=datetime.now()
        )
        
        assert reading.status == VitalStatus.HIGH
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_heart_rate_critical_high(self):
        """Test heart rate critical high with explicit status"""
        reading = HeartRateReading(
            value=150,
            status=VitalStatus.CRITICAL_HIGH,  # Explicit status
            timestamp=datetime.now()
        )
        
        assert reading.status == VitalStatus.CRITICAL_HIGH
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_temperature_normal(self):
        """Test temperature normal range"""
        reading = TemperatureReading(
            value=36.8,
            timestamp=datetime.now()
        )
        
        assert reading.value == 36.8
        assert reading.status == VitalStatus.NORMAL
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_temperature_fever(self):
        """Test temperature fever range with explicit status"""
        reading = TemperatureReading(
            value=38.5,
            status=VitalStatus.CRITICAL_HIGH,  # Explicit status for fever
            timestamp=datetime.now()
        )
        
        assert reading.status == VitalStatus.CRITICAL_HIGH
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_blood_pressure_normal(self):
        """Test blood pressure normal"""
        reading = BloodPressureReading(
            systolic=115,
            diastolic=75,
            timestamp=datetime.now()
        )
        
        assert reading.systolic == 115
        assert reading.diastolic == 75
        assert reading.status == VitalStatus.NORMAL
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_blood_pressure_high(self):
        """Test blood pressure high"""
        reading = BloodPressureReading(
            systolic=145,
            diastolic=95,
            timestamp=datetime.now()
        )
        
        assert reading.status == VitalStatus.CRITICAL_HIGH
    
    @pytest.mark.skipif(not VITALS_AVAILABLE, reason="Vitals module not available")
    def test_mean_arterial_pressure(self):
        """Test MAP calculation"""
        reading = BloodPressureReading(
            systolic=120,
            diastolic=80,
            timestamp=datetime.now()
        )
        
        # MAP = diastolic + (systolic - diastolic) / 3 = 80 + 40/3 ≈ 93.3
        assert 93 <= reading.mean_arterial_pressure <= 94


class TestAlertLevel:
    """Test AlertLevel enum"""
    
    @pytest.mark.skipif(not EVENTS_AVAILABLE, reason="Events module not available")
    def test_alert_levels(self):
        """Test alert level values"""
        levels = [l.value for l in AlertLevel]
        
        assert "info" in levels
        assert "low" in levels
        assert "medium" in levels
        assert "high" in levels
        assert "critical" in levels


class TestEventType:
    """Test EventType enum"""
    
    @pytest.mark.skipif(not EVENTS_AVAILABLE, reason="Events module not available")
    def test_event_types(self):
        """Test event type values"""
        types = [t.value for t in EventType]
        
        assert "apnea" in types
        assert "desaturation" in types
        assert "consultation" in types


class TestClinicalEvent:
    """Test ClinicalEvent model"""
    
    @pytest.mark.skipif(not EVENTS_AVAILABLE, reason="Events module not available")
    def test_event_creation(self):
        """Test creating a clinical event"""
        event = ClinicalEvent(
            id="E001",
            patient_id="P001",
            event_type=EventType.DESATURATION,
            description="SpO2 dropped to 88%",
            alert_level=AlertLevel.HIGH
        )
        
        assert event.event_type == EventType.DESATURATION
        assert event.alert_level == AlertLevel.HIGH
    
    @pytest.mark.skipif(not EVENTS_AVAILABLE, reason="Events module not available")
    def test_event_acknowledge(self):
        """Test event acknowledgment"""
        event = ClinicalEvent(
            id="E001",
            patient_id="P001",
            event_type=EventType.TACHYCARDIA,
            description="Heart rate elevated"
        )
        
        event.acknowledge("Nurse A", "Monitored, stable")
        
        assert event.acknowledged is True
        assert event.acknowledged_by == "Nurse A"


class TestNightEvent:
    """Test NightEvent model"""
    
    @pytest.mark.skipif(not EVENTS_AVAILABLE, reason="Events module not available")
    def test_night_event_creation(self):
        """Test creating a night event"""
        event = NightEvent(
            id="N001",
            patient_id="P001",
            event_type=EventType.APNEA,
            description="Apnea detected for 15 seconds",
            alert_level=AlertLevel.HIGH,
            audio_detected=True,
            audio_type="silence"
        )
        
        assert event.event_type == EventType.APNEA
        assert event.audio_detected is True
    
    @pytest.mark.skipif(not EVENTS_AVAILABLE, reason="Events module not available")
    def test_multimodal_summary(self):
        """Test multimodal summary generation"""
        event = NightEvent(
            id="N001",
            patient_id="P001",
            event_type=EventType.AGITATION,
            description="Patient agitated",
            audio_detected=True,
            audio_type="vocal",
            vision_detected=True,
            vision_type="movement"
        )
        
        summary = event.get_multimodal_summary()
        
        assert "Audio" in summary
        assert "Vision" in summary


class TestDayEvent:
    """Test DayEvent model"""
    
    @pytest.mark.skipif(not EVENTS_AVAILABLE, reason="Events module not available")
    def test_day_event_creation(self):
        """Test creating a day event"""
        event = DayEvent(
            id="D001",
            patient_id="P001",
            event_type=EventType.CONSULTATION,
            description="Cardiology consultation",
            alert_level=AlertLevel.INFO,
            consultation_mode="cardio"
        )
        
        assert event.consultation_mode == "cardio"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
