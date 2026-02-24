"""
Tests for MedGemma Engine Integration
"""

import pytest
from src.reporting.medgemma_engine import MedGemmaEngine
from src.reporting.prompts import MedGemmaReportGenerator


class TestMedGemmaEngine:
    """Test MedGemma Engine initialization and methods"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance"""
        return MedGemmaEngine()
    
    def test_engine_initialization(self, engine):
        """Test engine initializes without error"""
        assert engine is not None
        assert engine.temperature == 0.3
        assert engine.max_tokens is None  # Unlimited by default
    
    def test_engine_status(self, engine):
        """Test engine status reporting"""
        status = engine.get_status()
        
        assert "loaded" in status
        assert "mode" in status
        assert status["mode"] in ["inference", "simulation", "server", "huggingface", "llama-cpp-python"]
    
    def test_night_report_generation(self, engine):
        """Test night report generation (with fallback)"""
        report = engine.generate_night_report(
            patient_context="Test patient",
            night_summary="Calm night",
            events=[
                {"type": "desaturation", "severity": "high", "description": "SpO2 low"}
            ]
        )
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "surveillance" in report.lower() or "nuit" in report.lower()
    
    def test_day_report_generation(self, engine):
        """Test day report generation (with fallback)"""
        report = engine.generate_day_report(
            patient_context="Test patient",
            night_context="Previous night summary",
            consultation_data={"symptoms": ["chest pain"]},
            specialty="cardio"
        )
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "consultation" in report.lower() or "cardio" in report.lower()
    
    def test_symptom_analysis(self, engine):
        """Test symptom analysis (with fallback)"""
        result = engine.analyze_symptoms(
            symptoms=["fever", "cough"],
            patient_context="65-year-old male"
        )
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "analysis" in result


class TestMedGemmaReportGenerator:
    """Test integrated report generator"""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance"""
        return MedGemmaReportGenerator()
    
    def test_generator_initialization(self, generator):
        """Test generator initializes"""
        assert generator is not None
        assert generator.engine is not None
    
    def test_engine_status_check(self, generator):
        """Test can check engine status"""
        status = generator.get_engine_status()
        
        assert isinstance(status, dict)
        assert "mode" in status
    
    def test_night_report_with_generator(self, generator):
        """Test night report generation through generator"""
        patient_context = {
            "name": "Test Patient",
            "id": "TEST001",
        }
        
        night_data = {
            "total_events": 2,
            "events": [
                {"type": "apnea", "severity": "critical", "description": "Apnée détectée"}
            ]
        }
        
        report = generator.generate_night_report(
            patient_context=patient_context,
            night_data=night_data,
        )
        
        assert isinstance(report, str)
        assert len(report) > 100
    
    def test_day_report_with_generator(self, generator):
        """Test day report generation through generator"""
        patient_context = {
            "name": "Test Patient",
            "id": "TEST001",
        }
        
        day_data = {
            "symptoms": ["chest pain"],
            "consultation_mode": "cardio"
        }
        
        report = generator.generate_day_report(
            patient_context=patient_context,
            night_context="Calm night",
            day_data=day_data,
            specialty="cardio"
        )
        
        assert isinstance(report, str)
        assert len(report) > 100


class TestMedGemmaIntegration:
    """Integration tests for complete workflow"""
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow with simulated model"""
        from examples.demo_with_medgemma import (
            create_demo_patient,
            simulate_night_events
        )
        
        # Create patient
        patient = create_demo_patient()
        assert patient.id == "MEDGEMMA_001"
        assert patient.age >= 73  # Born May 1952, current date Feb 2026
        
        # Simulate events
        events, vitals = simulate_night_events()
        assert len(events) == 4
        assert len(vitals) == 4
    
    def test_report_generator_with_context(self):
        """Test report generation with full context"""
        generator = MedGemmaReportGenerator()
        
        patient_context = {
            "name": "Jean Camara",
            "id": "MEDGEMMA_001",
        }
        
        night_data = {
            "total_events": 4,
            "critical_alerts": 1,
            "events": [
                {
                    "type": "desaturation",
                    "severity": "high",
                    "description": "SpO2: 86%",
                    "timestamp": "2026-02-11T00:00:00"
                },
                {
                    "type": "apnea",
                    "severity": "critical",
                    "description": "Apnée détectée (11s)",
                    "timestamp": "2026-02-11T05:22:00"
                }
            ]
        }
        
        report = generator.generate_night_report(
            patient_context=patient_context,
            night_data=night_data,
        )
        
        assert isinstance(report, str)
        assert len(report) > 200
        # Should contain some clinical content
        assert any(word in report.lower() for word in ["patient", "alerte", "événement", "nuit"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
