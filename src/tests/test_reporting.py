"""
Unit tests for the Reporting module
Tests MedGemmaPrompts, ReportTemplates, and PDFReportGenerator
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime


# Import modules - handle missing dependencies gracefully
try:
    from src.reporting.prompts import (
        MedGemmaPrompts,
        PromptType,
        SteeringPrompt
    )
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False

try:
    from src.reporting.templates import (
        NightReportTemplate,
        DayReportTemplate
    )
    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False

try:
    from src.reporting.pdf_generator import PDFReportGenerator
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class TestPromptType:
    """Test PromptType enum"""
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompt_types(self):
        """Test that expected prompt types exist"""
        types = [t.value for t in PromptType]
        
        assert "night_surveillance" in types
        assert "day_consultation" in types
        assert "cardio_analysis" in types


class TestSteeringPrompt:
    """Test SteeringPrompt dataclass"""
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompt_structure(self):
        """Test prompt has required fields"""
        prompt = MedGemmaPrompts.get_prompt(PromptType.NIGHT_SURVEILLANCE)
        
        assert hasattr(prompt, "name")
        assert hasattr(prompt, "prompt_type")
        assert hasattr(prompt, "system_prompt")
        assert hasattr(prompt, "temperature")
        assert hasattr(prompt, "max_tokens")
        assert hasattr(prompt, "output_sections")


class TestMedGemmaPrompts:
    """Test MedGemmaPrompts class"""
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_get_all_prompts(self):
        """Test listing all prompts"""
        prompts = MedGemmaPrompts.list_prompts()
        
        assert len(prompts) > 0
        assert all("name" in p for p in prompts)
        assert all("type" in p for p in prompts)
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_get_night_prompt(self):
        """Test getting night surveillance prompt"""
        prompt = MedGemmaPrompts.get_prompt(PromptType.NIGHT_SURVEILLANCE)
        
        assert prompt is not None
        assert "surveillance" in prompt.name.lower() or "nuit" in prompt.name.lower()
        assert len(prompt.system_prompt) > 100
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_get_day_prompt(self):
        """Test getting day consultation prompt"""
        prompt = MedGemmaPrompts.get_prompt(PromptType.DAY_CONSULTATION)
        
        assert prompt is not None
        assert len(prompt.system_prompt) > 100
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_get_cardio_prompt(self):
        """Test getting cardio specialty prompt"""
        prompt = MedGemmaPrompts.get_prompt(PromptType.CARDIO_ANALYSIS)
        
        assert prompt is not None
        # Should contain cardiology-specific content
        prompt_lower = prompt.system_prompt.lower()
        assert "cardio" in prompt_lower or "cœur" in prompt_lower or "cardiologie" in prompt_lower
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_output_sections(self):
        """Test that prompts have defined output sections"""
        prompt = MedGemmaPrompts.get_prompt(PromptType.NIGHT_SURVEILLANCE)
        
        assert prompt.output_sections is not None
        assert len(prompt.output_sections) > 0
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompt_temperature_range(self):
        """Test that temperature is in valid range"""
        prompt = MedGemmaPrompts.get_prompt(PromptType.NIGHT_SURVEILLANCE)
        assert 0.0 <= prompt.temperature <= 1.0


class TestNightReportTemplate:
    """Test NightReportTemplate class"""
    
    @pytest.fixture
    def sample_night_data(self):
        """Sample data for night report"""
        return {
            "patient_id": "TEST001",
            "patient_name": "Jean Dupont",
            "room": "101",
            "summary": "Nuit calme avec un épisode de désaturation",
            "events": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "desaturation",
                    "description": "SpO2 à 88%",
                    "severity": "high"
                }
            ],
            "night_data": {
                "start_time": "22:00",
                "end_time": "06:00",
                "total_events": 1,
                "total_alerts": 1
            },
            "vitals_summary": {
                "SpO2": {"min": 88, "max": 98, "avg": 95, "anomalies": 1}
            },
            "recommendations": ["Surveillance SpO2", "ECG de contrôle"]
        }
    
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="Templates not available")
    def test_template_initialization(self):
        """Test template creation"""
        template = NightReportTemplate()
        assert template is not None
    
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="Templates not available")
    def test_render_markdown(self, sample_night_data):
        """Test rendering markdown"""
        template = NightReportTemplate()
        md = template.render_markdown(sample_night_data)
        
        assert isinstance(md, str)
        assert "Jean Dupont" in md
    
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="Templates not available")
    def test_render_html(self, sample_night_data):
        """Test rendering HTML"""
        template = NightReportTemplate()
        html = template.render_html(sample_night_data)
        
        assert isinstance(html, str)
        assert "html" in html.lower()
        assert "Jean Dupont" in html


class TestDayReportTemplate:
    """Test DayReportTemplate class"""
    
    @pytest.fixture
    def sample_day_data(self):
        """Sample data for day report"""
        return {
            "patient_id": "TEST001",
            "patient_name": "Jean Dupont",
            "day_data": {
                "consultation_mode": "cardio",
                "symptoms": ["Douleur thoracique", "Dyspnée"],
                "exam_findings": {"Auscultation": "BDC réguliers"},
                "differential_diagnosis": ["Angine stable", "Douleur pariétale"],
                "recommendations": ["ECG", "Dosage troponines"]
            },
            "night_context": "Nuit calme"
        }
    
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="Templates not available")
    def test_template_initialization(self):
        """Test template creation"""
        template = DayReportTemplate()
        assert template is not None
    
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="Templates not available")
    def test_render_markdown(self, sample_day_data):
        """Test rendering markdown"""
        template = DayReportTemplate()
        md = template.render_markdown(sample_day_data)
        
        assert isinstance(md, str)
        assert "Jean Dupont" in md
    
    @pytest.mark.skipif(not TEMPLATES_AVAILABLE, reason="Templates not available")
    def test_render_html(self, sample_day_data):
        """Test rendering HTML"""
        template = DayReportTemplate()
        html = template.render_html(sample_day_data)
        
        assert isinstance(html, str)
        assert "html" in html.lower()


class TestPDFReportGenerator:
    """Test PDFReportGenerator class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for output"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def generator(self, temp_dir):
        """Create a generator with temp output"""
        return PDFReportGenerator(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_report_data(self):
        """Sample report data"""
        return {
            "patient_id": "TEST001",
            "patient_name": "Jean Dupont",
            "room": "101",
            "summary": "Test report summary",
            "events": [],
            "night_data": {},
            "day_data": {},
            "recommendations": ["Test recommendation"]
        }
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="PDF generator not available")
    def test_initialization(self, generator):
        """Test generator creation"""
        assert generator is not None
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="PDF generator not available")
    def test_generate_night_report(self, generator, sample_report_data, temp_dir):
        """Test generating night PDF report"""
        pdf_path = generator.generate_night_report(sample_report_data)
        
        # Should return a path
        assert pdf_path is not None
        assert Path(pdf_path).exists()
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="PDF generator not available")
    def test_generate_day_report(self, generator, sample_report_data, temp_dir):
        """Test generating day PDF report"""
        pdf_path = generator.generate_day_report(sample_report_data)
        
        assert pdf_path is not None
        assert Path(pdf_path).exists()
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="PDF generator not available")
    def test_output_directory_creation(self, temp_dir):
        """Test that output directory is created"""
        new_dir = Path(temp_dir) / "new_reports"
        generator = PDFReportGenerator(output_dir=str(new_dir))
        
        assert new_dir.exists()


class TestIntegration:
    """Integration tests for reporting module"""
    
    @pytest.mark.skipif(not all([PROMPTS_AVAILABLE, TEMPLATES_AVAILABLE, PDF_AVAILABLE]),
                        reason="Not all modules available")
    def test_full_report_workflow(self):
        """Test complete workflow: prompt -> template -> PDF"""
        # 1. Get appropriate prompt
        prompt = MedGemmaPrompts.get_prompt(PromptType.NIGHT_SURVEILLANCE)
        assert prompt is not None
        
        # 2. Format input (simulating LLM output)
        report_data = {
            "patient_id": "INT001",
            "patient_name": "Test Integration",
            "room": "001",
            "summary": "Nuit calme - intégration test",
            "events": [],
            "night_data": {"total_events": 0},
            "vitals_summary": {},
            "recommendations": ["Continue surveillance"]
        }
        
        # 3. Generate template
        template = NightReportTemplate()
        md = template.render_markdown(report_data)
        assert len(md) > 0
        
        # 4. Generate PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = PDFReportGenerator(output_dir=temp_dir)
            pdf_path = generator.generate_night_report(report_data)
            assert pdf_path is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
