"""
Unit tests for the Orchestration module
Tests LangGraph state machine and workflow nodes
"""

import pytest
from datetime import datetime
from typing import Dict, Any


# Import modules - handle missing dependencies gracefully
try:
    from src.orchestration.state import (
        SentinelState,
        WorkflowPhase,
        SteeringMode,
        NightData,
        DayData,
        ReportData
    )
    STATE_AVAILABLE = True
except ImportError:
    STATE_AVAILABLE = False

try:
    from src.orchestration.nodes import NightNode, Rap1Node, DayNode, Rap2Node
    NODES_AVAILABLE = True
except ImportError:
    NODES_AVAILABLE = False

try:
    from src.orchestration.graph import MedGemmaSentinelGraph
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


class TestWorkflowPhase:
    """Test WorkflowPhase enum"""
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_phase_values(self):
        """Test that all expected phases exist"""
        assert WorkflowPhase.IDLE.value == "idle"
        assert WorkflowPhase.NIGHT.value == "night"
        assert WorkflowPhase.RAP1.value == "rap1"
        assert WorkflowPhase.DAY.value == "day"
        assert WorkflowPhase.RAP2.value == "rap2"
        assert WorkflowPhase.COMPLETED.value == "completed"
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_phase_count(self):
        """Test number of phases"""
        phases = list(WorkflowPhase)
        assert len(phases) == 6


class TestSteeringMode:
    """Test SteeringMode enum"""
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_steering_modes(self):
        """Test that all steering modes exist"""
        modes = [m.value for m in SteeringMode]
        assert "night_surveillance" in modes
        assert "specialist_virtual" in modes
        assert "triage_priority" in modes
        assert "longitudinal" in modes


class TestSentinelState:
    """Test SentinelState model"""
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_default_state(self):
        """Test default state initialization"""
        state = SentinelState()
        
        assert state.phase == WorkflowPhase.IDLE
        assert state.patient_id == ""
        assert state.messages == []
        assert state.errors == []
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_state_with_patient(self):
        """Test state with patient data"""
        state = SentinelState(
            patient_id="TEST001",
            patient_context={"name": "Test Patient", "age": 65}
        )
        
        assert state.patient_id == "TEST001"
        assert state.patient_context["name"] == "Test Patient"


class TestNightData:
    """Test NightData model"""
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_night_data_defaults(self):
        """Test default night data"""
        night_data = NightData()
        
        assert night_data.vitals_readings == []
        assert night_data.audio_events == []
        assert night_data.vision_events == []
        assert night_data.events == []
        assert night_data.alerts_triggered == 0
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_night_data_with_values(self):
        """Test night data with values"""
        night_data = NightData(
            events=[{"type": "event1"}, {"type": "event2"}],
            alerts_triggered=2,
            critical_alerts=1
        )
        
        assert len(night_data.events) == 2
        assert night_data.alerts_triggered == 2
        assert night_data.critical_alerts == 1


class TestDayData:
    """Test DayData model"""
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_day_data_defaults(self):
        """Test default day data"""
        day_data = DayData()
        
        assert day_data.consultation_mode == "general"
        assert day_data.symptoms == []
        assert day_data.differential_diagnosis == []
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_day_data_with_values(self):
        """Test day data with consultation info"""
        day_data = DayData(
            consultation_mode="cardio",
            symptoms=["chest pain", "dyspnea"],
            presenting_complaint="Chest pain"
        )
        
        assert day_data.consultation_mode == "cardio"
        assert len(day_data.symptoms) == 2


class TestReportData:
    """Test ReportData model"""
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_report_data_defaults(self):
        """Test default report data"""
        report = ReportData()
        
        assert report.report_type == "night"
        assert report.title == "Rapport Clinique"
        assert report.sections == []
    
    @pytest.mark.skipif(not STATE_AVAILABLE, reason="State module not available")
    def test_report_data_with_content(self):
        """Test report data with content"""
        report = ReportData(
            report_type="consultation",
            title="Consultation Report",
            summary="Patient consulted for chest pain",
            sections=[{"title": "Findings", "content": "Normal"}]
        )
        
        assert report.report_type == "consultation"
        assert len(report.sections) == 1


class TestNightNode:
    """Test NightNode class"""
    
    @pytest.mark.skipif(not NODES_AVAILABLE, reason="Nodes not available")
    def test_night_node_initialization(self):
        """Test NightNode creation"""
        node = NightNode()
        assert node.name == "NIGHT"
    
    @pytest.mark.skipif(not NODES_AVAILABLE, reason="Nodes not available")
    def test_night_node_has_execute(self):
        """Test NightNode has execute method"""
        node = NightNode()
        assert hasattr(node, "execute")


class TestRap1Node:
    """Test Rap1Node class"""
    
    @pytest.mark.skipif(not NODES_AVAILABLE, reason="Nodes not available")
    def test_rap1_node_initialization(self):
        """Test Rap1Node creation"""
        node = Rap1Node()
        assert node.name == "RAP1"
    
    @pytest.mark.skipif(not NODES_AVAILABLE, reason="Nodes not available")
    def test_rap1_node_has_execute(self):
        """Test Rap1Node has execute method"""
        node = Rap1Node()
        assert hasattr(node, "execute")


class TestDayNode:
    """Test DayNode class"""
    
    @pytest.mark.skipif(not NODES_AVAILABLE, reason="Nodes not available")
    def test_day_node_initialization(self):
        """Test DayNode creation"""
        node = DayNode()
        assert node.name == "DAY"
    
    @pytest.mark.skipif(not NODES_AVAILABLE, reason="Nodes not available")
    def test_day_node_has_execute(self):
        """Test DayNode has execute method"""
        node = DayNode()
        assert hasattr(node, "execute")


class TestRap2Node:
    """Test Rap2Node class"""
    
    @pytest.mark.skipif(not NODES_AVAILABLE, reason="Nodes not available")
    def test_rap2_node_initialization(self):
        """Test Rap2Node creation"""
        node = Rap2Node()
        assert node.name == "RAP2"
    
    @pytest.mark.skipif(not NODES_AVAILABLE, reason="Nodes not available")
    def test_rap2_node_has_execute(self):
        """Test Rap2Node has execute method"""
        node = Rap2Node()
        assert hasattr(node, "execute")


class TestMedGemmaSentinelGraph:
    """Test the main graph class"""
    
    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Graph not available")
    def test_graph_initialization(self):
        """Test graph creation"""
        graph = MedGemmaSentinelGraph()
        assert graph is not None
    
    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Graph not available")
    def test_graph_visualization(self):
        """Test graph visualization output"""
        graph = MedGemmaSentinelGraph()
        viz = graph.get_graph_visualization()
        
        # Should contain workflow elements (case insensitive check)
        viz_lower = viz.lower()
        assert "night" in viz_lower
        assert "day" in viz_lower
    
    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Graph not available")
    def test_graph_has_run_method(self):
        """Test graph has run method"""
        graph = MedGemmaSentinelGraph()
        assert hasattr(graph, "run")
    
    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Graph not available")
    def test_graph_run_night_only(self):
        """Test running night-only workflow"""
        graph = MedGemmaSentinelGraph()
        
        result = graph.run_night_only(
            patient_id="TEST001",
            patient_context={"name": "Test"},
            vitals_input=[{"spo2": 95}],
            audio_input=[],
            vision_input=[]
        )
        
        assert "night_data" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
