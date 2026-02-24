"""
Unit tests for the Memory module
Tests GraphRAG, LocalGraphStore, and GraphRetriever
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


# Import modules - handle missing dependencies gracefully
try:
    from src.memory.patient_graph import PatientGraphRAG, NodeType, RelationType
    GRAPH_RAG_AVAILABLE = True
except ImportError:
    GRAPH_RAG_AVAILABLE = False

try:
    from src.memory.graph_store import LocalGraphStore
    STORE_AVAILABLE = True
except ImportError:
    STORE_AVAILABLE = False

try:
    from src.memory.retriever import GraphRetriever, RetrievalMode, RetrievalResult
    RETRIEVER_AVAILABLE = True
except ImportError:
    RETRIEVER_AVAILABLE = False


class TestNodeType:
    """Test NodeType enum"""
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_node_types(self):
        """Test that all node types exist"""
        types = [t.value for t in NodeType]
        assert "patient" in types
        assert "condition" in types
        assert "medication" in types


class TestRelationType:
    """Test RelationType enum"""
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_relation_types(self):
        """Test that relation types exist"""
        types = [t.value for t in RelationType]
        assert "has_condition" in types
        assert "has_medication" in types
        assert "has_allergy" in types


class TestPatientGraphRAG:
    """Test PatientGraphRAG class"""
    
    @pytest.fixture
    def graph_rag(self):
        """Create a fresh GraphRAG instance"""
        return PatientGraphRAG()
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing"""
        return {
            "patient_id": "TEST001",
            "name": "Jean Dupont",
            "age": 72,
            "conditions": ["Hypertension artérielle", "Diabète type 2"],
            "medications": ["Amlodipine 5mg", "Metformine 500mg"],
            "allergies": ["Pénicilline"],
            "risk_factors": ["Age > 65", "HTA"],
            "room": "101"
        }
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_initialization(self, graph_rag):
        """Test GraphRAG initialization"""
        assert graph_rag is not None
        stats = graph_rag.get_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_add_patient(self, graph_rag, sample_patient_data):
        """Test adding a patient"""
        patient_node_id = graph_rag.add_patient(**sample_patient_data)
        
        assert patient_node_id is not None
        
        stats = graph_rag.get_statistics()
        assert stats["total_patients"] == 1
        assert stats["total_nodes"] > 1
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_get_patient_context(self, graph_rag, sample_patient_data):
        """Test retrieving patient context"""
        graph_rag.add_patient(**sample_patient_data)
        
        context = graph_rag.get_patient_context("TEST001")
        
        assert context is not None
        assert context.get("name") == "Jean Dupont"
        assert context.get("age") == 72
        assert "conditions" in context
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_get_patient_summary(self, graph_rag, sample_patient_data):
        """Test generating patient summary"""
        graph_rag.add_patient(**sample_patient_data)
        
        summary = graph_rag.get_patient_summary("TEST001")
        
        assert isinstance(summary, str)
        assert "Jean Dupont" in summary
        assert "72" in summary
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_add_clinical_event(self, graph_rag, sample_patient_data):
        """Test adding a clinical event"""
        graph_rag.add_patient(**sample_patient_data)
        
        event_id = graph_rag.add_clinical_event(
            patient_id="TEST001",
            event_type="desaturation",
            description="SpO2 dropped to 87%",
            severity="high"
        )
        
        assert event_id is not None
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_add_consultation(self, graph_rag, sample_patient_data):
        """Test adding a consultation"""
        graph_rag.add_patient(**sample_patient_data)
        
        consult_id = graph_rag.add_consultation(
            patient_id="TEST001",
            consultation_type="cardio",
            presenting_complaint="Douleur thoracique",
            diagnosis="Angine stable",
            provider="Dr. Martin"
        )
        
        assert consult_id is not None
    
    @pytest.mark.skipif(not GRAPH_RAG_AVAILABLE, reason="GraphRAG not available")
    def test_get_statistics(self, graph_rag, sample_patient_data):
        """Test getting graph statistics"""
        graph_rag.add_patient(**sample_patient_data)
        
        stats = graph_rag.get_statistics()
        
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "total_patients" in stats
        assert stats["total_patients"] == 1


class TestLocalGraphStore:
    """Test LocalGraphStore class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for storage"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def store(self, temp_dir):
        """Create a store with temporary storage"""
        return LocalGraphStore(base_dir=temp_dir)
    
    @pytest.mark.skipif(not STORE_AVAILABLE, reason="Store not available")
    def test_initialization(self, store):
        """Test store initialization"""
        assert store is not None
    
    @pytest.mark.skipif(not STORE_AVAILABLE, reason="Store not available")
    def test_save_and_load_node(self, store):
        """Test saving and loading a node"""
        node_data = {"name": "Test Patient", "age": 65, "node_type": "patient"}
        store.save_node("patient_P001", node_data)
        
        loaded = store.load_node("patient_P001")
        
        assert loaded is not None
        assert loaded["name"] == "Test Patient"
    
    @pytest.mark.skipif(not STORE_AVAILABLE, reason="Store not available")
    def test_save_and_load_edge(self, store):
        """Test saving and loading edges"""
        store.save_node("patient_P001", {"name": "Patient", "node_type": "patient"})
        store.save_node("condition_C001", {"name": "Hypertension", "node_type": "condition"})
        
        store.save_edge("patient_P001", "condition_C001", "has_condition")
        
        # get_edges returns all edges, optionally filtered by node
        edges = store.get_edges(source_id="patient_P001")
        assert len(edges) > 0


class TestGraphRetriever:
    """Test GraphRetriever class"""
    
    @pytest.fixture
    def graph_with_data(self):
        """Create a GraphRAG with sample data"""
        graph = PatientGraphRAG()
        
        graph.add_patient(
            patient_id="TEST001",
            name="Jean Dupont",
            age=72,
            conditions=["Hypertension", "Diabète"],
            medications=["Amlodipine"],
            allergies=[],
            risk_factors=["Age > 65"],
            room="101"
        )
        
        graph.add_clinical_event(
            patient_id="TEST001",
            event_type="desaturation",
            description="Épisode de désaturation nocturne",
            severity="high"
        )
        
        return graph
    
    @pytest.fixture
    def retriever(self, graph_with_data):
        """Create a retriever with the test graph"""
        return GraphRetriever(graph_with_data)
    
    @pytest.mark.skipif(not RETRIEVER_AVAILABLE or not GRAPH_RAG_AVAILABLE, reason="Modules not available")
    def test_initialization(self, retriever):
        """Test retriever initialization"""
        assert retriever is not None
    
    @pytest.mark.skipif(not RETRIEVER_AVAILABLE or not GRAPH_RAG_AVAILABLE, reason="Modules not available")
    def test_retrieve_basic(self, retriever):
        """Test basic retrieval"""
        result = retriever.retrieve(
            query="hypertension",
            patient_id="TEST001"
        )
        
        assert isinstance(result, RetrievalResult)
        assert result.retrieval_time_ms >= 0
    
    @pytest.mark.skipif(not RETRIEVER_AVAILABLE or not GRAPH_RAG_AVAILABLE, reason="Modules not available")
    def test_get_patient_context_for_night(self, retriever):
        """Test getting night context"""
        context = retriever.get_patient_context_for_night("TEST001")
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Jean Dupont" in context
    
    @pytest.mark.skipif(not RETRIEVER_AVAILABLE or not GRAPH_RAG_AVAILABLE, reason="Modules not available")
    def test_get_patient_context_for_consultation(self, retriever):
        """Test getting consultation context"""
        context = retriever.get_patient_context_for_consultation(
            patient_id="TEST001",
            specialty="cardio"
        )
        
        assert isinstance(context, str)
        assert len(context) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
