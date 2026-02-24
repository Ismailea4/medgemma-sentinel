"""
Patient Graph - GraphRAG implementation using LlamaIndex
Stores and retrieves patient medical history using a knowledge graph structure
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import json
import hashlib

# Try to import LlamaIndex components
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("Warning: LlamaIndex not installed. Using fallback graph implementation.")

# NetworkX for local graph storage (always available as fallback)
import networkx as nx


class RelationType(str, Enum):
    """Types of relationships in the patient knowledge graph"""
    # Patient relationships
    HAS_CONDITION = "has_condition"
    HAS_MEDICATION = "has_medication"
    HAS_ALLERGY = "has_allergy"
    
    # Clinical relationships
    DIAGNOSED_WITH = "diagnosed_with"
    TREATED_WITH = "treated_with"
    SYMPTOM_OF = "symptom_of"
    CAUSED_BY = "caused_by"
    
    # Temporal relationships
    PRECEDED_BY = "preceded_by"
    FOLLOWED_BY = "followed_by"
    OCCURRED_DURING = "occurred_during"
    
    # Event relationships
    TRIGGERED_ALERT = "triggered_alert"
    REQUIRED_INTERVENTION = "required_intervention"
    
    # Care relationships
    ATTENDED_BY = "attended_by"
    REFERRED_TO = "referred_to"


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph"""
    PATIENT = "patient"
    CONDITION = "condition"
    MEDICATION = "medication"
    SYMPTOM = "symptom"
    EVENT = "event"
    VITAL_SIGN = "vital_sign"
    CONSULTATION = "consultation"
    REPORT = "report"
    PROVIDER = "provider"
    PROCEDURE = "procedure"


class PatientNode(BaseModel):
    """A node in the patient knowledge graph"""
    id: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Type of node")
    name: str = Field(..., description="Node name/label")
    
    # Content
    description: str = Field(default="", description="Node description")
    data: Dict[str, Any] = Field(default_factory=dict, description="Node data/attributes")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source: str = Field(default="system", description="Data source")
    
    # For embedding/search
    embedding_text: str = Field(default="", description="Text for embedding generation")
    
    def generate_embedding_text(self) -> str:
        """Generate text representation for embedding"""
        parts = [
            f"Type: {self.node_type.value}",
            f"Name: {self.name}",
            self.description
        ]
        
        # Add relevant data fields
        for key, value in self.data.items():
            if isinstance(value, (str, int, float)):
                parts.append(f"{key}: {value}")
        
        self.embedding_text = " | ".join(parts)
        return self.embedding_text
    
    @staticmethod
    def generate_id(node_type: NodeType, name: str, patient_id: str = "") -> str:
        """Generate a consistent node ID"""
        key = f"{node_type.value}:{patient_id}:{name}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


class PatientGraphRAG:
    """
    GraphRAG system for patient medical history.
    
    Uses LlamaIndex for vector-based retrieval and NetworkX for graph structure.
    Stores patient history, conditions, events, and their relationships.
    
    Features:
    - Add/update patient information
    - Store clinical events with relationships
    - Retrieve relevant history using semantic search
    - Graph-based traversal for connected information
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize the PatientGraphRAG.
        
        Args:
            persist_dir: Directory for persisting the graph (optional)
        """
        self.persist_dir = persist_dir
        
        # Initialize NetworkX graph
        self.graph = nx.MultiDiGraph()
        
        # Node storage
        self.nodes: Dict[str, PatientNode] = {}
        
        # LlamaIndex index (if available)
        self.vector_index = None
        if LLAMAINDEX_AVAILABLE:
            self._init_llamaindex()
        
        # Patient registry
        self.patients: Dict[str, str] = {}  # patient_id -> node_id
        
    def _init_llamaindex(self) -> None:
        """Initialize LlamaIndex components"""
        try:
            # Configure settings for offline use
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            # Note: In production, configure local embedding model here
        except Exception as e:
            print(f"Warning: Could not initialize LlamaIndex: {e}")
    
    # ==================== Patient Management ====================
    
    def add_patient(
        self,
        patient_id: str,
        name: str,
        age: int,
        conditions: Optional[List[str]] = None,
        medications: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None,
        risk_factors: Optional[List[str]] = None,
        room: Optional[str] = None,
        **extra_data
    ) -> str:
        """
        Add a new patient to the graph.
        
        Returns the node ID of the patient.
        """
        # Create patient node
        node_id = PatientNode.generate_id(NodeType.PATIENT, patient_id)
        
        patient_node = PatientNode(
            id=node_id,
            node_type=NodeType.PATIENT,
            name=name,
            description=f"Patient {name}, {age} ans",
            data={
                "patient_id": patient_id,
                "age": age,
                "room": room,
                "risk_factors": risk_factors or [],
                **extra_data
            }
        )
        patient_node.generate_embedding_text()
        
        # Add to graph
        self._add_node(patient_node)
        self.patients[patient_id] = node_id
        
        # Add conditions as connected nodes
        if conditions:
            for condition in conditions:
                self.add_condition(patient_id, condition)
        
        # Add medications
        if medications:
            for med in medications:
                self.add_medication(patient_id, med)
        
        # Add allergies
        if allergies:
            for allergy in allergies:
                self.add_allergy(patient_id, allergy)
        
        return node_id
    
    def add_condition(
        self,
        patient_id: str,
        condition_name: str,
        severity: Optional[str] = None,
        diagnosed_date: Optional[str] = None,
        icd_code: Optional[str] = None
    ) -> str:
        """Add a medical condition for a patient"""
        node_id = PatientNode.generate_id(NodeType.CONDITION, condition_name, patient_id)
        
        condition_node = PatientNode(
            id=node_id,
            node_type=NodeType.CONDITION,
            name=condition_name,
            description=f"Condition: {condition_name}",
            data={
                "severity": severity,
                "diagnosed_date": diagnosed_date,
                "icd_code": icd_code
            }
        )
        condition_node.generate_embedding_text()
        
        self._add_node(condition_node)
        
        # Create relationship to patient
        if patient_id in self.patients:
            self._add_edge(
                self.patients[patient_id],
                node_id,
                RelationType.HAS_CONDITION
            )
        
        return node_id
    
    def add_medication(
        self,
        patient_id: str,
        medication_name: str,
        dosage: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> str:
        """Add a medication for a patient"""
        node_id = PatientNode.generate_id(NodeType.MEDICATION, medication_name, patient_id)
        
        med_node = PatientNode(
            id=node_id,
            node_type=NodeType.MEDICATION,
            name=medication_name,
            description=f"Medication: {medication_name} {dosage or ''} {frequency or ''}",
            data={
                "dosage": dosage,
                "frequency": frequency
            }
        )
        med_node.generate_embedding_text()
        
        self._add_node(med_node)
        
        if patient_id in self.patients:
            self._add_edge(
                self.patients[patient_id],
                node_id,
                RelationType.HAS_MEDICATION
            )
        
        return node_id
    
    def add_allergy(
        self,
        patient_id: str,
        allergen: str,
        severity: str = "moderate",
        reaction: Optional[str] = None
    ) -> str:
        """Add an allergy for a patient"""
        node_id = PatientNode.generate_id(NodeType.CONDITION, f"allergy_{allergen}", patient_id)
        
        allergy_node = PatientNode(
            id=node_id,
            node_type=NodeType.CONDITION,
            name=f"Allergie: {allergen}",
            description=f"Allergie à {allergen}, sévérité {severity}",
            data={
                "allergen": allergen,
                "severity": severity,
                "reaction": reaction,
                "is_allergy": True
            }
        )
        allergy_node.generate_embedding_text()
        
        self._add_node(allergy_node)
        
        if patient_id in self.patients:
            self._add_edge(
                self.patients[patient_id],
                node_id,
                RelationType.HAS_ALLERGY
            )
        
        return node_id
    
    # ==================== Event Management ====================
    
    def add_clinical_event(
        self,
        patient_id: str,
        event_type: str,
        description: str,
        severity: str = "medium",
        timestamp: Optional[datetime] = None,
        data: Optional[Dict[str, Any]] = None,
        related_conditions: Optional[List[str]] = None
    ) -> str:
        """
        Add a clinical event to the patient's history.
        
        Events include: vital anomalies, alerts, consultations, procedures
        """
        timestamp = timestamp or datetime.now()
        node_id = PatientNode.generate_id(
            NodeType.EVENT,
            f"{event_type}_{timestamp.isoformat()}",
            patient_id
        )
        
        event_node = PatientNode(
            id=node_id,
            node_type=NodeType.EVENT,
            name=event_type,
            description=description,
            data={
                "event_type": event_type,
                "severity": severity,
                "timestamp": timestamp.isoformat(),
                **(data or {})
            }
        )
        event_node.generate_embedding_text()
        
        self._add_node(event_node)
        
        # Link to patient
        if patient_id in self.patients:
            self._add_edge(
                self.patients[patient_id],
                node_id,
                RelationType.TRIGGERED_ALERT if severity in ["high", "critical"] 
                else RelationType.OCCURRED_DURING
            )
        
        # Link to related conditions
        if related_conditions:
            for condition in related_conditions:
                condition_id = PatientNode.generate_id(NodeType.CONDITION, condition, patient_id)
                if condition_id in self.nodes:
                    self._add_edge(node_id, condition_id, RelationType.SYMPTOM_OF)
        
        return node_id
    
    def add_consultation(
        self,
        patient_id: str,
        consultation_type: str,
        presenting_complaint: str,
        findings: Optional[List[str]] = None,
        diagnosis: Optional[str] = None,
        treatment: Optional[str] = None,
        provider: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """Add a consultation record"""
        timestamp = timestamp or datetime.now()
        node_id = PatientNode.generate_id(
            NodeType.CONSULTATION,
            f"consultation_{timestamp.isoformat()}",
            patient_id
        )
        
        consultation_node = PatientNode(
            id=node_id,
            node_type=NodeType.CONSULTATION,
            name=f"Consultation {consultation_type}",
            description=f"Motif: {presenting_complaint}. {diagnosis or ''}",
            data={
                "consultation_type": consultation_type,
                "presenting_complaint": presenting_complaint,
                "findings": findings or [],
                "diagnosis": diagnosis,
                "treatment": treatment,
                "provider": provider,
                "timestamp": timestamp.isoformat()
            }
        )
        consultation_node.generate_embedding_text()
        
        self._add_node(consultation_node)
        
        if patient_id in self.patients:
            self._add_edge(
                self.patients[patient_id],
                node_id,
                RelationType.ATTENDED_BY
            )
        
        return node_id
    
    def add_report(
        self,
        patient_id: str,
        report_type: str,
        summary: str,
        content: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> str:
        """Add a generated report to the graph"""
        timestamp = timestamp or datetime.now()
        node_id = PatientNode.generate_id(
            NodeType.REPORT,
            f"report_{report_type}_{timestamp.isoformat()}",
            patient_id
        )
        
        report_node = PatientNode(
            id=node_id,
            node_type=NodeType.REPORT,
            name=f"Rapport {report_type}",
            description=summary,
            data={
                "report_type": report_type,
                "content": content,
                "timestamp": timestamp.isoformat()
            }
        )
        report_node.generate_embedding_text()
        
        self._add_node(report_node)
        
        if patient_id in self.patients:
            self._add_edge(
                self.patients[patient_id],
                node_id,
                RelationType.OCCURRED_DURING
            )
        
        return node_id
    
    # ==================== Graph Operations ====================
    
    def _add_node(self, node: PatientNode) -> None:
        """Add a node to both storage and graph"""
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id,
            type=node.node_type.value,
            name=node.name,
            data=node.model_dump()
        )
    
    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an edge between two nodes"""
        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation.value,
            data=data or {},
            created_at=datetime.now().isoformat()
        )
    
    def get_patient_node(self, patient_id: str) -> Optional[PatientNode]:
        """Get the patient node by patient ID"""
        if patient_id in self.patients:
            return self.nodes.get(self.patients[patient_id])
        return None
    
    def get_patient_context(self, patient_id: str) -> Dict[str, Any]:
        """
        Get comprehensive patient context for clinical decision support.
        Returns all connected information for a patient.
        """
        if patient_id not in self.patients:
            return {}
        
        patient_node_id = self.patients[patient_id]
        patient_node = self.nodes.get(patient_node_id)
        
        if not patient_node:
            return {}
        
        context = {
            "patient_id": patient_id,
            "name": patient_node.name,
            **patient_node.data,
            "conditions": [],
            "medications": [],
            "allergies": [],
            "recent_events": [],
            "consultations": [],
            "reports": []
        }
        
        # Get all connected nodes
        for neighbor_id in self.graph.neighbors(patient_node_id):
            neighbor_node = self.nodes.get(neighbor_id)
            if not neighbor_node:
                continue
            
            if neighbor_node.node_type == NodeType.CONDITION:
                if neighbor_node.data.get("is_allergy"):
                    context["allergies"].append({
                        "name": neighbor_node.name,
                        **neighbor_node.data
                    })
                else:
                    context["conditions"].append({
                        "name": neighbor_node.name,
                        **neighbor_node.data
                    })
            
            elif neighbor_node.node_type == NodeType.MEDICATION:
                context["medications"].append({
                    "name": neighbor_node.name,
                    **neighbor_node.data
                })
            
            elif neighbor_node.node_type == NodeType.EVENT:
                context["recent_events"].append({
                    "type": neighbor_node.name,
                    "description": neighbor_node.description,
                    **neighbor_node.data
                })
            
            elif neighbor_node.node_type == NodeType.CONSULTATION:
                context["consultations"].append({
                    "type": neighbor_node.name,
                    **neighbor_node.data
                })
            
            elif neighbor_node.node_type == NodeType.REPORT:
                context["reports"].append({
                    "type": neighbor_node.name,
                    **neighbor_node.data
                })
        
        return context
    
    def get_patient_summary(self, patient_id: str) -> str:
        """Generate a text summary of patient history for LLM context"""
        context = self.get_patient_context(patient_id)
        
        if not context:
            return f"Aucune information disponible pour le patient {patient_id}"
        
        summary_parts = [
            f"=== Résumé Patient: {context.get('name', 'N/A')} ===",
            f"ID: {patient_id}",
            f"Âge: {context.get('age', 'N/A')} ans",
            f"Chambre: {context.get('room', 'N/A')}"
        ]
        
        if context.get("conditions"):
            conditions = [c["name"] for c in context["conditions"]]
            summary_parts.append(f"\nConditions: {', '.join(conditions)}")
        
        if context.get("allergies"):
            allergies = [a["name"] for a in context["allergies"]]
            summary_parts.append(f"Allergies: {', '.join(allergies)}")
        
        if context.get("medications"):
            meds = [m["name"] for m in context["medications"]]
            summary_parts.append(f"Traitements: {', '.join(meds)}")
        
        if context.get("risk_factors"):
            summary_parts.append(f"Facteurs de risque: {', '.join(context['risk_factors'])}")
        
        if context.get("recent_events"):
            summary_parts.append(f"\nÉvénements récents: {len(context['recent_events'])}")
            for event in context["recent_events"][:5]:  # Last 5
                summary_parts.append(f"  - {event['type']}: {event['description']}")
        
        return "\n".join(summary_parts)
    
    # ==================== Search/Retrieval ====================
    
    def search(
        self,
        query: str,
        patient_id: Optional[str] = None,
        node_types: Optional[List[NodeType]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph using semantic similarity.
        
        Args:
            query: Search query
            patient_id: Optional filter by patient
            node_types: Optional filter by node types
            limit: Maximum results
            
        Returns:
            List of matching nodes with relevance scores
        """
        results = []
        
        # Simple keyword matching (fallback without embeddings)
        query_lower = query.lower()
        
        for node_id, node in self.nodes.items():
            # Filter by patient if specified
            if patient_id and patient_id in self.patients:
                patient_node_id = self.patients[patient_id]
                if not self._is_connected(patient_node_id, node_id):
                    continue
            
            # Filter by node type
            if node_types and node.node_type not in node_types:
                continue
            
            # Calculate simple relevance score
            text = node.embedding_text.lower()
            score = 0.0
            
            for word in query_lower.split():
                if word in text:
                    score += 1.0
            
            if score > 0:
                results.append({
                    "node_id": node_id,
                    "node": node.model_dump(),
                    "score": score,
                    "type": node.node_type.value
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _is_connected(self, source_id: str, target_id: str) -> bool:
        """Check if two nodes are connected (directly or indirectly)"""
        if source_id == target_id:
            return True
        try:
            return nx.has_path(self.graph, source_id, target_id) or \
                   nx.has_path(self.graph, target_id, source_id)
        except nx.NetworkXError:
            return False
    
    def get_related_nodes(
        self,
        node_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 2
    ) -> List[PatientNode]:
        """Get nodes related to a given node"""
        related = []
        
        if node_id not in self.graph:
            return related
        
        # BFS traversal
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            if current_id != node_id and current_id in self.nodes:
                related.append(self.nodes[current_id])
            
            # Get neighbors
            for neighbor_id in self.graph.neighbors(current_id):
                if neighbor_id not in visited:
                    # Check relation type filter
                    if relation_types:
                        edge_data = self.graph.get_edge_data(current_id, neighbor_id)
                        if edge_data:
                            for key, data in edge_data.items():
                                if data.get("relation") in [r.value for r in relation_types]:
                                    queue.append((neighbor_id, depth + 1))
                                    break
                    else:
                        queue.append((neighbor_id, depth + 1))
        
        return related
    
    # ==================== Persistence ====================
    
    def save(self, filepath: Optional[str] = None) -> None:
        """Save the graph to disk"""
        path = filepath or (self.persist_dir + "/graph.json" if self.persist_dir else None)
        if not path:
            return
        
        data = {
            "nodes": {nid: node.model_dump() for nid, node in self.nodes.items()},
            "edges": list(self.graph.edges(data=True)),
            "patients": self.patients
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, filepath: Optional[str] = None) -> None:
        """Load the graph from disk"""
        path = filepath or (self.persist_dir + "/graph.json" if self.persist_dir else None)
        if not path:
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Restore nodes
            for nid, node_data in data.get("nodes", {}).items():
                node = PatientNode(**node_data)
                self.nodes[nid] = node
                self.graph.add_node(nid, **node_data)
            
            # Restore edges
            for source, target, edge_data in data.get("edges", []):
                self.graph.add_edge(source, target, **edge_data)
            
            # Restore patient registry
            self.patients = data.get("patients", {})
            
        except FileNotFoundError:
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": self.graph.number_of_edges(),
            "total_patients": len(self.patients),
            "node_types": {
                nt.value: len([n for n in self.nodes.values() if n.node_type == nt])
                for nt in NodeType
            }
        }
