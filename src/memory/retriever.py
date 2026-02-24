"""
Graph Retriever - Semantic and graph-based retrieval for patient context
Combines vector search with graph traversal for comprehensive retrieval
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum

from .patient_graph import PatientGraphRAG, PatientNode, NodeType, RelationType


class RetrievalMode(str, Enum):
    """Modes for context retrieval"""
    SEMANTIC = "semantic"        # Vector similarity search
    GRAPH = "graph"              # Graph traversal
    HYBRID = "hybrid"            # Combined approach
    TEMPORAL = "temporal"        # Time-based retrieval


class RetrievalResult(BaseModel):
    """Result from a retrieval query"""
    query: str
    mode: RetrievalMode
    patient_id: Optional[str] = None
    
    # Results
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    context_text: str = Field(default="")
    
    # Metadata
    total_results: int = Field(default=0)
    retrieval_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def get_context_for_llm(self) -> str:
        """Format context for LLM consumption"""
        if self.context_text:
            return self.context_text
        
        parts = [f"=== Contexte Récupéré (Query: {self.query}) ===\n"]
        
        for i, node in enumerate(self.nodes, 1):
            node_type = node.get("type", "unknown")
            name = node.get("name", "N/A")
            description = node.get("description", "")
            
            parts.append(f"{i}. [{node_type}] {name}")
            if description:
                parts.append(f"   {description}")
        
        return "\n".join(parts)


class GraphRetriever:
    """
    Retriever for patient knowledge graph.
    
    Combines multiple retrieval strategies:
    - Semantic search (vector similarity)
    - Graph traversal (relationships)
    - Temporal filtering (recent events)
    - Hybrid approach (combining all)
    """
    
    def __init__(self, graph_rag: PatientGraphRAG):
        """
        Initialize the retriever.
        
        Args:
            graph_rag: The PatientGraphRAG instance to query
        """
        self.graph = graph_rag
    
    def retrieve(
        self,
        query: str,
        patient_id: Optional[str] = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        max_results: int = 10,
        include_relationships: bool = True,
        time_window_days: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant context from the knowledge graph.
        
        Args:
            query: Search query or topic
            patient_id: Optional patient filter
            mode: Retrieval mode
            max_results: Maximum results to return
            include_relationships: Whether to include related nodes
            time_window_days: Filter to recent N days
            
        Returns:
            RetrievalResult with matching context
        """
        start_time = datetime.now()
        
        result = RetrievalResult(
            query=query,
            mode=mode,
            patient_id=patient_id
        )
        
        if mode == RetrievalMode.SEMANTIC:
            nodes = self._semantic_retrieval(query, patient_id, max_results)
        elif mode == RetrievalMode.GRAPH:
            nodes = self._graph_retrieval(query, patient_id, max_results)
        elif mode == RetrievalMode.TEMPORAL:
            nodes = self._temporal_retrieval(patient_id, time_window_days or 7, max_results)
        else:  # HYBRID
            nodes = self._hybrid_retrieval(
                query, patient_id, max_results, 
                time_window_days, include_relationships
            )
        
        result.nodes = nodes
        result.total_results = len(nodes)
        result.context_text = self._format_context(nodes, patient_id)
        result.retrieval_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return result
    
    def _semantic_retrieval(
        self,
        query: str,
        patient_id: Optional[str],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search"""
        search_results = self.graph.search(
            query=query,
            patient_id=patient_id,
            limit=max_results
        )
        
        return [
            {
                "id": r["node_id"],
                "type": r["type"],
                "name": r["node"]["name"],
                "description": r["node"]["description"],
                "score": r["score"],
                "data": r["node"]["data"]
            }
            for r in search_results
        ]
    
    def _graph_retrieval(
        self,
        query: str,
        patient_id: Optional[str],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Retrieve using graph traversal"""
        nodes = []
        
        if patient_id:
            # Get patient context
            context = self.graph.get_patient_context(patient_id)
            
            # Convert to node format
            if context.get("conditions"):
                for cond in context["conditions"][:max_results // 4]:
                    nodes.append({
                        "type": "condition",
                        "name": cond["name"],
                        "description": f"Condition: {cond['name']}",
                        "data": cond
                    })
            
            if context.get("medications"):
                for med in context["medications"][:max_results // 4]:
                    nodes.append({
                        "type": "medication",
                        "name": med["name"],
                        "description": f"Traitement: {med['name']}",
                        "data": med
                    })
            
            if context.get("recent_events"):
                for event in context["recent_events"][:max_results // 2]:
                    nodes.append({
                        "type": "event",
                        "name": event["type"],
                        "description": event["description"],
                        "data": event
                    })
        
        return nodes[:max_results]
    
    def _temporal_retrieval(
        self,
        patient_id: Optional[str],
        days: int,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Retrieve recent events within time window"""
        nodes = []
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for node_id, node in self.graph.nodes.items():
            # Filter by patient if specified
            if patient_id:
                patient_node_id = self.graph.patients.get(patient_id)
                if patient_node_id and not self.graph._is_connected(patient_node_id, node_id):
                    continue
            
            # Check timestamp
            timestamp_str = node.data.get("timestamp")
            if timestamp_str:
                try:
                    node_time = datetime.fromisoformat(timestamp_str).timestamp()
                    if node_time >= cutoff:
                        nodes.append({
                            "id": node_id,
                            "type": node.node_type.value,
                            "name": node.name,
                            "description": node.description,
                            "timestamp": timestamp_str,
                            "data": node.data
                        })
                except (ValueError, TypeError):
                    pass
        
        # Sort by timestamp (most recent first)
        nodes.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return nodes[:max_results]
    
    def _hybrid_retrieval(
        self,
        query: str,
        patient_id: Optional[str],
        max_results: int,
        time_window_days: Optional[int],
        include_relationships: bool
    ) -> List[Dict[str, Any]]:
        """Combined retrieval approach"""
        all_nodes = []
        seen_ids = set()
        
        # 1. Semantic search (40% of results)
        semantic_results = self._semantic_retrieval(query, patient_id, max_results // 2)
        for node in semantic_results:
            node_id = node.get("id", node.get("name"))
            if node_id not in seen_ids:
                node["source"] = "semantic"
                all_nodes.append(node)
                seen_ids.add(node_id)
        
        # 2. Graph traversal (30% of results)
        if patient_id:
            graph_results = self._graph_retrieval(query, patient_id, max_results // 3)
            for node in graph_results:
                node_id = node.get("id", node.get("name"))
                if node_id not in seen_ids:
                    node["source"] = "graph"
                    all_nodes.append(node)
                    seen_ids.add(node_id)
        
        # 3. Temporal (30% of results)
        if time_window_days:
            temporal_results = self._temporal_retrieval(patient_id, time_window_days, max_results // 3)
            for node in temporal_results:
                node_id = node.get("id", node.get("name"))
                if node_id not in seen_ids:
                    node["source"] = "temporal"
                    all_nodes.append(node)
                    seen_ids.add(node_id)
        
        # 4. Add related nodes if requested
        if include_relationships and patient_id:
            patient_node_id = self.graph.patients.get(patient_id)
            if patient_node_id:
                related = self.graph.get_related_nodes(patient_node_id, max_depth=1)
                for rel_node in related[:max_results // 4]:
                    if rel_node.id not in seen_ids:
                        all_nodes.append({
                            "id": rel_node.id,
                            "type": rel_node.node_type.value,
                            "name": rel_node.name,
                            "description": rel_node.description,
                            "source": "relationship",
                            "data": rel_node.data
                        })
                        seen_ids.add(rel_node.id)
        
        return all_nodes[:max_results]
    
    def _format_context(
        self,
        nodes: List[Dict[str, Any]],
        patient_id: Optional[str]
    ) -> str:
        """Format retrieved nodes into context text for LLM"""
        parts = []
        
        # Add patient header if available
        if patient_id:
            summary = self.graph.get_patient_summary(patient_id)
            if summary:
                parts.append(summary)
                parts.append("\n--- Informations Contextuelles ---\n")
        
        # Group by type
        conditions = [n for n in nodes if n.get("type") == "condition"]
        medications = [n for n in nodes if n.get("type") == "medication"]
        events = [n for n in nodes if n.get("type") == "event"]
        others = [n for n in nodes if n.get("type") not in ["condition", "medication", "event"]]
        
        if conditions:
            parts.append("**Conditions:**")
            for c in conditions:
                parts.append(f"- {c['name']}: {c.get('description', '')}")
        
        if medications:
            parts.append("\n**Traitements:**")
            for m in medications:
                parts.append(f"- {m['name']}")
        
        if events:
            parts.append("\n**Événements Récents:**")
            for e in events:
                ts = e.get("data", {}).get("timestamp", "")
                parts.append(f"- [{ts}] {e['name']}: {e.get('description', '')}")
        
        if others:
            parts.append("\n**Autres Informations:**")
            for o in others:
                parts.append(f"- [{o.get('type')}] {o['name']}")
        
        return "\n".join(parts)
    
    def get_patient_context_for_night(self, patient_id: str) -> str:
        """Get context specifically for night surveillance"""
        result = self.retrieve(
            query="surveillance nocturne alertes risques",
            patient_id=patient_id,
            mode=RetrievalMode.HYBRID,
            max_results=15,
            time_window_days=7
        )
        
        # Add specific night context
        context = result.context_text
        
        # Highlight risk factors
        patient_node = self.graph.get_patient_node(patient_id)
        if patient_node:
            risk_factors = patient_node.data.get("risk_factors", [])
            if risk_factors:
                context += f"\n\n⚠️ **Facteurs de Risque:** {', '.join(risk_factors)}"
        
        return context
    
    def get_patient_context_for_consultation(
        self,
        patient_id: str,
        specialty: str = "general"
    ) -> str:
        """Get context for day consultation based on specialty"""
        # Build specialty-specific query
        specialty_keywords = {
            "cardio": "cardiaque coeur rythme tension ECG",
            "dermato": "peau lésion cutané éruption",
            "ophtalmo": "oeil vision rétine fond",
            "pneumo": "poumon respiration toux saturation",
            "general": "symptômes antécédents traitements"
        }
        
        query = specialty_keywords.get(specialty, specialty_keywords["general"])
        
        result = self.retrieve(
            query=query,
            patient_id=patient_id,
            mode=RetrievalMode.HYBRID,
            max_results=20,
            time_window_days=30
        )
        
        return result.context_text
    
    def get_longitudinal_summary(
        self,
        patient_id: str,
        days: int = 30
    ) -> str:
        """Get longitudinal summary for trend analysis"""
        result = self.retrieve(
            query="évolution tendance progression",
            patient_id=patient_id,
            mode=RetrievalMode.TEMPORAL,
            max_results=50,
            time_window_days=days
        )
        
        # Build timeline
        timeline_parts = [f"=== Analyse Longitudinale ({days} jours) ===\n"]
        
        # Group events by day
        events_by_day: Dict[str, List[Dict]] = {}
        for node in result.nodes:
            ts = node.get("data", {}).get("timestamp", "")
            if ts:
                day = ts[:10]  # YYYY-MM-DD
                if day not in events_by_day:
                    events_by_day[day] = []
                events_by_day[day].append(node)
        
        # Format timeline
        for day in sorted(events_by_day.keys(), reverse=True):
            events = events_by_day[day]
            timeline_parts.append(f"\n📅 {day} ({len(events)} événements)")
            for event in events:
                timeline_parts.append(f"  • {event['name']}: {event.get('description', '')}")
        
        return "\n".join(timeline_parts)
