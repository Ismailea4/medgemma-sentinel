"""
LangGraph State Machine - MedGemma Sentinel Workflow
Connects: Night -> Rap1 -> Day -> Rap2

This is the main orchestration graph that coordinates the full
clinical surveillance and reporting cycle.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Literal, TypedDict, Annotated
from enum import Enum
import operator
import uuid

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # Fallback for type hints
    END = "END"  # Fallback constant
    MemorySaver = None  # Fallback
    print("Warning: LangGraph not installed. Using fallback implementation.")

from .state import SentinelState, WorkflowPhase, create_initial_state
from .nodes import NightNode, Rap1Node, DayNode, Rap2Node


# Type definitions for LangGraph state
class GraphState(TypedDict, total=False):
    """State type for LangGraph compatibility"""
    session_id: str
    patient_id: str
    phase: str
    steering_mode: str
    workflow_start: str
    workflow_end: Optional[str]
    current_phase_start: Optional[str]
    patient_context: Dict[str, Any]
    patient_history_summary: str
    risk_factors: list
    night_data: Optional[Dict[str, Any]]
    day_data: Optional[Dict[str, Any]]
    rap1_report: Optional[Dict[str, Any]]
    rap2_report: Optional[Dict[str, Any]]
    messages: Annotated[list, operator.add]
    errors: list
    warnings: list
    total_events_processed: int
    total_alerts: int
    processing_time_seconds: float
    # Input fields
    vitals_input: list
    audio_input: list
    vision_input: list
    consultation_mode: str
    symptoms_input: list
    exam_input: Dict[str, str]
    day_vitals_input: Dict[str, Any]
    images_input: list


class MedGemmaSentinelGraph:
    """
    Main orchestration class for MedGemma Sentinel workflow.
    
    Implements the state machine:
    Night -> Rap1 -> Day -> Rap2 -> END
    
    Usage:
        sentinel = MedGemmaSentinelGraph()
        result = sentinel.run(patient_id="P001", vitals_input=[...])
    """
    
    def __init__(self, use_memory: bool = True, use_medgemma: bool = True):
        """
        Initialize the Sentinel graph.
        
        Args:
            use_memory: Whether to use LangGraph's memory saver for checkpointing
            use_medgemma: Whether to use MedGemma for AI-powered report generation
        """
        self.night_node = NightNode()
        self.rap1_node = Rap1Node(use_medgemma=use_medgemma)
        self.day_node = DayNode()
        self.rap2_node = Rap2Node(use_medgemma=use_medgemma)
        
        self.use_memory = use_memory
        self.memory = MemorySaver() if LANGGRAPH_AVAILABLE and use_memory else None
        
        # Build the graph
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None
        self.compiled_graph = None
        
        if self.graph:
            self._compile_graph()
    
    def _build_graph(self) -> Optional[StateGraph]:
        """Build the LangGraph state machine"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        # Create the graph with our state type
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("night", self._night_wrapper)
        graph.add_node("rap1", self._rap1_wrapper)
        graph.add_node("day", self._day_wrapper)
        graph.add_node("rap2", self._rap2_wrapper)
        
        # Define edges: Night -> Rap1 -> Day -> Rap2 -> END
        graph.add_edge("night", "rap1")
        graph.add_edge("rap1", "day")
        graph.add_edge("day", "rap2")
        graph.add_edge("rap2", END)
        
        # Set entry point
        graph.set_entry_point("night")
        
        return graph
    
    def _compile_graph(self) -> None:
        """Compile the graph for execution"""
        if self.graph:
            if self.memory:
                self.compiled_graph = self.graph.compile(checkpointer=self.memory)
            else:
                self.compiled_graph = self.graph.compile()
    
    # Node wrapper functions for LangGraph
    def _night_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for night node execution"""
        return self.night_node.execute(dict(state))
    
    def _rap1_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for rap1 node execution"""
        return self.rap1_node.execute(dict(state))
    
    def _day_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for day node execution"""
        return self.day_node.execute(dict(state))
    
    def _rap2_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for rap2 node execution"""
        return self.rap2_node.execute(dict(state))
    
    def run(
        self,
        patient_id: str,
        patient_context: Optional[Dict[str, Any]] = None,
        vitals_input: Optional[list] = None,
        audio_input: Optional[list] = None,
        vision_input: Optional[list] = None,
        consultation_mode: str = "general",
        symptoms_input: Optional[list] = None,
        exam_input: Optional[Dict[str, str]] = None,
        day_vitals_input: Optional[Dict[str, Any]] = None,
        images_input: Optional[list] = None,
        presenting_complaint: Optional[str] = None,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete Sentinel workflow.
        
        Args:
            patient_id: Unique patient identifier
            patient_context: Patient information from GraphRAG
            vitals_input: Night vital signs readings
            audio_input: Night audio events
            vision_input: Night vision events
            consultation_mode: Day consultation specialty
            symptoms_input: Patient symptoms for day consultation
            exam_input: Physical examination findings
            day_vitals_input: Vitals during day consultation
            images_input: Clinical images for analysis
            session_id: Optional session identifier
            config: LangGraph configuration
            
        Returns:
            Final state with both reports generated
        """
        # Create initial state
        initial_state: GraphState = {
            "session_id": session_id or str(uuid.uuid4()),
            "patient_id": patient_id,
            "phase": WorkflowPhase.NIGHT.value,
            "steering_mode": "night_surveillance",
            "workflow_start": datetime.now().isoformat(),
            "workflow_end": None,
            "current_phase_start": datetime.now().isoformat(),
            "patient_context": patient_context or {},
            "patient_history_summary": "",
            "risk_factors": [],
            "night_data": None,
            "day_data": None,
            "rap1_report": None,
            "rap2_report": None,
            "messages": [],
            "errors": [],
            "warnings": [],
            "total_events_processed": 0,
            "total_alerts": 0,
            "processing_time_seconds": 0.0,
            # Inputs
            "vitals_input": vitals_input or [],
            "audio_input": audio_input or [],
            "vision_input": vision_input or [],
            "consultation_mode": consultation_mode,
            "symptoms_input": symptoms_input or [],
            "exam_input": exam_input or {},
            "day_vitals_input": day_vitals_input or {},
            "images_input": images_input or [],
            "presenting_complaint": presenting_complaint or ""
        }
        
        # Run the graph
        if self.compiled_graph:
            # Use LangGraph
            run_config = config or {"configurable": {"thread_id": initial_state["session_id"]}}
            result = self.compiled_graph.invoke(initial_state, run_config)
            return result
        else:
            # Fallback: run nodes sequentially
            return self._run_fallback(initial_state)
    
    def _run_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution without LangGraph"""
        print("[Fallback] Running without LangGraph...")
        
        # Execute nodes sequentially
        state = self.night_node.execute(state)
        state = self.rap1_node.execute(state)
        state = self.day_node.execute(state)
        state = self.rap2_node.execute(state)
        
        return state
    
    def run_night_only(
        self,
        patient_id: str,
        vitals_input: Optional[list] = None,
        audio_input: Optional[list] = None,
        vision_input: Optional[list] = None,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run only the night surveillance phase.
        
        Returns state after Night -> Rap1
        """
        initial_state: GraphState = {
            "session_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "phase": WorkflowPhase.NIGHT.value,
            "patient_context": patient_context or {},
            "vitals_input": vitals_input or [],
            "audio_input": audio_input or [],
            "vision_input": vision_input or [],
            "messages": [],
            "errors": [],
            "warnings": [],
            "total_events_processed": 0,
            "total_alerts": 0,
        }
        
        # Run night and rap1 only
        state = self.night_node.execute(initial_state)
        state = self.rap1_node.execute(state)
        
        return state
    
    def run_day_only(
        self,
        patient_id: str,
        consultation_mode: str = "general",
        symptoms_input: Optional[list] = None,
        exam_input: Optional[Dict[str, str]] = None,
        vitals_input: Optional[Dict[str, Any]] = None,
        images_input: Optional[list] = None,
        patient_context: Optional[Dict[str, Any]] = None,
        prior_night_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run only the day consultation phase.
        
        Returns state after Day -> Rap2
        """
        initial_state: GraphState = {
            "session_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "phase": WorkflowPhase.DAY.value,
            "patient_context": patient_context or {},
            "consultation_mode": consultation_mode,
            "symptoms_input": symptoms_input or [],
            "exam_input": exam_input or {},
            "day_vitals_input": vitals_input or {},
            "images_input": images_input or [],
            "night_data": prior_night_data,
            "messages": [],
            "errors": [],
            "warnings": [],
        }
        
        # Run day and rap2 only
        state = self.day_node.execute(initial_state)
        state = self.rap2_node.execute(state)
        
        return state
    
    def get_graph_visualization(self) -> Optional[str]:
        """Get ASCII visualization of the graph"""
        return """
+==============================================================+
|           MedGemma Sentinel - Workflow Graph                  |
+==============================================================+
|                                                               |
|    +---------+      +---------+      +---------+      +----+  |
|    |  NIGHT  | ---> |  RAP1   | ---> |   DAY   | ---> |RAP2|  |
|    |  (Sur-  |      | (Night  |      | (Consul-|      |(Day|  |
|    |veillance|      | Report) |      | tation) |      |Rpt)|  |
|    +---------+      +---------+      +---------+      +--+-+  |
|         |                                                |    |
|         |                                                v    |
|         |                                            +------+ |
|         +--------------------------------------------| END  | |
|                                                      +------+ |
|                                                               |
|    Steering Modes:                                            |
|    - NIGHT: night_surveillance (apnees, hypoxie)              |
|    - RAP1/RAP2: longitudinal (analyse temporelle)             |
|    - DAY: specialist_virtual (aide diagnostic)                |
|                                                               |
+==============================================================+
"""


def create_sentinel_graph(use_memory: bool = True) -> MedGemmaSentinelGraph:
    """
    Factory function to create a MedGemma Sentinel graph.
    
    Args:
        use_memory: Whether to enable checkpointing
        
    Returns:
        Configured MedGemmaSentinelGraph instance
    """
    return MedGemmaSentinelGraph(use_memory=use_memory)
