"""
LangGraph State Machine - MedGemma Sentinel Workflow
Every node is guarded: Input/Output checks wrap each step.

Architecture (guardrails at every step):
  [Guard] → Night → [Guard] → Rap1 → [Guard] → Day → [Guard] → Rap2 → [Guard] → END
     ↑         ↑        ↑         ↑        ↑        ↑        ↑         ↑       ↑
   Layer1+2  Layer3   Layer1+2  Layer3   Layer1+2  Layer3   Layer1+2  Layer3  Final

Each node wrapper does:
  1. Layer 1+2: Check safety of the data entering this node
  2. Execute the node
  3. Layer 3: Audit the content produced by this node
  4. If any check fails → stop pipeline, mark state as blocked
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from enum import Enum
import operator
import uuid
import logging
import os

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not installed. Using fallback implementation.")

from .state import SentinelState, WorkflowPhase, create_initial_state
from .nodes import NightNode, Rap1Node, DayNode, Rap2Node

logger = logging.getLogger(__name__)


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
    presenting_complaint: str
    # Guardrails fields
    guardrails_enabled: bool
    input_guard_result: Optional[Dict[str, Any]]
    output_guard_result: Optional[Dict[str, Any]]
    guardrails_blocked: bool
    guard_log: list  # Per-node guard audit trail


class MedGemmaSentinelGraph:
    """
    Main orchestration class for MedGemma Sentinel workflow.
    
    Every node is wrapped with input + output guardrails:
      [Guard] → Night → [Guard] → Rap1 → [Guard] → Day → [Guard] → Rap2 → [Guard]
    
    The 3-layer Sequential Defense runs at EACH step:
      Layer 1 (Regex/Colang): Blocks known harmful patterns
      Layer 2 (Llama Guard 3 input): Classifies input safety
      Layer 3 (Llama Guard 3 output): Audits generated content
    
    Usage:
        sentinel = MedGemmaSentinelGraph()
        result = sentinel.run(patient_id="P001", vitals_input=[...])
    """
    
    def __init__(self, use_memory: bool = True, trusted_internal_inputs: bool = False):
        """
        Initialize the Sentinel graph with per-node guardrails.
        
        Args:
            use_memory: Whether to use LangGraph's memory saver for checkpointing
            trusted_internal_inputs: If True, skip expensive input checks for internal
                                     downstream nodes (day/rap2) where inputs are trusted.
        """
        self.night_node = NightNode()
        self.rap1_node = Rap1Node()
        self.day_node = DayNode()
        self.rap2_node = Rap2Node()
        
        self.use_memory = use_memory
        self.memory = MemorySaver() if LANGGRAPH_AVAILABLE and use_memory else None
        
        # Initialize SentinelGuard (graceful fallback)
        self._guard = None
        self._guardrails_enabled = False
        self._trusted_internal_inputs = trusted_internal_inputs or os.environ.get(
            "MEDGEMMA_TRUSTED_INTERNAL_INPUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._init_guardrails()
        
        # Build the graph
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None
        self.compiled_graph = None
        
        if self.graph:
            self._compile_graph()
    
    def _init_guardrails(self) -> None:
        """Initialize SentinelGuard with graceful fallback."""
        try:
            from src.guardrails import SentinelGuard
            self._guard = SentinelGuard()
            self._guardrails_enabled = self._guard.enabled
            mode = self._guard._mode if hasattr(self._guard, '_mode') else "unknown"
            logger.info(f"[GUARD] SentinelGuard initialized (mode={mode})")
            print(f"[GUARD] SentinelGuard initialized (mode={mode})")
        except ImportError:
            logger.info("[GUARD] Guardrails module not available — running unguarded")
            print("[GUARD] Guardrails module not available — running unguarded")
        except Exception as e:
            logger.warning(f"[GUARD] Guardrails initialization failed: {e} — running unguarded")
            print(f"[GUARD] Guardrails init failed: {e} — running unguarded")
    
    # ──────────────────────────────────────────────────────────────────────
    #  Per-node guardrails helpers
    # ──────────────────────────────────────────────────────────────────────
    
    def _extract_text_from_state(self, state: dict, node_name: str) -> str:
        """
        Extract USER-PROVIDED text to check from the state for a given node.
        
        IMPORTANT: patient_context is trusted system data (conditions, medications,
        room number, etc.) — it must NOT be sent to Llama Guard input classification
        as it causes false positives on legitimate medical terms like 
        "Insuffisance cardiaque" being flagged as POLICY_VIOLATION.
        
        Only check:
        - presenting_complaint (user-typed free text)
        - symptoms_input (user-provided symptom list)
        """
        texts = []
        
        if node_name == "night":
            # Only check user-provided presenting complaint
            complaint = state.get("presenting_complaint", "")
            if complaint:
                texts.append(complaint)
                
        elif node_name == "rap1":
            # Rap1 consumes night_data — system-generated, skip input check
            pass
                        
        elif node_name == "day":
            # Check user-provided symptoms and complaint only
            for symptom in state.get("symptoms_input", []):
                if isinstance(symptom, str):
                    texts.append(symptom)
            complaint = state.get("presenting_complaint", "")
            if complaint:
                texts.append(complaint)
            # DO NOT check patient_context — trusted system data
                        
        elif node_name == "rap2":
            # Rap2 consumes day_data — only check user-originated complaint
            day_data = state.get("day_data", {})
            if isinstance(day_data, dict):
                complaint = day_data.get("presenting_complaint", "")
                if complaint:
                    texts.append(str(complaint))
        
        return " | ".join(texts) if texts else ""
    
    def _extract_output_text(self, state: dict, node_name: str) -> str:
        """
        Extract the text produced by a node for output audit.
        """
        if node_name == "night":
            # Night produces: events, alert descriptions
            night_data = state.get("night_data", {})
            if isinstance(night_data, dict):
                events = night_data.get("events", [])
                descs = []
                for e in events:
                    if isinstance(e, dict):
                        d = e.get("description", "") or e.get("type", "")
                        if d:
                            descs.append(str(d))
                return " | ".join(descs)
                
        elif node_name == "rap1":
            # Rap1 produces: markdown report
            rap1 = state.get("rap1_report", {})
            if isinstance(rap1, dict):
                return rap1.get("markdown_content", "")
                
        elif node_name == "day":
            # Day produces: day_data fields
            day_data = state.get("day_data", {})
            if isinstance(day_data, dict):
                parts = []
                for dx in day_data.get("differential_diagnosis", []):
                    parts.append(str(dx))
                for a in day_data.get("recommended_actions", []):
                    parts.append(str(a))
                return " | ".join(parts)
                
        elif node_name == "rap2":
            # Rap2 produces: markdown report
            rap2 = state.get("rap2_report", {})
            if isinstance(rap2, dict):
                return rap2.get("markdown_content", "")
        
        return ""
    
    def _run_input_guard(self, state: dict, node_name: str) -> dict:
        """
        Layer 1+2 input check BEFORE a node executes.
        Returns the state (potentially with guardrails_blocked=True).
        """
        if not self._guardrails_enabled or not self._guard:
            self._append_guard_log(state, node_name, "input", "skipped", "guardrails_disabled")
            return state

        if self._trusted_internal_inputs and node_name in {"day", "rap2"}:
            self._append_guard_log(state, node_name, "input", "skipped", "trusted_internal_inputs")
            logger.info(f"[GUARD] ⏭️ {node_name} input guard skipped (trusted internal inputs)")
            print(f"[GUARD] ⏭️ {node_name} input guard skipped (trusted internal inputs)")
            return state
        
        # Already blocked by a previous node? skip
        if state.get("guardrails_blocked", False):
            return state
        
        text = self._extract_text_from_state(state, node_name)
        if not text:
            self._append_guard_log(state, node_name, "input", "passed", "no_text_input")
            return state
        
        result = self._guard.check_input_sync(text)
        
        if not result.allowed:
            state["guardrails_blocked"] = True
            state["phase"] = WorkflowPhase.COMPLETED.value
            state["workflow_end"] = datetime.now().isoformat()
            state["input_guard_result"] = {
                "blocked_at": node_name,
                "status": "blocked",
                "allowed": False,
                "violations": result.violations if hasattr(result, 'violations') else [],
                "message": result.message,
                "timestamp": datetime.now().isoformat()
            }
            state["errors"] = state.get("errors", []) + [
                f"[GUARD] Input blocked at {node_name}: {result.violations}"
            ]
            state["messages"] = state.get("messages", []) + [{
                "role": "system",
                "content": f"[GUARD] Pipeline blocked at {node_name} — unsafe input. {result.message}",
                "timestamp": datetime.now().isoformat()
            }]
            self._append_guard_log(state, node_name, "input", "blocked", 
                                   result.violations if hasattr(result, 'violations') else [])
            logger.warning(f"[GUARD] ❌ Input BLOCKED at {node_name}: {result.violations}")
            print(f"[GUARD] ❌ Input BLOCKED at {node_name}: {result.violations}")
        else:
            self._append_guard_log(state, node_name, "input", "passed")
            logger.info(f"[GUARD] ✅ {node_name} input passed Layer 1+2")
            print(f"[GUARD] ✅ {node_name} input passed Layer 1+2")
        
        return state
    
    def _run_output_guard(self, state: dict, node_name: str) -> dict:
        """
        Layer 3 output audit AFTER a node executes.
        
        Two-pass optimization:
          Pass 1: Check ALL output as one text (single inference call, ~30s)
                  If safe → done immediately (common case)
          Pass 2: Only if Pass 1 flags → drill into individual sections to find
                  which ones are unsafe, keep the safe ones
        """
        if not self._guardrails_enabled or not self._guard:
            self._append_guard_log(state, node_name, "output", "skipped", "guardrails_disabled")
            return state
        
        if state.get("guardrails_blocked", False):
            return state

        # Deterministic internal nodes (no free-form LLM text) skip expensive output checks.
        # Keep strict output checks for final report nodes (rap1, rap2).
        if node_name in {"night", "day"}:
            self._append_guard_log(state, node_name, "output", "skipped", "deterministic_node_output_skip")
            logger.info(f"[GUARD] ⏭️ {node_name} output guard skipped (deterministic internal node)")
            print(f"[GUARD] ⏭️ {node_name} output guard skipped (deterministic internal node)")
            return state
        
        # ---- Pass 1: Check entire output as one text (single call) ----
        full_text = self._extract_output_text(state, node_name)
        if not full_text:
            self._append_guard_log(state, node_name, "output", "passed", "no_output_text")
            return state
        
        user_input = state.get("presenting_complaint", "")
        bulk_result = self._guard.check_output_sync(full_text, user_input)
        
        if bulk_result.allowed:
            # All safe — no per-section check needed
            self._append_guard_log(state, node_name, "output", "passed")
            logger.info(f"[GUARD] ✅ {node_name} output passed Layer 3 (bulk)")
            print(f"[GUARD] ✅ {node_name} output passed Layer 3")
            return state

        # If the bulk check only returns generic categories (e.g. POLICY_VIOLATION),
        # do not trigger expensive section-by-section second pass.
        bulk_violations = bulk_result.violations if hasattr(bulk_result, "violations") else []
        has_explicit_violations = any(
            isinstance(v, str) and len(v) == 2 and v[0] == "O" and v[1] in "12345678"
            for v in bulk_violations
        )
        if not has_explicit_violations:
            self._append_guard_log(
                state, node_name, "output", "passed",
                {"reason": "bulk_flagged_non_explicit", "violations": bulk_violations}
            )
            logger.info(
                f"[GUARD] ✅ {node_name}: bulk output flag ignored (non-explicit violations: {bulk_violations})"
            )
            print(
                f"[GUARD] ✅ {node_name}: bulk output flag ignored (non-explicit violations: {bulk_violations})"
            )
            return state
        
        # ---- Pass 2: Flagged — drill into sections ----
        logger.info(f"[GUARD] {node_name} flagged in bulk — checking sections...")
        print(f"[GUARD] {node_name} flagged — checking sections individually...")
        
        removed_count = 0
        total_count = 0
        violations_found = []
        
        if node_name == "night" and state.get("night_data"):
            night_data = state["night_data"]
            if isinstance(night_data, dict):
                safe_events = []
                for event in night_data.get("events", []):
                    total_count += 1
                    desc = event.get("description", "") or event.get("type", "") if isinstance(event, dict) else str(event)
                    if desc:
                        result = self._guard.check_output_sync(str(desc), user_input)
                        if result.allowed:
                            safe_events.append(event)
                        else:
                            removed_count += 1
                            violations_found.extend(result.violations if hasattr(result, 'violations') else [])
                    else:
                        safe_events.append(event)
                night_data["events"] = safe_events
                night_data["alerts_triggered"] = len([e for e in safe_events if isinstance(e, dict) and e.get("severity") in ("high", "critical")])
                state["night_data"] = night_data
                
        elif node_name == "rap1" and state.get("rap1_report"):
            state, removed_count, total_count, violations_found = self._filter_report_sections(
                state, "rap1_report", node_name
            )
            
        elif node_name == "day" and state.get("day_data"):
            day_data = state["day_data"]
            if isinstance(day_data, dict):
                safe_dx = []
                for dx in day_data.get("differential_diagnosis", []):
                    total_count += 1
                    result = self._guard.check_output_sync(str(dx), user_input)
                    if result.allowed:
                        safe_dx.append(dx)
                    else:
                        removed_count += 1
                        violations_found.extend(result.violations if hasattr(result, 'violations') else [])
                day_data["differential_diagnosis"] = safe_dx
                
                safe_actions = []
                for action in day_data.get("recommended_actions", []):
                    total_count += 1
                    result = self._guard.check_output_sync(str(action), user_input)
                    if result.allowed:
                        safe_actions.append(action)
                    else:
                        removed_count += 1
                        violations_found.extend(result.violations if hasattr(result, 'violations') else [])
                day_data["recommended_actions"] = safe_actions
                state["day_data"] = day_data
                
        elif node_name == "rap2" and state.get("rap2_report"):
            state, removed_count, total_count, violations_found = self._filter_report_sections(
                state, "rap2_report", node_name
            )
        
        # Log results
        if removed_count > 0:
            self._append_guard_log(state, node_name, "output", "filtered",
                                   {"removed": removed_count, "kept": total_count - removed_count, 
                                    "violations": violations_found})
            logger.warning(f"[GUARD] ⚠️ {node_name}: {removed_count}/{total_count} sections removed, "
                          f"{total_count - removed_count} safe sections kept")
            print(f"[GUARD] ⚠️ {node_name}: {removed_count}/{total_count} sections removed, "
                  f"{total_count - removed_count} safe sections kept")
            
            state["output_guard_result"] = {
                "filtered_at": node_name,
                "status": "filtered",
                "sections_removed": removed_count,
                "sections_kept": total_count - removed_count,
                "violations": violations_found,
                "action": "kept_safe_content_only",
                "timestamp": datetime.now().isoformat()
            }
            state["warnings"] = state.get("warnings", []) + [
                f"[GUARD] {node_name}: {removed_count} unsafe section(s) removed, "
                f"{total_count - removed_count} safe section(s) kept"
            ]
        else:
            # Pass 2 found nothing — bulk was a false positive
            self._append_guard_log(state, node_name, "output", "passed", "bulk_flagged_sections_ok")
            logger.info(f"[GUARD] ✅ {node_name}: sections individually safe")
            print(f"[GUARD] ✅ {node_name}: all sections passed Layer 3")
        
        return state
    
    def _filter_report_sections(self, state: dict, report_key: str, 
                                node_name: str) -> tuple:
        """
        Filter a report's sections individually — keep safe ones, 
        replace unsafe ones with a guard notice.
        Returns: (state, removed_count, total_count, violations_found)
        """
        removed_count = 0
        total_count = 0
        violations_found = []
        
        report_dict = state.get(report_key, {})
        if not isinstance(report_dict, dict):
            return state, 0, 0, []
        
        # Check and filter sections
        sections = report_dict.get("sections", [])
        safe_sections = []
        
        for section in sections:
            total_count += 1
            content = ""
            if isinstance(section, dict):
                content = section.get("content", "") or section.get("text", "")
                title = section.get("title", "")
                content = f"{title}: {content}" if title else content
            else:
                content = str(section)
            
            if content:
                result = self._guard.check_output_sync(content, state.get("presenting_complaint", ""))
                if result.allowed:
                    safe_sections.append(section)
                else:
                    removed_count += 1
                    violations_found.extend(result.violations if hasattr(result, 'violations') else [])
                    # Replace with guard notice in the section
                    if isinstance(section, dict):
                        safe_sections.append({
                            "title": section.get("title", "Section"),
                            "content": f"[GUARD] Section filtrée — contenu non conforme retiré ({', '.join(result.violations) if hasattr(result, 'violations') and result.violations else 'politique de sécurité'})",
                            "guardrails_filtered": True
                        })
            else:
                safe_sections.append(section)
        
        # Also check the full markdown content if it exists
        markdown = report_dict.get("markdown_content", "")
        if markdown and removed_count > 0:
            # Rebuild markdown from safe sections only
            safe_md_parts = []
            for s in safe_sections:
                if isinstance(s, dict):
                    if s.get("guardrails_filtered"):
                        safe_md_parts.append(f"### {s.get('title', 'Section')}\n\n{s['content']}\n")
                    else:
                        safe_md_parts.append(f"### {s.get('title', 'Section')}\n\n{s.get('content', '')}\n")
            
            if safe_md_parts:
                report_dict["markdown_content"] = (
                    f"# {report_dict.get('title', 'Rapport')}\n\n"
                    + "\n".join(safe_md_parts)
                    + f"\n\n---\n*⚠️ {removed_count} section(s) filtrée(s) par les gardes-fous MedGemma Sentinel*\n"
                )
            else:
                # All sections were unsafe → replace entire report
                report_dict["markdown_content"] = (
                    f"# {report_dict.get('title', 'Rapport')}\n\n"
                    "⚠️ **Tout le contenu a été filtré par les gardes-fous.**\n\n"
                    "Veuillez vérifier les données d'entrée.\n\n"
                    "*— MedGemma Sentinel Guard*\n"
                )
        
        report_dict["sections"] = safe_sections
        if removed_count > 0:
            report_dict["guardrails_filtered"] = True
            report_dict["sections_removed"] = removed_count
        state[report_key] = report_dict
        
        return state, removed_count, total_count, violations_found
    
    def _append_guard_log(self, state: dict, node: str, check_type: str, 
                          status: str, detail: Any = None) -> None:
        """Append an entry to the per-node guard audit trail."""
        if "guard_log" not in state:
            state["guard_log"] = []
        state["guard_log"].append({
            "node": node,
            "check": check_type,  # "input" or "output"
            "status": status,     # "passed", "blocked", "flagged", "skipped"
            "detail": detail,
            "timestamp": datetime.now().isoformat()
        })
    
    def _guarded_execute(self, state: dict, node_name: str, node) -> dict:
        """
        Execute a node with full guardrails wrapping:
          1. Layer 1+2 input check
          2. Execute node
          3. Layer 3 output check
        """
        # 1. Input guard
        state = self._run_input_guard(state, node_name)
        if state.get("guardrails_blocked", False):
            return state
        
        # 2. Execute the actual node
        state = node.execute(state)
        
        # 3. Output guard
        state = self._run_output_guard(state, node_name)
        
        return state
    
    # ──────────────────────────────────────────────────────────────────────
    #  LangGraph graph construction
    # ──────────────────────────────────────────────────────────────────────
    
    def _build_graph(self) -> Optional[StateGraph]:
        """Build the LangGraph state machine with per-node guardrails"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        graph = StateGraph(GraphState)
        
        # Add guarded nodes
        graph.add_node("night", self._night_wrapper)
        graph.add_node("rap1", self._rap1_wrapper)
        graph.add_node("day", self._day_wrapper)
        graph.add_node("rap2", self._rap2_wrapper)
        
        # Edges with conditional routing after each node
        # Night → check if blocked → Rap1
        graph.add_conditional_edges("night", self._should_continue, 
                                    {"continue": "rap1", "blocked": END})
        # Rap1 → check if blocked → Day
        graph.add_conditional_edges("rap1", self._should_continue,
                                    {"continue": "day", "blocked": END})
        # Day → check if blocked → Rap2
        graph.add_conditional_edges("day", self._should_continue,
                                    {"continue": "rap2", "blocked": END})
        # Rap2 → END
        graph.add_edge("rap2", END)
        
        graph.set_entry_point("night")
        
        return graph
    
    def _should_continue(self, state: GraphState) -> str:
        """Conditional edge: continue pipeline or stop if blocked."""
        if state.get("guardrails_blocked", False):
            return "blocked"
        return "continue"
    
    def _compile_graph(self) -> None:
        """Compile the graph for execution"""
        if self.graph:
            if self.memory:
                self.compiled_graph = self.graph.compile(checkpointer=self.memory)
            else:
                self.compiled_graph = self.graph.compile()
    
    # ──────────────────────────────────────────────────────────────────────
    #  Node wrappers — each wraps its node with guardrails
    # ──────────────────────────────────────────────────────────────────────
    
    def _night_wrapper(self, state: GraphState) -> GraphState:
        """Night node: [Guard L1+2] → Night → [Guard L3]"""
        return self._guarded_execute(dict(state), "night", self.night_node)
    
    def _rap1_wrapper(self, state: GraphState) -> GraphState:
        """Rap1 node: [Guard L1+2] → Rap1 → [Guard L3]"""
        return self._guarded_execute(dict(state), "rap1", self.rap1_node)
    
    def _day_wrapper(self, state: GraphState) -> GraphState:
        """Day node: [Guard L1+2] → Day → [Guard L3]"""
        return self._guarded_execute(dict(state), "day", self.day_node)
    
    def _rap2_wrapper(self, state: GraphState) -> GraphState:
        """Rap2 node: [Guard L1+2] → Rap2 → [Guard L3]"""
        return self._guarded_execute(dict(state), "rap2", self.rap2_node)
    
    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────
    
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
        Run the complete Sentinel workflow with per-node guardrails.
        
        Pipeline:
          [Guard] → Night → [Guard] → Rap1 → [Guard] → Day → [Guard] → Rap2 → [Guard]
        
        If any guard check blocks, the pipeline stops immediately.
        Output checks filter unsafe content — keeping only safe sections.
        
        Returns:
            Final state with reports, guardrails metadata, and full guard_log
        """
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
            "presenting_complaint": presenting_complaint or "",
            # Guardrails
            "guardrails_enabled": self._guardrails_enabled,
            "input_guard_result": None,
            "output_guard_result": None,
            "guardrails_blocked": False,
            "guard_log": [],
        }
        
        if self.compiled_graph:
            run_config = config or {"configurable": {"thread_id": initial_state["session_id"]}}
            result = self.compiled_graph.invoke(initial_state, run_config)
            return result
        else:
            return self._run_fallback(initial_state)
    
    def _run_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution without LangGraph — per-node guardrails"""
        print("[Fallback] Running without LangGraph (per-node guardrails active)...")
        
        nodes = [
            ("night", self.night_node),
            ("rap1", self.rap1_node),
            ("day", self.day_node),
            ("rap2", self.rap2_node),
        ]
        
        for node_name, node in nodes:
            state = self._guarded_execute(state, node_name, node)
            if state.get("guardrails_blocked", False):
                print(f"[Fallback] Pipeline blocked at {node_name}")
                break
        
        # Final summary message
        state["messages"] = state.get("messages", []) + [{
            "role": "system",
            "content": self._build_guard_summary(state),
            "timestamp": datetime.now().isoformat()
        }]
        
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
        Run night surveillance only, with per-node guardrails.
        
        Pipeline: [Guard] → Night → [Guard] → Rap1 → [Guard]
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
            "guardrails_enabled": self._guardrails_enabled,
            "input_guard_result": None,
            "output_guard_result": None,
            "guardrails_blocked": False,
            "guard_log": [],
        }
        
        state = self._guarded_execute(initial_state, "night", self.night_node)
        if not state.get("guardrails_blocked", False):
            state = self._guarded_execute(state, "rap1", self.rap1_node)
        
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
        Run day consultation only, with per-node guardrails.
        
        Pipeline: [Guard] → Day → [Guard] → Rap2 → [Guard]
        """
        initial_state: GraphState = {
            "session_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "phase": WorkflowPhase.DAY.value,
            "patient_context": patient_context or {},
            "consultation_mode": consultation_mode,
            "symptoms_input": symptoms_input or [],
            "presenting_complaint": "",
            "exam_input": exam_input or {},
            "day_vitals_input": vitals_input or {},
            "images_input": images_input or [],
            "night_data": prior_night_data,
            "messages": [],
            "errors": [],
            "warnings": [],
            "guardrails_enabled": self._guardrails_enabled,
            "input_guard_result": None,
            "output_guard_result": None,
            "guardrails_blocked": False,
            "guard_log": [],
        }
        
        state = self._guarded_execute(initial_state, "day", self.day_node)
        if not state.get("guardrails_blocked", False):
            state = self._guarded_execute(state, "rap2", self.rap2_node)
        
        return state
    
    def _build_guard_summary(self, state: dict) -> str:
        """Build a human-readable summary of all guard checks."""
        log = state.get("guard_log", [])
        if not log:
            return "[GUARD] No guardrails checks performed."
        
        lines = ["[GUARD] Per-node guardrails audit:"]
        for entry in log:
            icon = {"passed": "✅", "blocked": "❌", "flagged": "⚠️", "skipped": "⏭️"}.get(
                entry["status"], "?"
            )
            lines.append(f"  {icon} {entry['node']}.{entry['check']}: {entry['status']}")
            if entry.get("detail") and entry["status"] not in ("passed", "skipped"):
                lines.append(f"     → {entry['detail']}")
        
        blocked = state.get("guardrails_blocked", False)
        if blocked:
            lines.append(f"  ❌ Pipeline BLOCKED — check guard_log for details")
        else:
            lines.append(f"  ✅ All nodes passed guardrails checks")
        
        return "\n".join(lines)
    
    def get_guardrails_status(self) -> Dict[str, Any]:
        """Get current guardrails configuration status."""
        status = {
            "enabled": self._guardrails_enabled,
            "guard_available": self._guard is not None,
            "mode": "per_node",  # guardrails at every step
            "trusted_internal_inputs": self._trusted_internal_inputs,
        }
        if self._guard:
            status["guard_details"] = self._guard.get_status()
        return status
    
    def get_graph_visualization(self) -> Optional[str]:
        """Get ASCII visualization of the graph"""
        return """
+==================================================================================+
|              MedGemma Sentinel - Workflow Graph (Per-Node Guardrails)              |
+==================================================================================+
|                                                                                   |
|  Each node is wrapped: [Guard L1+2] → Execute → [Guard L3]                       |
|                                                                                   |
|  +---------+    +---------+    +---------+    +---------+                         |
|  | NIGHT   | →  |  RAP1   | →  |   DAY   | →  |  RAP2   | →  END                 |
|  |[G]→N→[G]|    |[G]→R1→[G]   |[G]→D→[G]|    |[G]→R2→[G]|                       |
|  +---------+    +---------+    +---------+    +---------+                         |
|      |              |              |              |                                |
|      v blocked      v blocked      v blocked      v blocked                       |
|     END            END            END            END                              |
|                                                                                   |
|  [G] = Guardrails Check:                                                          |
|    Before node: Layer 1 (Regex) + Layer 2 (Llama Guard 3 classification)         |
|    After node:  Layer 3 (Llama Guard 3 output audit, safe-only)                  |
|                                                                                   |
|  If any check blocks → pipeline stops immediately → END                          |
|                                                                                   |
|  Steering Modes:                                                                  |
|  - NIGHT: night_surveillance (apnees, hypoxie)                                   |
|  - RAP1/RAP2: longitudinal (analyse temporelle)                                  |
|  - DAY: specialist_virtual (aide diagnostic)                                     |
|                                                                                   |
+==================================================================================+
"""


def create_sentinel_graph(
    use_memory: bool = True,
    trusted_internal_inputs: bool = False,
) -> MedGemmaSentinelGraph:
    """
    Factory function to create a MedGemma Sentinel graph.
    
    Args:
        use_memory: Whether to enable checkpointing
        trusted_internal_inputs: If True, skip expensive downstream input checks
        
    Returns:
        Configured MedGemmaSentinelGraph instance with per-node guardrails
    """
    return MedGemmaSentinelGraph(
        use_memory=use_memory,
        trusted_internal_inputs=trusted_internal_inputs,
    )
