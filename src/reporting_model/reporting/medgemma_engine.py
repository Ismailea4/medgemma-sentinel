"""
MedGemma Engine - Local & Remote LLM Inference
Uses hmzBen/medgemma-1.5-medical-q4km (Q4_K_M quantized GGUF)

Supports 4 modes (in priority order):
1. Local llama.cpp server (if running on localhost:8080)
2. llama-cpp-python direct loading (if llama_cpp package installed)
3. HuggingFace Inference API (remote, needs HF_TOKEN)
4. Simulation fallback (for dev/testing)

Guardrails: NeMo Guardrails + Llama Guard 3 integration for input/output safety.
"""

import os
import json
import glob
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# HuggingFace model - use the base model for inference API
HF_REPO_ID = "google/medgemma-1.5-4b-it"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_REPO_ID}/v1/chat/completions"


class MedGemmaEngine:
    """
    MedGemma 1.5 Medical LLM Engine (4B)
    Supports llama-server, llama-cpp-python, HuggingFace API, or simulation.
    """

    # Default locations to look for GGUF model files
    MODEL_SEARCH_PATHS = [
        "models",
        ".",
        "../models",
    ]
    MODEL_GLOB = "*medgemma*.gguf"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        server_url: Optional[str] = None,
        model_path: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = -1,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
    ):
        """
        Initialize MedGemma engine.

        Args:
            hf_token: HuggingFace API token (or set HF_TOKEN env var)
            server_url: URL of local llama.cpp server (optional)
            model_path: Path to GGUF model file for llama-cpp-python (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            n_ctx: Context window size for llama-cpp-python
            n_gpu_layers: Layers to offload to GPU (0 = CPU only)
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.server_url = server_url
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens != -1 else None
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

        self.is_loaded = False
        self.mode = "simulation"
        self._llm = None  # llama-cpp-python Llama instance

        # Initialize guardrails (NeMo Guardrails + Llama Guard)
        self._guard = None
        self._guardrails_enabled = False
        self._init_guardrails()

        # Try local llama.cpp server first (preferred)
        self._check_local_server(server_url or "http://localhost:8080")

        # Try llama-cpp-python direct loading
        if not self.is_loaded:
            self._check_llama_cpp_python(model_path)

        # Fallback: try HuggingFace Inference API
        if not self.is_loaded and self.hf_token:
            self._check_hf_api()

        if not self.is_loaded:
            logger.info("Running in simulation mode (start llama-server, install llama-cpp-python, or set HF_TOKEN)")

    def _check_hf_api(self):
        """Verify HuggingFace API access"""
        try:
            import requests
            response = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {self.hf_token}"},
                timeout=5,
            )
            if response.status_code == 200:
                user = response.json().get("name", "unknown")
                logger.info(f"HuggingFace API connected as: {user}")
                print(f"[OK] HuggingFace API connected (user: {user})")
                self.is_loaded = True
                self.mode = "huggingface"
            else:
                logger.warning(f"HF API auth failed: {response.status_code}")
                print(f"[WARN] HF token invalid (status {response.status_code})")
        except Exception as e:
            logger.warning(f"Cannot reach HuggingFace API: {e}")
            print(f"[WARN] Cannot reach HuggingFace API: {e}")

    def _check_local_server(self, url: str):
        """Check if local llama.cpp server is running"""
        try:
            import requests
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"Connected to local server at {url}")
                print(f"[OK] Connected to local llama.cpp server at {url}")
                self.server_url = url
                self.is_loaded = True
                self.mode = "server"
        except Exception:
            pass

    def _find_gguf_model(self) -> Optional[str]:
        """Search for a GGUF model file in common locations"""
        # Check from workspace root
        for search_dir in self.MODEL_SEARCH_PATHS:
            pattern = str(Path(search_dir) / self.MODEL_GLOB)
            matches = glob.glob(pattern)
            # Prefer Q4_K_M if multiple matches
            q4_matches = [m for m in matches if 'Q4_K_M' in m.upper()]
            if q4_matches:
                return q4_matches[0]
            if matches:
                return matches[0]
        return None

    def _check_llama_cpp_python(self, model_path: Optional[str] = None):
        """Try loading model directly via llama-cpp-python"""
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.debug("llama-cpp-python not installed, skipping direct loading")
            return

        # Find model file
        path = model_path or os.environ.get("MEDGEMMA_MODEL_PATH") or self._find_gguf_model()
        if not path or not Path(path).exists():
            logger.debug(f"No GGUF model file found (searched: {self.MODEL_SEARCH_PATHS})")
            return

        try:
            print(f"[...] Loading model via llama-cpp-python: {Path(path).name}")
            self._llm = Llama(
                model_path=str(path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            self.model_path = str(path)
            self.is_loaded = True
            self.mode = "llama-cpp-python"
            print(f"[OK] Model loaded via llama-cpp-python ({Path(path).name})")
            logger.info(f"Model loaded via llama-cpp-python: {path}")
        except Exception as e:
            logger.warning(f"Failed to load model via llama-cpp-python: {e}")
            print(f"[WARN] llama-cpp-python load failed: {e}")
            self._llm = None

    def _call_hf_api(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Call MedGemma via HuggingFace Inference API"""
        import requests

        # Try OpenAI-compatible chat completions endpoint first
        response = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": HF_REPO_ID,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]

        # Fallback: try the text-generation endpoint
        logger.info(f"Chat endpoint returned {response.status_code}, trying text-generation...")
        fallback_url = f"https://api-inference.huggingface.co/models/{HF_REPO_ID}"
        prompt = self._messages_to_prompt(messages)

        response2 = requests.post(
            fallback_url,
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "return_full_text": False,
                },
            },
            timeout=120,
        )

        if response2.status_code == 200:
            data = response2.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            return str(data)

        logger.error(f"HF API error {response2.status_code}: {response2.text[:300]}")
        return None

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string"""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model")
        return "\n".join(parts)

    def _call_server(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Call local llama.cpp server"""
        import requests

        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return None

    def _call_llama_cpp(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Call model directly via llama-cpp-python"""
        if not self._llm:
            return None

        try:
            kwargs = dict(messages=messages, temperature=self.temperature)
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            response = self._llm.create_chat_completion(**kwargs)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"llama-cpp-python call failed: {e}")
            return None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip chain-of-thought / thinking prefix from model output.
        Some models (gemma3 thinking variants) prefix output with
        reasoning, mental sandbox, constraint checklists before the actual report.
        """
        if not text:
            return text
        import re

        # Look for ```markdown block which wraps the actual report
        md_match = re.search(r'```markdown\s*\n', text)
        if md_match:
            result = text[md_match.end():]
            # Remove closing ```
            result = re.sub(r'\n```\s*$', '', result)
            return result.strip()

        # Look for report header patterns after any thinking content
        report_patterns = [
            r'\n(\*\*(?:Night Surveillance|Cardiology|Medical|Rapport|Consultation|Patient)[^\n]*\*\*)',
            r'\n(#{1,3}\s+(?:Night|Rapport|Consultation|Patient|Report))',
        ]
        for pat in report_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return text[m.start() + 1:].strip()

        # If text starts with thinking keywords, skip to first real header
        thinking_kw = ['thought', '**constraint', '**mental sandbox', '**key learning', 'strategizing']
        stripped = text.lstrip().lower()
        if any(stripped.startswith(p) for p in thinking_kw):
            m = re.search(r'\n(\*\*[A-Z][a-z])', text)
            if m:
                return text[m.start() + 1:].strip()

        return text.strip()

    def _init_guardrails(self):
        """Initialize NeMo Guardrails + Llama Guard (graceful fallback)."""
        try:
            from src.guardrails import SentinelGuard
            self._guard = SentinelGuard()
            self._guardrails_enabled = self._guard.enabled
        except ImportError:
            logger.info("Guardrails module not available")
        except Exception as e:
            logger.warning(f"Guardrails initialization failed: {e}")

    def enable_guardrails(self):
        """Enable guardrails (re-initializes if needed)."""
        if self._guard:
            self._guard.enable()
            self._guardrails_enabled = self._guard.enabled
        else:
            self._init_guardrails()

    def disable_guardrails(self):
        """Temporarily disable guardrails."""
        self._guardrails_enabled = False
        if self._guard:
            self._guard.disable()

    def _check_input_safety(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Check input messages against guardrails. Returns refusal message if blocked."""
        if not self._guardrails_enabled or not self._guard:
            return None

        # Extract the user message from the messages list
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        if not user_content:
            return None

        result = self._guard.check_input_sync(user_content)
        if not result.allowed:
            logger.warning(f"[GUARDRAILS] Input blocked. Violations: {result.violations}")
            return result.message
        return None

    def _check_output_safety(self, response: str, messages: List[Dict[str, str]]) -> str:
        """Check model output against guardrails. Returns sanitized response."""
        if not self._guardrails_enabled or not self._guard:
            return response

        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        result = self._guard.check_output_sync(response, user_content)
        if not result.allowed:
            logger.warning(f"[GUARDRAILS] Output blocked. Violations: {result.violations}")
            return result.message
        return response

    def _call_model(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Route to the correct backend, with guardrails input/output checks."""
        # --- Input guardrails check ---
        refusal = self._check_input_safety(messages)
        if refusal:
            return refusal

        try:
            result = None
            if self.mode == "huggingface":
                result = self._call_hf_api(messages)
            elif self.mode == "server":
                result = self._call_server(messages)
            elif self.mode == "llama-cpp-python":
                result = self._call_llama_cpp(messages)
            if result:
                result = self._strip_thinking(result)
                # --- Output guardrails check ---
                result = self._check_output_safety(result, messages)
                return result
        except Exception as e:
            logger.error(f"Model call failed: {e}")
        return None

    # ---- Report Generation ----

    def generate_night_report(
        self,
        patient_context: str,
        night_summary: str,
        events: List[Dict[str, Any]],
    ) -> str:
        """Generate clinical night surveillance report."""
        if not self.is_loaded:
            return self._simulate_night_report(patient_context, night_summary, events)

        prompt = self._build_night_prompt(patient_context, night_summary, events)
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a clinical expert. Generate a night surveillance report "
                    "in Markdown format with: Executive Summary, Event Timeline, "
                    "Vital Signs Analysis, Sleep Quality, Clinical Recommendations. "
                    "Be concise but thorough. Do NOT include any thinking or reasoning - "
                    "output ONLY the final report.\n\n" + prompt
                ),
            }
        ]

        result = self._call_model(messages)
        if result:
            return result
        return self._simulate_night_report(patient_context, night_summary, events)

    def generate_day_report(
        self,
        patient_context: str,
        night_context: str,
        consultation_data: Dict[str, Any],
        specialty: str = "general",
    ) -> str:
        """Generate clinical day consultation report."""
        if not self.is_loaded:
            return self._simulate_day_report(
                patient_context, night_context, consultation_data, specialty
            )

        prompt = self._build_day_prompt(
            patient_context, night_context, consultation_data, specialty
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are a {specialty} specialist. Generate a medical consultation "
                    "report in Markdown with: Patient ID, Night Summary, Chief Complaint, "
                    "Exam Findings, Differential Diagnoses, Investigations, Treatment Plan. "
                    "Use evidence-based reasoning. Do NOT include any thinking or reasoning - "
                    "output ONLY the final report.\n\n" + prompt
                ),
            }
        ]

        result = self._call_model(messages)
        if result:
            return result
        return self._simulate_day_report(
            patient_context, night_context, consultation_data, specialty
        )

    def analyze_symptoms(
        self, symptoms: List[str], patient_context: str
    ) -> Dict[str, Any]:
        """Analyze patient symptoms using MedGemma."""
        if not self.is_loaded:
            return self._simulate_symptom_analysis(symptoms, patient_context)

        prompt = (
            f"Patient context: {patient_context}\n\n"
            f"Reported symptoms:\n"
            + "\n".join(f"- {s}" for s in symptoms)
            + "\n\nAnalyze these symptoms. Provide:\n"
            "1. Differential diagnoses (with likelihood)\n"
            "2. Red flags to monitor\n"
            "3. Recommended investigations\n"
            "4. Initial management suggestions"
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        result = self._call_model(messages)
        if result:
            return {"status": "success", "analysis": result}
        return self._simulate_symptom_analysis(symptoms, patient_context)

    def analyze_history_evolution(
        self,
        patient_id: str,
        session_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze clinical evolution across recent session cycles.

        Args:
            patient_id: Patient identifier
            session_history: Chronological list of compact session snapshots

        Returns:
            Dict with status + markdown analysis text.
        """
        if not session_history or len(session_history) < 2:
            return {
                "status": "insufficient_history",
                "analysis": (
                    "## Evolution Inter-Cycles\n\n"
                    "- Historique insuffisant: au moins 2 sessions (2 cycles nuit+jour) sont requises."
                ),
            }

        if not self.is_loaded:
            return {
                "status": "simulated",
                "analysis": self._simulate_history_evolution(patient_id, session_history),
            }

        prompt = self._build_history_evolution_prompt(patient_id, session_history)
        messages = [
            {
                "role": "user",
                "content": (
                    "Tu es MedGemma, assistant clinique. Analyse l'evolution entre 2 cycles "
                    "nuit+jour consecutifs. Reponds UNIQUEMENT en Markdown avec ces sections:\n"
                    "## Synthese Evolution\n"
                    "## Insights Nuit\n"
                    "## Insights Jour\n"
                    "## Points de Vigilance\n"
                    "## Recommandations Pratiques\n\n"
                    "Contrainte: maximum 8 puces au total, concret, sans texte hors sections.\n\n"
                    + prompt
                ),
            }
        ]

        result = self._call_model(messages)
        if result:
            return {"status": "success", "analysis": result}

        return {
            "status": "fallback",
            "analysis": self._simulate_history_evolution(patient_id, session_history),
        }

    # ---- Prompt Builders ----

    @staticmethod
    def _build_night_prompt(
        patient_context: str,
        night_summary: str,
        events: List[Dict[str, Any]],
    ) -> str:
        events_text = "\n".join(
            f"- {e.get('timestamp', 'N/A')}: {e.get('type', 'Unknown')} "
            f"(Severity: {e.get('severity', 'N/A')}) - {e.get('description', '')}"
            for e in events
        )
        return (
            f"Patient Context:\n{patient_context}\n\n"
            f"Night Summary:\n{night_summary}\n\n"
            f"Detected Events:\n{events_text}"
        )

    @staticmethod
    def _build_day_prompt(
        patient_context: str,
        night_context: str,
        consultation_data: Dict[str, Any],
        specialty: str,
    ) -> str:
        # Make consultation_data JSON-serializable
        def _serialize(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return str(obj)

        return (
            f"Specialty: {specialty}\n\n"
            f"Patient Context:\n{patient_context}\n\n"
            f"Night Summary:\n{night_context}\n\n"
            f"Consultation Data:\n{json.dumps(consultation_data, indent=2, ensure_ascii=False, default=_serialize)}"
        )

    @staticmethod
    def _build_history_evolution_prompt(
        patient_id: str,
        session_history: List[Dict[str, Any]],
    ) -> str:
        """Build prompt payload for longitudinal evolution analysis."""
        history_json = json.dumps(
            session_history,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        return (
            f"Patient ID: {patient_id}\n\n"
            "Historique compact (ordre chronologique):\n"
            f"{history_json}\n\n"
            "Focalise sur: tendances des alertes, oxygenation (SpO2), FC, temperature, "
            "evolution des symptomes, et priorites de surveillance."
        )

    # ---- Simulation Fallbacks ----

    @staticmethod
    def _simulate_night_report(
        patient_context: str,
        night_summary: str,
        events: List[Dict[str, Any]],
    ) -> str:
        event_count = len(events)
        critical_count = sum(1 for e in events if e.get("severity") == "critical")
        return (
            "# Rapport de Surveillance Nocturne (Simulated)\n\n"
            "## Informations Generales\n"
            f"**Periode:** 21:00 - 07:00\n"
            f"**Total d'evenements:** {event_count}\n\n"
            f"## Resume Executif\n{night_summary}\n\n"
            f"## Alertes Detectees\n"
            f"- Alertes critiques: {critical_count}\n"
            f"- Evenements totaux: {event_count}\n\n"
            "## Recommandations\n"
            "- Surveillance continue recommandee\n"
            "- Suivi des constantes vitales\n"
            "- Reevaluation clinique le jour"
        )

    @staticmethod
    def _simulate_day_report(
        patient_context: str,
        night_context: str,
        consultation_data: Dict[str, Any],
        specialty: str,
    ) -> str:
        return (
            "# Rapport de Consultation Medicale (Simulated)\n\n"
            f"## Identification\n**Specialite:** {specialty}\n"
            "**Mode:** Consultation assistee\n\n"
            f"## Contexte Nocturne\n{night_context}\n\n"
            "## Motif de Consultation\nA evaluer cliniquement\n\n"
            "## Recommandations\n"
            "- Examens complementaires recommandes\n"
            "- Suivi specialise suggere\n"
            "- Reevaluation dans les 48h"
        )

    @staticmethod
    def _simulate_symptom_analysis(
        symptoms: List[str],
        patient_context: str,
    ) -> Dict[str, Any]:
        return {
            "status": "simulated",
            "analysis": (
                f"Symptome(s) detecte(s): {', '.join(symptoms)}\n\n"
                "Mode simulation (MedGemma non connecte).\n"
                "Pour utiliser le modele reel, definir HF_TOKEN:\n"
                "  set HF_TOKEN=hf_xxxxxxxxxxxx\n"
                "  (get token at https://huggingface.co/settings/tokens)"
            ),
        }

    @staticmethod
    def _simulate_history_evolution(
        patient_id: str,
        session_history: List[Dict[str, Any]],
    ) -> str:
        """Deterministic fallback for history evolution analysis."""
        previous = session_history[-2]
        current = session_history[-1]

        prev_events = int(previous.get("total_alerts", 0) or 0)
        curr_events = int(current.get("total_alerts", 0) or 0)
        delta_events = curr_events - prev_events

        prev_spo2 = previous.get("spo2_min")
        curr_spo2 = current.get("spo2_min")

        prev_symptoms = previous.get("symptoms", [])
        curr_symptoms = current.get("symptoms", [])

        trend_label = (
            "aggravation"
            if delta_events > 0
            else "amelioration"
            if delta_events < 0
            else "stabilite"
        )

        return (
            "## Synthese Evolution\n\n"
            f"- Patient {patient_id}: tendance globale = {trend_label} "
            f"(alertes {prev_events} -> {curr_events}).\n\n"
            "## Insights Nuit\n\n"
            f"- SpO2 minimale precedente: {prev_spo2 if prev_spo2 is not None else 'N/A'}%.\n"
            f"- SpO2 minimale actuelle: {curr_spo2 if curr_spo2 is not None else 'N/A'}%.\n\n"
            "## Insights Jour\n\n"
            f"- Symptomes precedents: {', '.join(prev_symptoms) if prev_symptoms else 'aucun documente'}.\n"
            f"- Symptomes actuels: {', '.join(curr_symptoms) if curr_symptoms else 'aucun documente'}.\n\n"
            "## Points de Vigilance\n\n"
            "- Surveiller toute baisse recurrente de SpO2 < 92%.\n"
            "- Correlier alertes nocturnes et plainte clinique diurne.\n\n"
            "## Recommandations Pratiques\n\n"
            "- Prioriser reevaluation cardio-respiratoire si progression des alertes.\n"
            "- Revoir les constantes et le plan de suivi au prochain cycle."
        )

    def get_status(self) -> Dict[str, Any]:
        """Get engine status including guardrails information."""
        model_info = None
        if self.mode == "server":
            model_info = self.server_url
        elif self.mode == "llama-cpp-python":
            model_info = self.model_path
        elif self.mode == "huggingface":
            model_info = HF_REPO_ID

        status = {
            "loaded": self.is_loaded,
            "model_path": model_info,
            "server_url": self.server_url if self.mode == "server" else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "mode": self.mode,
            "guardrails_enabled": self._guardrails_enabled,
        }

        if self._guard:
            status["guardrails"] = self._guard.get_status()

        return status
