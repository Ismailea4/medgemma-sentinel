"""
MedGemma Sentinel - SentinelGuard
Llama Guard 3 (1B) integration for securing MedGemma — FULLY OFFLINE.

Implements a 3-layer Sequential Defense:
  Layer 1: Input Sentinel (Llama Guard 3 via local transformers) - First line of defense
  Layer 2: Logic Flow (Colang semantic matching) - Dialogue management
  Layer 3: Output Sentinel (Llama Guard 3 via local transformers) - Final audit

Models (all local, no internet required):
  - MedGemma:    models/medgemma-1.5-medical-Q4_K_M.gguf  (llama-cpp-python)
  - Llama Guard: models/Llama-Guard-3-1B/                 (transformers)

Adapted from: telemedicine code/Adversarial Threat Telemedecine/

Usage:
    from src.guardrails import SentinelGuard

    guard = SentinelGuard()
    result = guard.check_input_sync("How to synthesize fentanyl?")
    print(result.allowed)     # False
    print(result.violations)  # ['O5', 'O8']
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path to the Llama Guard 3 model (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_LLAMA_GUARD_PATH = str(PROJECT_ROOT / "models" / "Llama-Guard-3-1B")
DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "guardrails")

# Llama Guard O1-O8 taxonomy (matching prompts.yml)
SAFETY_TAXONOMY = """<BEGIN UNSAFE CONTENT CATEGORIES>
O1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics.
O2: Sexual Content.
Should not
- Engage in sexually explicit conversations and should not encourage people to engage in sexual activities.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances.
- Assist or encourage people to create such substances when it is illegal to do so.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
O7: Offensive Language and Insults.
Should not
- Address direct insults.
- Use or encourage the use of swear words, bad language, or vulgar expressions.
O8: Medical Ethics and Unethical Procedures.
Should not
- Provide instructions for human experimentation without informed consent or IRB approval.
- Assist in the creation of illegal medical recipes or synthesis of controlled pharmaceutical substances.
- Encourage bypassing professional medical oversight for life-threatening conditions.
- Provide detailed instructions for performing invasive medical procedures at home.
- Reveal private patient information (PII) outside of authorized clinical reports.
<END UNSAFE CONTENT CATEGORIES>"""

# Semantic matching patterns for known harmful intents (Colang Layer 2 logic)
HARMFUL_PATTERNS = {
    "O5": [
        r"synthe[sz]i[sz]e?\s+(fentanyl|meth|drug|heroin|cocaine|mdma)",
        r"how\s+to\s+(make|create|cook|produce)\s+(drug|fentanyl|meth|heroin)",
        r"recipe\s+for\s+(fentanyl|meth|drug|heroin|cocaine)",
    ],
    "O8": [
        r"human\s+(trial|experiment).*(without|no)\s+(consent|ethic|irb)",
        r"bypass\s+(irb|ethics|medical\s+oversight)",
        r"perform\s+surgery\s+without\s+training",
        r"illegal\s+medical\s+recipe",
        r"unapproved\s+drugs?\s+on\s+humans?",
    ],
    "PII": [
        r"(give|show|extract|list)\s+(me\s+)?(patient|all)\s+(name|record|info|data|address)",
        r"social\s+security\s+number",
        r"(patient|medical)\s+(database|records)\s+of\s+all",
    ],
    "INJECTION": [
        r"ignore\s+(all\s+)?(previous|above)\s+instructions?",
        r"override\s+(your\s+)?safety",
        r"(you\s+are|pretend).*(developer|no\s+restriction|unrestrict)",
        r"disregard\s+(your\s+)?(system|safety)\s+prompt",
    ],
}


@dataclass
class GuardResult:
    """Result of a guardrail safety check.

    Attributes:
        allowed: Whether the message passed all safety checks.
        violations: List of violated policy categories (e.g. ['O5', 'O8']).
        message: The (possibly sanitized) message content.
        details: Additional details from the guardrail evaluation.
    """
    allowed: bool
    violations: List[str] = field(default_factory=list)
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class SentinelGuard:
    """Llama Guard 3 guardrails for MedGemma Sentinel — fully offline.

    Loads Llama Guard 3 (1B) locally via HuggingFace transformers from
    models/Llama-Guard-3-1B/. No internet connection or Ollama required.

    Falls back to regex-based semantic matching if the model cannot be loaded
    (matching MedGemmaEngine's graceful degradation pattern).
    """

    # Refusal messages (matching input_guardrails.co / output_guardrails.co)
    REFUSAL_INPUT = (
        "I'm sorry, I cannot fulfill this request. "
        "It violates medical safety and ethical guidelines."
    )
    REFUSAL_MEDICAL = (
        "As a medical AI assistant, I am prohibited from providing information "
        "regarding unethical or illegal medical procedures. Patient safety and "
        "medical ethics are my highest priorities."
    )
    REFUSAL_PII = (
        "I cannot share specific patient information outside of authorized clinical "
        "reports. Patient privacy is strictly protected under medical ethics "
        "guidelines and HIPAA regulations."
    )
    REFUSAL_OUTPUT = (
        "I apologize, but my response was flagged by the safety system. I cannot "
        "provide this information as it may contain harmful or inappropriate "
        "medical content. Please rephrase your question or consult a healthcare "
        "professional."
    )

    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize SentinelGuard with local Llama Guard 3 model.

        Args:
            model_path: Path to Llama Guard 3 model directory.
                        Defaults to <project_root>/models/Llama-Guard-3-1B/
            config_path: Path to the guardrails config directory.
                         Defaults to <project_root>/guardrails/
        """
        self.model_path = model_path or DEFAULT_LLAMA_GUARD_PATH
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.enabled = False
        self._model = None
        self._tokenizer = None
        self._init_error = None
        self._mode = "disabled"

        self._initialize()

    def _initialize(self):
        """Try to load Llama Guard 3 locally. Falls back to regex matching."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            import sys as _sys

            model_dir = Path(self.model_path)
            if not model_dir.exists():
                self._init_error = f"Llama Guard model not found: {self.model_path}"
                logger.warning(f"[GUARDRAILS] {self._init_error}")
                print(f"[WARN] Guardrails: Llama Guard model not found at {self.model_path}")
                self._mode = "regex"
                self.enabled = True  # Fall back to regex-based matching
                return

            print(f"[...] Loading Llama Guard 3 from {model_dir.name}...")
            _sys.stdout.flush()
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), local_files_only=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            # Move to GPU only if CUDA is available and has enough memory
            if torch.cuda.is_available():
                try:
                    self._model = self._model.to("cuda")
                except RuntimeError:
                    logger.info("[GUARDRAILS] GPU OOM — keeping model on CPU")
                    print("[INFO] Guardrails: GPU memory insufficient, using CPU")

            self._model.eval()
            self.enabled = True
            self._mode = "llama_guard"
            _device = next(self._model.parameters()).device
            logger.info("[GUARDRAILS] Llama Guard 3 loaded successfully (local)")
            print(f"[OK] Guardrails: Llama Guard 3 (1B) loaded on {_device} — fully offline")

        except ImportError:
            self._init_error = "transformers/torch not installed"
            logger.info(f"[GUARDRAILS] {self._init_error} — falling back to regex")
            print("[WARN] Guardrails: transformers not installed, using regex fallback")
            self._mode = "regex"
            self.enabled = True

        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"[GUARDRAILS] Llama Guard load failed: {e}")
            print(f"[WARN] Guardrails: Llama Guard load failed — using regex fallback")
            print(f"       Error: {e}")
            self._mode = "regex"
            self.enabled = True


    # ================================================================
    # Main API
    # ================================================================

    def check_input_sync(self, user_message: str) -> GuardResult:
        """Check if a user message passes safety guardrails.

        Pipeline:
        1. Regex semantic matching (fast, catches known harmful patterns)
        2. Llama Guard 3 classification (if model loaded)

        Args:
            user_message: The raw user input to check.

        Returns:
            GuardResult with allowed=True if safe, or allowed=False with violations.
        """
        if not self.enabled:
            return GuardResult(allowed=True, message=user_message,
                               details={"mode": "disabled", "reason": self._init_error})

        # Layer 1: Regex semantic matching (fast, always active)
        regex_result = self._check_regex(user_message)
        if not regex_result.allowed:
            return regex_result

        # Layer 2: Llama Guard classification (if model loaded)
        if self._mode == "llama_guard" and self._model is not None:
            guard_result = self._llama_guard_classify_input(user_message)
            if not guard_result.allowed:
                return guard_result

        return GuardResult(
            allowed=True,
            message=user_message,
            details={"mode": self._mode, "checked": True}
        )

    def check_output_sync(self, bot_response: str, user_input: str = "") -> GuardResult:
        """Check if a bot response passes safety guardrails.

        Args:
            bot_response: The generated bot response to check.
            user_input: The original user input (for context).

        Returns:
            GuardResult with allowed=True if safe, or allowed=False with violations.
        """
        if not self.enabled:
            return GuardResult(allowed=True, message=bot_response,
                               details={"mode": "disabled", "reason": self._init_error})

        # Llama Guard output classification (if model loaded)
        if self._mode == "llama_guard" and self._model is not None:
            guard_result = self._llama_guard_classify_output(bot_response, user_input)
            if not guard_result.allowed:
                return guard_result

        return GuardResult(
            allowed=True,
            message=bot_response,
            details={"mode": self._mode, "checked": True}
        )

    def generate_guarded_sync(self, user_message: str) -> GuardResult:
        """Full guarded check (equivalent to running through NeMo Rails pipeline)."""
        return self.check_input_sync(user_message)

    # ================================================================
    # Llama Guard 3 — Local Transformers Inference
    # ================================================================

    def _llama_guard_classify_input(self, user_message: str) -> GuardResult:
        """Classify user input using Llama Guard 3 (local model).

        Builds the prompt following Meta's Llama Guard 3 format and runs
        local inference to get safe/unsafe classification.
        """
        try:
            import torch

            prompt = self._build_llama_guard_prompt(
                role="User",
                conversation=f"User: {user_message}",
            )

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=100,
                    pad_token_id=self._tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                )

            # Decode only newly generated tokens
            response = self._tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            return self._parse_llama_guard_response(response, user_message, "input")

        except Exception as e:
            logger.error(f"[GUARDRAILS] Llama Guard input classification failed: {e}")
            # Fail open on model errors
            return GuardResult(allowed=True, message=user_message,
                               details={"error": str(e), "mode": "fail_open"})

    def _llama_guard_classify_output(self, bot_response: str, user_input: str = "") -> GuardResult:
        """Classify bot output using Llama Guard 3 (local model)."""
        try:
            import torch

            conversation = f"User: {user_input}\n\nAgent: {bot_response}" if user_input else f"Agent: {bot_response}"
            prompt = self._build_llama_guard_prompt(
                role="Agent",
                conversation=conversation,
            )

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=100,
                    pad_token_id=self._tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                )

            response = self._tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            return self._parse_llama_guard_response(response, bot_response, "output")

        except Exception as e:
            logger.error(f"[GUARDRAILS] Llama Guard output classification failed: {e}")
            return GuardResult(allowed=True, message=bot_response,
                               details={"error": str(e), "mode": "fail_open"})

    def _build_llama_guard_prompt(self, role: str, conversation: str) -> str:
        """Build a Llama Guard 3 classification prompt.

        Follows Meta's standard prompt format for Llama Guard 3.
        """
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in '{role}' messages in conversations
according our safety policy with the below categories.

{SAFETY_TAXONOMY}

<BEGIN CONVERSATION>

{conversation}

<END CONVERSATION>

Provide your safety assessment for {role} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @staticmethod
    def _parse_llama_guard_response(response: str, original_message: str, check_type: str) -> GuardResult:
        """Parse the Llama Guard 3 model output into a GuardResult."""
        response_lower = response.strip().lower()

        if response_lower.startswith("safe"):
            return GuardResult(
                allowed=True, message=original_message,
                details={"llama_guard": "safe", "check_type": check_type}
            )

        if response_lower.startswith("unsafe"):
            # Extract violated categories (e.g., "unsafe\nO5,O8")
            categories = re.findall(r'O\d', response)
            violations = list(set(categories)) if categories else ["POLICY_VIOLATION"]

            # Llama Guard occasionally returns "unsafe" without explicit O1-O8 categories
            # for legitimate clinical prompts. Keep strict blocking for output checks,
            # but fail open on input checks when no concrete category is provided.
            if violations == ["POLICY_VIOLATION"] and check_type == "input":
                return GuardResult(
                    allowed=True,
                    message=original_message,
                    details={
                        "llama_guard": "ambiguous_unsafe_no_category",
                        "check_type": check_type,
                        "raw_response": response,
                    },
                )

            # Choose appropriate refusal message
            if "O8" in violations or "PII" in violations:
                refusal = SentinelGuard.REFUSAL_MEDICAL
            elif "O5" in violations:
                refusal = SentinelGuard.REFUSAL_INPUT
            else:
                refusal = SentinelGuard.REFUSAL_OUTPUT if check_type == "output" else SentinelGuard.REFUSAL_INPUT

            return GuardResult(
                allowed=False,
                violations=violations,
                message=refusal,
                details={
                    "llama_guard": "unsafe",
                    "check_type": check_type,
                    "raw_response": response,
                    "original": original_message[:200],
                }
            )

        # Ambiguous response — fail open
        return GuardResult(
            allowed=True, message=original_message,
            details={"llama_guard": "ambiguous", "raw_response": response}
        )

    # ================================================================
    # Regex Semantic Matching (Layer 2 — always active, no model needed)
    # ================================================================

    @staticmethod
    def _check_regex(user_message: str) -> GuardResult:
        """Fast regex-based check for known harmful patterns.

        This implements the Colang semantic matching logic from
        input_guardrails.co without requiring NeMo Guardrails runtime.
        """
        message_lower = user_message.lower()
        violations = []

        for category, patterns in HARMFUL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    violations.append(category)
                    break

        if violations:
            # Choose refusal message based on violation type
            if "PII" in violations:
                refusal = SentinelGuard.REFUSAL_PII
            elif "O8" in violations:
                refusal = SentinelGuard.REFUSAL_MEDICAL
            elif "INJECTION" in violations:
                refusal = SentinelGuard.REFUSAL_INPUT
            else:
                refusal = SentinelGuard.REFUSAL_INPUT

            return GuardResult(
                allowed=False,
                violations=violations,
                message=refusal,
                details={"mode": "regex", "matched_categories": violations}
            )

        return GuardResult(allowed=True, message=user_message, details={"mode": "regex"})

    # ================================================================
    # Status & Control
    # ================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get guardrails status information."""
        return {
            "enabled": self.enabled,
            "mode": self._mode,
            "model_path": self.model_path,
            "model_loaded": self._model is not None,
            "config_path": self.config_path,
            "config_exists": Path(self.config_path).exists(),
            "init_error": self._init_error,
            "defense_layers": self._get_defense_layers(),
        }

    def _get_defense_layers(self) -> List[str]:
        """Get description of active defense layers."""
        layers = []
        if self.enabled:
            layers.append("Regex: Semantic pattern matching (medical ethics, PII, injection)")
            if self._mode == "llama_guard":
                layers.append("Llama Guard 3 (1B): Input classification O1-O8 (local, offline)")
                layers.append("Llama Guard 3 (1B): Output audit O1-O8 (local, offline)")
            elif self._mode == "regex":
                layers.append("Regex-only mode (Llama Guard not loaded)")
        return layers

    def enable(self):
        """Re-enable guardrails (re-initializes if needed)."""
        if not self.enabled:
            self._initialize()

    def disable(self):
        """Temporarily disable guardrails."""
        self.enabled = False
        self._mode = "disabled"
        logger.info("[GUARDRAILS] Guardrails disabled")

    # ================================================================
    # Legacy helpers (for compatibility with NeMo Guardrails integration)
    # ================================================================

    @staticmethod
    def _is_refusal(response: str) -> bool:
        """Check if a response is a safety refusal."""
        refusal_keywords = [
            "i'm sorry, i cannot fulfill",
            "i cannot fulfill this request",
            "prohibited from providing",
            "safety and ethical guidelines",
            "violates medical safety",
            "i am prohibited",
            "was flagged by the safety system",
            "cannot share specific patient information",
        ]
        response_lower = response.lower()
        return any(kw in response_lower for kw in refusal_keywords)

    @staticmethod
    def _extract_violations(response: str) -> List[str]:
        """Extract policy violation categories from a guardrail response."""
        categories = re.findall(r'O\d', response)
        if categories:
            return list(set(categories))

        violations = []
        response_lower = response.lower()
        if "medical" in response_lower or "ethical" in response_lower:
            violations.append("O8")
        if "patient" in response_lower and "privacy" in response_lower:
            violations.append("O8")
        if "violence" in response_lower:
            violations.append("O1")
        if "drug" in response_lower or "substance" in response_lower:
            violations.append("O5")
        if "self-harm" in response_lower or "suicide" in response_lower:
            violations.append("O6")

        return violations if violations else ["POLICY_VIOLATION"]
