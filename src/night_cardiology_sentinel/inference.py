"""Model inference utilities for Night Sentinel."""

from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

from .data_parser import SubjectInfo


def build_prompt(subject: SubjectInfo, window_summary: str) -> str:
    """Build a structured prompt for the Night Sentinel model."""
    return (
        "<start_of_turn>user\n"
        "[NIGHT SENTINEL SYSTEM]\n"
        f"Patient Profile:\n{subject.as_prompt_block()}\n"
        "Event/Anomaly (window summary):\n"
        f"{window_summary}\n"
        "    TASK:\n"
        "1. Compare the current data to the patient's specific baseline (Name it : ##Comparaison).\n"
        "2. Identify any clinical anomalies (Name it : ##Detection).\n"
        "3. Provide a short clinical interpretation of what might be happening (Name it : ##Interpretation)."
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


class SentinelInference:
    """Wrapper for loading and running inference with the Night Sentinel model."""

    def __init__(self, model_path: str, n_ctx: int = 2048):
        """Initialize the inference engine with a GGUF model."""
        if Llama is None:
            raise RuntimeError("llama_cpp is not installed in this environment.")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)

    def predict(
        self,
        prompt: str,
        max_tokens: int = 256,
        stop: Optional[list] = None,
    ) -> str:
        """Run inference on a prompt and return the generated text."""
        if stop is None:
            stop = ["<end_of_turn>"]
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            echo=False,
        )
        return response["choices"][0]["text"].strip()

    @staticmethod
    def resolve_model_path(
        model_source: str,
        local_path: str,
        hf_repo: str,
        hf_filename: str,
    ) -> str:
        """Resolve the model path from local file or Hugging Face."""
        if model_source == "Local file path":
            return local_path
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is not installed for HF downloads.")
        return hf_hub_download(repo_id=hf_repo, filename=hf_filename)
