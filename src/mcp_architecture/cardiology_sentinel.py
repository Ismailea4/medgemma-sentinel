"""
Night Cardiology Sentinel - Consolidated Module
Data parsing and model inference for cardiac anomaly detection
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# ============================================================================
# DATA PARSER - Vitals parsing and windowing
# ============================================================================

TIME_HR_LINE = re.compile(
    r"Time:\s*(?P<time>[^-]+?)\s*-\s*heart rate\s*\[#/min\]\s*:\s*(?P<hr>[\d.]+)",
    re.IGNORECASE,
)
TIME_KV_LINE = re.compile(r"Time:\s*(?P<time>[^-]+?)\s*-\s*(?P<kv>.+)")


@dataclass
class SubjectInfo:
    """Patient subject information."""

    subject_code: Optional[int]
    gender: Optional[str]
    length_cm: Optional[float]
    weight_kg: Optional[float]
    age_years: Optional[int]

    def as_prompt_block(self) -> str:
        """Format subject info as a text block for prompts."""
        parts = []
        if self.subject_code is not None:
            parts.append(f"Subject Code: {self.subject_code}")
        if self.gender:
            parts.append(f"Gender: {self.gender}")
        if self.age_years is not None:
            parts.append(f"Age: {self.age_years}")
        if self.length_cm is not None:
            parts.append(f"Height (cm): {self.length_cm}")
        if self.weight_kg is not None:
            parts.append(f"Weight (kg): {self.weight_kg}")
        return "\n".join(parts) if parts else "Unknown subject"


@dataclass
class VitalsRow:
    """A single parsed row of vitals data."""

    time_raw: str
    time_seconds: Optional[float]
    values: Dict[str, float]
    raw_line: str


def parse_time_to_seconds(raw: str) -> Optional[float]:
    """Convert time string to seconds."""
    raw = raw.strip()
    if not raw:
        return None
    if raw.endswith("s"):
        try:
            return float(raw[:-1])
        except ValueError:
            return None
    if ":" in raw:
        parts = raw.split(":")
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            return None
        if len(numbers) == 2:
            hours, minutes = numbers
            return hours * 3600 + minutes * 60
        if len(numbers) == 3:
            hours, minutes, seconds = numbers
            return hours * 3600 + minutes * 60 + seconds
    return None


def normalize_key(key: str) -> str:
    """Normalize vital sign keys."""
    key = key.strip()
    key = key.replace("%", "").replace(" ", "")
    key = key.upper()
    if key in {"HEARTRATE", "HR"}:
        return "HR"
    if key in {"SPO2", "SP02", "O2SAT", "SPO2PERCENT"}:
        return "SPO2"
    if key in {"RESP", "RR"}:
        return "RESP"
    if key in {"PULSE"}:
        return "PULSE"
    return key


def parse_vitals_lines(lines: Iterable[str]) -> List[VitalsRow]:
    """Parse vitals from text lines with two supported formats."""
    rows: List[VitalsRow] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        match_hr = TIME_HR_LINE.match(line)
        if match_hr:
            time_raw = match_hr.group("time").strip()
            hr_val = float(match_hr.group("hr"))
            rows.append(
                VitalsRow(
                    time_raw=time_raw,
                    time_seconds=parse_time_to_seconds(time_raw),
                    values={"HR": hr_val},
                    raw_line=line,
                )
            )
            continue
        match_kv = TIME_KV_LINE.match(line)
        if match_kv:
            time_raw = match_kv.group("time").strip()
            kv_blob = match_kv.group("kv")
            values: Dict[str, float] = {}
            for part in kv_blob.split(","):
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                key = normalize_key(key)
                try:
                    values[key] = float(value.strip())
                except ValueError:
                    continue
            if values:
                rows.append(
                    VitalsRow(
                        time_raw=time_raw,
                        time_seconds=parse_time_to_seconds(time_raw),
                        values=values,
                        raw_line=line,
                    )
                )
    return rows


def chunk_rows(
    rows: List[VitalsRow],
    window_mode: str,
    window_minutes: int = 15,
    window_rows: int = 10,
) -> List[List[VitalsRow]]:
    """Split vitals rows into windows."""
    if not rows:
        return []
    if window_mode == "15-minute windows":
        with_time = [row for row in rows if row.time_seconds is not None]
        if not with_time:
            return chunk_rows(rows, "10-row windows", window_rows=window_rows)
        with_time.sort(key=lambda r: r.time_seconds or 0)
        windows: Dict[int, List[VitalsRow]] = {}
        window_seconds = window_minutes * 60
        for row in with_time:
            bucket = int((row.time_seconds or 0) // window_seconds)
            windows.setdefault(bucket, []).append(row)
        return [windows[key] for key in sorted(windows.keys())]
    if window_mode == "10-row windows":
        return [rows[i : i + window_rows] for i in range(0, len(rows), window_rows)]
    return [rows]


def summarize_window(rows: List[VitalsRow]) -> str:
    """Generate summary statistics for a window of vitals."""
    if not rows:
        return "No vitals"
    metrics: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.values.items():
            metrics.setdefault(key, []).append(value)
    summary_lines = [f"Window rows: {len(rows)}"]
    times = [r.time_raw for r in rows if r.time_raw]
    if times:
        summary_lines.append(f"Time range: {times[0]} -> {times[-1]}")
    for key, values in metrics.items():
        if not values:
            continue
        avg = sum(values) / len(values)
        summary_lines.append(
            f"{key}: avg {avg:.2f}, min {min(values):.2f}, max {max(values):.2f}"
        )
    return "\n".join(summary_lines)


# ============================================================================
# INFERENCE - Model loading and prediction
# ============================================================================

def build_prompt(subject: SubjectInfo, window_summary: str) -> str:
    """Build a structured prompt for the Night Sentinel model with clinical context."""
    return (
        "<start_of_turn>user\n"
        "[NIGHT SENTINEL CARDIAC MONITORING SYSTEM]\n\n"
        f"Patient Profile:\n{subject.as_prompt_block()}\n\n"
        "VITAL SIGNS REFERENCE RANGES (Adult):\n"
        "- Heart Rate (HR): 60-100 bpm (Tachycardia: >100, Bradycardia: <60)\n"
        "- Oxygen Saturation (SpO2): >95% (Hypoxemia: <92%)\n"
        "- Respiratory Rate (RESP): 12-20 breaths/min (Tachypnea: >20, Bradypnea: <12)\n"
        "- Blood Pressure (BP): <120/80 normal\n\n"
        "Current Vital Signs Data:\n"
        f"{window_summary}\n\n"
        "ANALYSIS TASK:\n"
        "1. ##Comparaison: Assess current vitals against normal reference ranges. Flag any values outside normal limits.\n"
        "2. ##Detection: Identify specific cardiac/respiratory anomalies. Look for:\n"
        "   - Sustained tachycardia (HR >100)\n"
        "   - Hypoxemia (SpO2 <92%)\n"
        "   - Tachypnea (RR >20)\n"
        "   - Progressive worsening trends\n"
        "3. ##Interpretation: Clinical assessment of the severity and urgency. Provide specific recommendations.\n\n"
        "Be specific about abnormal findings and their clinical significance.\n"
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
