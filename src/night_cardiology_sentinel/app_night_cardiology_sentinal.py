import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


TIME_HR_LINE = re.compile(
    r"Time:\s*(?P<time>[^-]+?)\s*-\s*heart rate\s*\[#/min\]\s*:\s*(?P<hr>[\d.]+)",
    re.IGNORECASE,
)
TIME_KV_LINE = re.compile(r"Time:\s*(?P<time>[^-]+?)\s*-\s*(?P<kv>.+)")


@dataclass
class SubjectInfo:
    subject_code: Optional[int]
    gender: Optional[str]
    length_cm: Optional[float]
    weight_kg: Optional[float]
    age_years: Optional[int]

    def as_prompt_block(self) -> str:
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
    time_raw: str
    time_seconds: Optional[float]
    values: Dict[str, float]
    raw_line: str


def parse_time_to_seconds(raw: str) -> Optional[float]:
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
            minutes, seconds = numbers
            return minutes * 60 + seconds
        if len(numbers) == 3:
            hours, minutes, seconds = numbers
            return hours * 3600 + minutes * 60 + seconds
    return None


def normalize_key(key: str) -> str:
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


def build_prompt(subject: SubjectInfo, window_summary: str) -> str:
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


@st.cache_resource(show_spinner=False)
def load_llama(model_path: str, n_ctx: int) -> "Llama":
    if Llama is None:
        raise RuntimeError("llama_cpp is not installed in this environment.")
    return Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)


def resolve_model_path(
    model_source: str,
    local_path: str,
    hf_repo: str,
    hf_filename: str,
) -> str:
    if model_source == "Local file path":
        return local_path
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not installed for HF downloads.")
    return hf_hub_download(repo_id=hf_repo, filename=hf_filename)


def render_subject_selector() -> SubjectInfo:
    st.subheader("Subject selection")
    mode = st.radio(
        "How do you want to provide subject info?",
        ["Upload subjects_info.json", "Enter manually"],
        horizontal=True,
    )
    if mode == "Upload subjects_info.json":
        uploaded = st.file_uploader("Upload JSON", type=["json"])
        if uploaded is None:
            st.info("Upload a JSON file to continue.")
            return SubjectInfo(None, None, None, None, None)
        data = json.loads(uploaded.read().decode("utf-8"))
        if not isinstance(data, list):
            st.error("Expected a JSON array of subjects.")
            return SubjectInfo(None, None, None, None, None)
        options = {str(item.get("subject_code")): item for item in data}
        selected_code = st.selectbox("Choose a subject", sorted(options.keys()))
        selected = options[selected_code]
        return SubjectInfo(
            subject_code=selected.get("subject_code"),
            gender=selected.get("gender"),
            length_cm=selected.get("length_cm"),
            weight_kg=selected.get("weight_kg"),
            age_years=selected.get("age_years"),
        )

    col1, col2 = st.columns(2)
    with col1:
        subject_code = st.number_input("Subject code", min_value=0, value=0, step=1)
        gender = st.selectbox("Gender", ["", "F", "M"])
        age_years = st.number_input("Age (years)", min_value=0, value=0, step=1)
    with col2:
        length_cm = st.number_input("Height (cm)", min_value=0.0, value=0.0, step=0.1)
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=0.0, step=0.1)

    return SubjectInfo(
        subject_code=int(subject_code) if subject_code else None,
        gender=gender or None,
        length_cm=float(length_cm) if length_cm else None,
        weight_kg=float(weight_kg) if weight_kg else None,
        age_years=int(age_years) if age_years else None,
    )


def render_model_selector() -> Tuple[str, str, str, int, int]:
    st.subheader("Model configuration")
    model_source = st.radio(
        "Model source", ["Local file path", "Hugging Face"], horizontal=True
    )
    local_path = st.text_input(
        "Local model path",
        value=str(Path("models/medgemma-night-sentinel-Q4_K_M.gguf")),
    )
    hf_repo = st.text_input(
        "HF repo id",
        value="Ismailea04/medgemma-night-sentinel",
    )
    hf_filename = st.text_input(
        "HF filename",
        value="medgemma-night-sentinel-Q4_K_M.gguf",
    )
    n_ctx = st.number_input("Context length", min_value=256, max_value=8192, value=2048)
    max_tokens = st.number_input(
        "Max tokens per window", min_value=64, max_value=1024, value=256
    )

    return model_source, local_path, hf_repo, hf_filename, int(n_ctx), int(max_tokens)


def main() -> None:
    st.set_page_config(page_title="Night Cardiology Sentinel", layout="wide")
    st.title("Night Cardiology Sentinel")
    st.caption("Upload subject info and vitals to generate windowed clinical insights.")

    subject = render_subject_selector()

    st.subheader("Vitals upload")
    vitals_file = st.file_uploader("Upload vitals text file", type=["txt"])
    window_mode = st.selectbox(
        "Windowing mode",
        ["15-minute windows", "10-row windows"],
    )

    model_path = ""
    n_ctx = 2048
    max_tokens = 256
    model_source = "Local file path"
    local_path = ""
    hf_repo = ""
    hf_filename = ""
    if Llama is None:
        st.warning("llama_cpp is not available. Install it to run inference.")
    else:
        (
            model_source,
            local_path,
            hf_repo,
            hf_filename,
            n_ctx,
            max_tokens,
        ) = render_model_selector()

    if st.button("Run analysis"):
        if vitals_file is None:
            st.error("Please upload a vitals text file.")
            return
        try:
            model_path = resolve_model_path(
                model_source, local_path, hf_repo, hf_filename
            )
        except Exception as exc:
            st.error(f"Failed to resolve model path: {exc}")
            return

        lines = vitals_file.read().decode("utf-8", errors="ignore").splitlines()
        rows = parse_vitals_lines(lines)
        if not rows:
            st.error("No vitals rows parsed. Check the file format.")
            return

        windows = chunk_rows(rows, window_mode)
        st.write(f"Parsed {len(rows)} rows into {len(windows)} windows.")

        try:
            llm = load_llama(model_path, n_ctx)
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            return

        progress = st.progress(0.0)
        results = []
        for idx, window_rows in enumerate(windows, start=1):
            window_summary = summarize_window(window_rows)
            prompt = build_prompt(subject, window_summary)
            response = llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<end_of_turn>"],
                echo=False,
            )
            text = response["choices"][0]["text"].strip()
            results.append((idx, window_summary, text))
            progress.progress(idx / len(windows))

        for idx, window_summary, text in results:
            st.markdown(f"### Window {idx}")
            st.code(window_summary, language="text")
            st.markdown(text)


if __name__ == "__main__":
    main()
