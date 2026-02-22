"""Night Cardiology Sentinel Streamlit App.

This app allows users to:
- Select a patient by uploading subjects_info.json or entering details manually
- Upload vitals data from a text file
- Process vitals in time-based or row-based windows
- Generate clinical insights using a quantized MedGemma model (local or HF)
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Add src to path so we can import night_cardiology_sentinel
sys.path.insert(0, str(Path(__file__).parent / "src"))

from night_cardiology_sentinel import (
    SentinelInference,
    SubjectInfo,
    build_prompt,
    chunk_rows,
    parse_vitals_lines,
    summarize_window,
)


def render_subject_selector() -> SubjectInfo:
    """Render UI for selecting or entering subject information."""
    st.subheader("📋 Subject selection")
    mode = st.radio(
        "How do you want to provide subject info?",
        ["Upload subjects_info.json", "Enter manually"],
        horizontal=True,
    )
    
    if mode == "Upload subjects_info.json":
        uploaded = st.file_uploader("Upload JSON", type=["json"], key="subject_json")
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
        
        subject = SubjectInfo(
            subject_code=selected.get("subject_code"),
            gender=selected.get("gender"),
            length_cm=selected.get("length_cm"),
            weight_kg=selected.get("weight_kg"),
            age_years=selected.get("age_years"),
        )
        
        # Display selected subject info
        st.success("✅ Subject selected")
        st.code(subject.as_prompt_block(), language="text")
        
        return subject

    # Manual entry
    col1, col2 = st.columns(2)
    with col1:
        subject_code = st.number_input("Subject code", min_value=0, value=0, step=1)
        gender = st.selectbox("Gender", ["", "F", "M"])
        age_years = st.number_input("Age (years)", min_value=0, value=0, step=1)
    with col2:
        length_cm = st.number_input("Height (cm)", min_value=0.0, value=0.0, step=0.1)
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=0.0, step=0.1)

    subject = SubjectInfo(
        subject_code=int(subject_code) if subject_code else None,
        gender=gender or None,
        length_cm=float(length_cm) if length_cm else None,
        weight_kg=float(weight_kg) if weight_kg else None,
        age_years=int(age_years) if age_years else None,
    )
    
    # Display entered subject info if any field is filled
    if any([subject_code, gender, age_years, length_cm, weight_kg]):
        st.info("Current subject info:")
        st.code(subject.as_prompt_block(), language="text")
    
    return subject


def render_model_selector() -> tuple:
    """Render UI for model configuration."""
    st.subheader("🧠 Model configuration")
    
    model_source = st.radio(
        "Model source",
        ["Local file path", "Hugging Face"],
        horizontal=True,
        key="model_source",
    )
    
    if model_source == "Local file path":
        local_path = st.text_input(
            "Local model path",
            value="models/medgemma-night-sentinel-Q4_K_M.gguf",
            key="local_path",
        )
        hf_repo = ""
        hf_filename = ""
    else:
        hf_repo = st.text_input(
            "HF repo id",
            value="Ismailea04/medgemma-night-sentinel",
            key="hf_repo",
        )
        hf_filename = st.text_input(
            "HF filename",
            value="medgemma-night-sentinel-Q4_K_M.gguf",
            key="hf_filename",
        )
        local_path = ""
    
    col1, col2 = st.columns(2)
    with col1:
        n_ctx = st.number_input(
            "Context length", min_value=256, max_value=8192, value=2048, step=256
        )
    with col2:
        max_tokens = st.number_input(
            "Max tokens per window", min_value=64, max_value=1024, value=256, step=64
        )

    return model_source, local_path, hf_repo, hf_filename, int(n_ctx), int(max_tokens)


def main() -> None:
    """Main app entry point."""
    st.set_page_config(
        page_title="Night Cardiology Sentinel",
        page_icon="🏥",
        layout="wide",
    )
    
    st.title("🏥 Night Cardiology Sentinel")
    st.caption(
        "Upload subject info and vitals to generate windowed clinical insights "
        "using the MedGemma Night Sentinel model."
    )

    # Subject selection
    subject = render_subject_selector()

    # Vitals upload
    st.subheader("💓 Vitals upload")
    vitals_file = st.file_uploader(
        "Upload vitals text file", type=["txt"], key="vitals_file"
    )
    window_mode = st.selectbox(
        "Windowing mode",
        ["15-minute windows", "10-row windows"],
        key="window_mode",
    )

    # Model configuration
    (
        model_source,
        local_path,
        hf_repo,
        hf_filename,
        n_ctx,
        max_tokens,
    ) = render_model_selector()

    # Run analysis button
    if st.button("🚀 Run analysis", type="primary"):
        if vitals_file is None:
            st.error("Please upload a vitals text file.")
            return

        # Parse vitals
        with st.spinner("Parsing vitals data..."):
            lines = vitals_file.read().decode("utf-8", errors="ignore").splitlines()
            rows = parse_vitals_lines(lines)
        
        if not rows:
            st.error("No vitals rows parsed. Check the file format.")
            return

        windows = chunk_rows(rows, window_mode)
        st.success(f"✅ Parsed {len(rows)} rows into {len(windows)} windows.")

        # Load model
        try:
            with st.spinner("Loading model..."):
                model_path = SentinelInference.resolve_model_path(
                    model_source, local_path, hf_repo, hf_filename
                )
                sentinel = SentinelInference(model_path, n_ctx)
            st.success(f"✅ Model loaded: {Path(model_path).name}")
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            return

        # Run inference on each window
        st.subheader("🔍 Analysis Results")
        progress_bar = st.progress(0.0, text="Analyzing windows...")
        
        for idx, window_rows in enumerate(windows, start=1):
            window_summary = summarize_window(window_rows)
            prompt = build_prompt(subject, window_summary)
            
            with st.spinner(f"Analyzing window {idx}/{len(windows)}..."):
                response_text = sentinel.predict(prompt, max_tokens=max_tokens)
            
            # Display results
            with st.expander(f"**Window {idx}** - {len(window_rows)} rows", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**Window Summary:**")
                    st.code(window_summary, language="text")
                with col2:
                    st.markdown("**Clinical Analysis:**")
                    st.markdown(response_text)
            
            progress_bar.progress(idx / len(windows), text=f"Completed {idx}/{len(windows)} windows")
        
        st.success("✅ Analysis complete!")


if __name__ == "__main__":
    main()
