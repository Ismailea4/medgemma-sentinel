"""
Combined Streamlit App: Night Cardiology Sentinel + MCP Client
Tabs for both the standalone analyzer and the MCP ReAct loop visualization
"""

import json
import sys
import os
import re
import asyncio
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Add src directory to path - updated for root location
src_path = Path(__file__).parent / "mcp_architecture" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports from cardiology_sentinel module
from cardiology_sentinel import (
    SentinelInference,
    SubjectInfo,
    build_prompt,
    chunk_rows,
    parse_vitals_lines,
    summarize_window,
)

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

load_dotenv(dotenv_path=src_path / ".env")

# Page configuration
st.set_page_config(
    page_title="Night Cardiology Sentinel + MCP",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.tool-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.thought-box {
    background-color: #e3f2fd;
    border-left: 4px solid #2196F3;
    padding: 10px;
    margin: 10px 0;
}
.action-box {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 10px;
    margin: 10px 0;
}
.observation-box {
    background-color: #e8f5e9;
    border-left: 4px solid #4caf50;
    padding: 10px;
    margin: 10px 0;
}
.cardiology-result {
    background-color: #fce4ec;
    border-left: 4px solid #e91e63;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ==================== NIGHT CARDIOLOGY SENTINEL TAB ====================

def render_subject_selector() -> SubjectInfo:
    """Render UI for selecting or entering subject information."""
    st.subheader("📋 Subject Selection")
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
        
        st.success("✅ Subject selected")
        st.code(subject.as_prompt_block(), language="text")
        return subject

    # Manual entry
    col1, col2 = st.columns(2)
    with col1:
        subject_code = st.number_input("Subject code", min_value=0, value=402, step=1)
        gender = st.selectbox("Gender", ["", "F", "M"])
        age_years = st.number_input("Age (years)", min_value=0, value=45, step=1)
    with col2:
        length_cm = st.number_input("Height (cm)", min_value=0.0, value=175.0, step=0.1)
        weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=80.0, step=0.1)

    subject = SubjectInfo(
        subject_code=int(subject_code) if subject_code else None,
        gender=gender or None,
        length_cm=float(length_cm) if length_cm else None,
        weight_kg=float(weight_kg) if weight_kg else None,
        age_years=int(age_years) if age_years else None,
    )
    
    if any([subject_code, gender, age_years, length_cm, weight_kg]):
        st.info("Current subject info:")
        st.code(subject.as_prompt_block(), language="text")
    
    return subject


def render_model_selector() -> tuple:
    """Render UI for model configuration."""
    st.subheader("🧠 Model Configuration")
    
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


def run_cardiology_analyzer():
    """Run the standalone Night Cardiology Sentinel analyzer."""
    st.title("🏥 Night Cardiology Sentinel - Standalone Analyzer")
    st.caption(
        "Upload subject info and vitals to generate windowed clinical insights "
        "using the MedGemma Night Sentinel model."
    )

    # Subject selection
    subject = render_subject_selector()

    # Vitals upload
    st.subheader("💓 Vitals Upload")
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
    if st.button("🚀 Run Analysis", type="primary", key="run_cardio"):
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


# ==================== MCP CLIENT TAB ====================

class MCPVisualizerClient:
    def __init__(self, model_name="amsaravi/medgemma-4b-it:q6"):
        self.model_name = model_name
        self.client = ollama.AsyncClient()
        self.conversation_history = []

    async def _query_llm(self, prompt: str, system_prompt: str) -> str:
        response = await self.client.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            options={
                "temperature": 0.1,
                "stop": ["Observation:"]
            }
        )
        return response['message']['content']

    async def _generate_instructions(self, analysis: str) -> str:
        """Generate actionable instructions based on cardiology analysis"""
        instructions_system = """You are a Night Watch Medical AI specialized in cardiology. 
Based on the cardiac analysis provided, generate clear, actionable instructions for the night nurse.
Be specific about interventions, monitoring frequency, and escalation criteria."""
        
        instructions_prompt = f"""Based on this cardiology sentinel analysis:

{analysis}

Provide actionable instructions for the night nurse including:
1. Immediate interventions needed
2. Monitoring frequency and parameters
3. Criteria for physician escalation
4. Vital sign targets to achieve

Be concise and clinical."""
        
        response = await self.client.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': instructions_system},
                {'role': 'user', 'content': instructions_prompt}
            ],
            options={
                "temperature": 0.1,
                "stop": []
            }
        )
        return response['message']['content']

    async def run_with_visualization(self, user_request: str, max_iterations=8):
        """Run ReAct loop with step-by-step visualization"""
        # Updated path for root location
        mcp_server_path = str(Path(__file__).parent / "mcp_architecture" / "src" / "medical_mcp_server.py")
        
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[mcp_server_path]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools_response = await session.list_tools()
                available_tools = {t.name: t for t in tools_response.tools}
                
                tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in available_tools.values()])
                
                system_prompt_template = """You are an autonomous Night Watch Medical AI specialized in cardiology.
You have access to the following tools via an MCP Server:
{tool_descriptions}

IMPORTANT: When dealing with heart-related conditions (tachycardia, arrhythmia, cardiac distress), 
ALWAYS use the 'analyze_cardiology_sentinel' tool with the patient's vital signs data.

You MUST solve the user's request by following this EXACT format:
Question: the input goal you must achieve
Thought: you should always think about what to do next
Action: the exact name of the tool to use
Action Input: a valid JSON object containing the arguments
Observation: the result of the action (provided by the system)
... (this Thought/Action/Action Input/Observation cycle can repeat N times)
Thought: I have enough information to form a clinical conclusion.
Final Answer: your final triage assessment and actionable instructions for the night nurse.

NEVER make up the Observation yourself. When you output 'Action Input:', STOP writing immediately."""
                
                system_prompt = system_prompt_template.format(tool_descriptions=tool_descriptions)
                
                prompt_scratchpad = f"Question: {user_request}\n"
                cardiology_analysis_complete = False
                analysis_text = ""
                
                steps = []
                
                for i in range(max_iterations):
                    llm_output = await self._query_llm(prompt_scratchpad, system_prompt)
                    prompt_scratchpad += llm_output + "\n"

                    if "Final Answer:" in llm_output:
                        steps.append({
                            "type": "final_answer",
                            "content": llm_output.split("Final Answer:")[-1].strip()
                        })
                        return steps

                    action_match = re.search(r"Action:\s*(.*?)\n", llm_output)
                    input_match = re.search(r"Action Input:\s*(.+?)(?:\n\s*\n|\Z)", llm_output, re.DOTALL)

                    if action_match and input_match:
                        action_name = action_match.group(1).strip()
                        raw_input = input_match.group(1).strip()

                        steps.append({
                            "type": "thought_action",
                            "thought": re.search(r"Thought:\s*(.+?)(?:\nAction:|$)", llm_output, re.DOTALL),
                            "action": action_name,
                            "action_input": raw_input
                        })

                        if action_name in available_tools:
                            try:
                                raw_input = raw_input.replace('```json', '').replace('```', '').strip()
                                
                                if raw_input.startswith('{'):
                                    last_brace = raw_input.rfind('}')
                                    if last_brace != -1:
                                        raw_input = raw_input[:last_brace + 1]
                                
                                action_args = json.loads(raw_input)
                                
                                tool_result = await session.call_tool(action_name, arguments=action_args)
                                observation = tool_result.content[0].text if tool_result.content else "Success, but no output."
                                
                                if action_name == "analyze_cardiology_sentinel":
                                    cardiology_analysis_complete = True
                                    analysis_text = observation
                                
                            except json.JSONDecodeError as je:
                                observation = f"Error: Invalid JSON. Details: {str(je)}"
                            except Exception as e:
                                observation = f"Error calling MCP tool: {str(e)}"
                        else:
                            observation = f"Error: Tool '{action_name}' is not registered on the MCP Server."

                        steps.append({
                            "type": "observation",
                            "content": observation
                        })
                        prompt_scratchpad += f"Observation: {observation}\n"
                        
                        if cardiology_analysis_complete:
                            instructions = await self._generate_instructions(observation)
                            steps.append({
                                "type": "instructions",
                                "content": instructions
                            })
                            return steps
                    else:
                        steps.append({
                            "type": "error",
                            "content": "Format error. Nudging agent back on track..."
                        })
                        prompt_scratchpad += "Observation: Format error. You must provide an 'Action' and 'Action Input'.\n"

                return steps


def render_conversation_step(step: Dict):
    """Render a single step from the ReAct loop."""
    if step["type"] == "thought_action":
        thought_text = step["thought"].group(1).strip() if step["thought"] else "..."
        st.markdown(f'<div class="thought-box"><b>💭 Thought:</b> {thought_text}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="action-box"><b>🎯 Action:</b> {step["action"]}<br><b>Input:</b> <code>{step["action_input"][:100]}...</code></div>', unsafe_allow_html=True)
    
    elif step["type"] == "observation":
        if "cardiology" in step["content"].lower() or "##" in step["content"]:
            st.markdown(f'<div class="cardiology-result">{step["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="observation-box">{step["content"]}</div>', unsafe_allow_html=True)
    
    elif step["type"] == "final_answer":
        st.markdown(f'<div class="observation-box"><b>✅ Final Assessment:</b> {step["content"]}</div>', unsafe_allow_html=True)
    
    elif step["type"] == "instructions":
        st.markdown(f'<div class="cardiology-result"><b>🏥 Actionable Instructions:</b><br>{step["content"]}</div>', unsafe_allow_html=True)
    
    elif step["type"] == "error":
        st.markdown(f'<div class="observation-box"><b>⚠️ {step["content"]}</b></div>', unsafe_allow_html=True)


def run_mcp_client():
    """Run the MCP Client with ReAct visualization."""
    st.title("🫀 Night Watch MCP Agent with ReAct Loop")
    st.caption("Interact with the MCP architecture and use the cardiology sentinel tool in real-time.")
    
    tab1, tab2, tab3 = st.tabs(["Run Scenario", "Custom Query", "Documentation"])
    
    with tab1:
        st.subheader("Pre-built Scenarios")
        
        scenarios = {
            "Cardiac Emergency - Patient 402": """Patient 402's ECG monitor is showing tachycardia with progressive worsening.

I have the vital signs from the monitoring system in text format:
Time: 00:00 - HR: 115, SPO2: 88, RESP: 24
Time: 00:01 - HR: 118, SPO2: 87, RESP: 25
Time: 00:02 - HR: 120, SPO2: 86, RESP: 26
Time: 00:03 - HR: 122, SPO2: 85, RESP: 27
Time: 00:04 - HR: 125, SPO2: 84, RESP: 28

Use the analyze_cardiology_sentinel tool with patient_id "402" and pass the vitals text (all 5 lines) as the vitals_text parameter. Then provide clinical recommendations.""",
            
            "Respiratory Distress": """Patient 301 is experiencing increased respiratory distress. 
Their vitals show:
Time: 10:00 - HR: 98, SPO2: 92, RESP: 22
Time: 10:05 - HR: 105, SPO2: 90, RESP: 26
Time: 10:10 - HR: 112, SPO2: 88, RESP: 28

Analyze this patient's cardiorespiratory status and recommend interventions.""",
            
            "Motion Detected": "Motion sensors are detecting unusual activity in the cardiac monitoring station. Review the current patient data and status.",
        }
        
        selected_scenario = st.selectbox("Choose a scenario:", list(scenarios.keys()))
        scenario_text = scenarios[selected_scenario]
        
        if st.button("▶️ Run Scenario", type="primary", key="run_scenario"):
            st.info("🔄 Running MCP agent with ReAct loop visualization...")
            
            client = MCPVisualizerClient()
            steps = asyncio.run(client.run_with_visualization(scenario_text, max_iterations=8))
            
            st.subheader("Agent Reasoning Process")
            for idx, step in enumerate(steps, 1):
                if step["type"] != "observation":
                    st.markdown(f"### Step {idx}")
                render_conversation_step(step)
                st.divider()

    with tab2:
        st.subheader("Custom Query")
        custom_query = st.text_area(
            "Enter your medical query or scenario:",
            placeholder="Describe the patient scenario and what you need the agent to do...",
            height=150
        )
        
        if st.button("▶️ Run Custom Query", type="primary", key="run_custom"):
            if not custom_query:
                st.error("Please enter a query.")
                return
            
            st.info("🔄 Running MCP agent with ReAct loop visualization...")
            
            client = MCPVisualizerClient()
            steps = asyncio.run(client.run_with_visualization(custom_query, max_iterations=8))
            
            st.subheader("Agent Reasoning Process")
            for idx, step in enumerate(steps, 1):
                if step["type"] != "observation":
                    st.markdown(f"### Step {idx}")
                render_conversation_step(step)
                st.divider()

    with tab3:
        st.subheader("📚 MCP Architecture Documentation")
        st.markdown("""
### Available Tools

The MCP Server exposes the following tools to the agent:

1. **analyze_cardiology_sentinel** - Analyzes patient vital signs using the Night Cardiology Sentinel model
2. **get_patient_vitals** - Retrieves current vital signs for a specific patient
3. **get_patient_history** - Retrieves medical history for a patient
4. **get_current_time** - Gets the current time for context
5. **analyze_room_audio** - Analyzes audio data from the monitoring station

### ReAct Loop Visualization

Each interaction follows the ReAct (Reasoning + Acting) pattern:
- **Thought**: Agent reasons about the next step
- **Action**: Agent selects which tool to use
- **Action Input**: JSON parameters for the tool
- **Observation**: Result returned by the tool
- **Final Answer**: Clinical conclusion and recommendations

### Night Cardiology Sentinel

The Night Cardiology Sentinel model analyzes vital signs and provides:
- **Comparison**: Assessment against clinical reference ranges
- **Detection**: Identification of cardiac/respiratory anomalies
- **Interpretation**: Clinical assessment and recommendations
        """)


def main():
    st.title("🏥 Night Cardiology Sentinel + MCP Agent")
    st.caption("Combined platform for cardiology analysis and MCP-based clinical decision support")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Standalone Analyzer", "🫀 MCP Agent with ReAct"])
    
    with tab1:
        run_cardiology_analyzer()
    
    with tab2:
        run_mcp_client()


if __name__ == "__main__":
    main()
