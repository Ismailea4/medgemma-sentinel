"""
Cardiology-focused scenario that will trigger the Night Cardiology Sentinel tool
"""

import os
import sys
import re
import json
import asyncio
from dotenv import load_dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

load_dotenv()

class CardiologyScenarioClient:
    def __init__(self, model_name="amsaravi/medgemma-4b-it:q6"):
        self.model_name = model_name
        self.client = ollama.AsyncClient()
        
        self.system_prompt_template = """You are an autonomous Night Watch Medical AI specialized in cardiology.
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

NEVER make up the Observation yourself. When you output 'Action Input:', STOP writing immediately.
"""

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

    async def run(self, user_request: str, max_iterations=8):
        print("\n[System] Starting Cardiology-Focused MCP Client...\n")
        
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["medical_mcp_server.py"]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools_response = await session.list_tools()
                available_tools = {t.name: t for t in tools_response.tools}
                
                tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in available_tools.values()])
                system_prompt = self.system_prompt_template.format(tool_descriptions=tool_descriptions)

                print(f"==============================================")
                print(f"🫀 CARDIOLOGY NIGHT WATCH AGENT")
                print(f"🛠️  Discovered {len(available_tools)} tools")
                print(f"🚨 SCENARIO: {user_request}")
                print(f"==============================================\n")
                
                prompt_scratchpad = f"Question: {user_request}\n"
                cardiology_analysis_complete = False
                
                for i in range(max_iterations):
                    print(f"\n[Loop {i+1}] MedGemma is thinking...")
                    llm_output = await self._query_llm(prompt_scratchpad, system_prompt)
                    
                    print(llm_output.strip())
                    prompt_scratchpad += llm_output + "\n"

                    if "Final Answer:" in llm_output:
                        print("\n✅ CARDIOLOGY ASSESSMENT COMPLETE.")
                        return
                    
                    action_match = re.search(r"Action:\s*(.*?)\n", llm_output)
                    input_match = re.search(r"Action Input:\s*(.+?)(?:\n\s*\n|\Z)", llm_output, re.DOTALL)

                    if action_match and input_match:
                        action_name = action_match.group(1).strip()
                        raw_input = input_match.group(1).strip()

                        if action_name in available_tools:
                            print(f"  --> [MCP] Calling '{action_name}' (Please wait...)")
                            try:
                                # Clean JSON: remove markdown code blocks
                                raw_input = raw_input.replace('```json', '').replace('```', '').strip()
                                
                                # Extract JSON if it's wrapped in code blocks
                                if raw_input.startswith('{'):
                                    # Find the last closing brace
                                    last_brace = raw_input.rfind('}')
                                    if last_brace != -1:
                                        raw_input = raw_input[:last_brace + 1]
                                
                                action_args = json.loads(raw_input)
                                
                                tool_result = await session.call_tool(action_name, arguments=action_args)
                                observation = tool_result.content[0].text if tool_result.content else "Success, but no output."
                                
                                # Check if this was the cardiology analysis tool
                                if action_name == "analyze_cardiology_sentinel":
                                    cardiology_analysis_complete = True
                                
                            except json.JSONDecodeError as je:
                                observation = f"Error: Invalid JSON. Details: {str(je)}"
                            except Exception as e:
                                observation = f"Error calling MCP tool: {str(e)}"
                        else:
                            observation = f"Error: Tool '{action_name}' is not registered on the MCP Server."

                        print(f"👀 Observation: {observation}")
                        prompt_scratchpad += f"Observation: {observation}\n"
                        
                        # After cardiology analysis, generate actionable instructions and stop
                        if cardiology_analysis_complete:
                            print(f"\n[Loop {i+2}] Generating actionable instructions for night nurse...\n")
                            final_instructions = await self._generate_instructions(observation)
                            print("\n" + "="*60)
                            print("🏥 ACTIONABLE INSTRUCTIONS FOR NIGHT NURSE:")
                            print("="*60)
                            print(final_instructions.strip())
                            print("="*60)
                            print("\n✅ SESSION COMPLETE.")
                            return
                    else:
                        print(f"⚠️ Formatting error. Nudging agent back on track...")
                        prompt_scratchpad += "Observation: Format error. You must provide an 'Action' and 'Action Input'.\n"
                        
                print("\n❌ Task aborted: Reached maximum iterations without a Final Answer.")

if __name__ == "__main__":
    # Cardiology-specific scenario  
    scenario = """Patient 402's ECG monitor is showing tachycardia with progressive worsening.

I have the vital signs from the monitoring system in text format:
Time: 00:00 - HR: 115, SPO2: 88, RESP: 24
Time: 00:01 - HR: 118, SPO2: 87, RESP: 25
Time: 00:02 - HR: 120, SPO2: 86, RESP: 26
Time: 00:03 - HR: 122, SPO2: 85, RESP: 27
Time: 00:04 - HR: 125, SPO2: 84, RESP: 28

Use the analyze_cardiology_sentinel tool with patient_id "402" and pass the vitals text (all 5 lines) as the vitals_text parameter. Then provide clinical recommendations."""
    
    agent = CardiologyScenarioClient(model_name="amsaravi/medgemma-4b-it:q6")
    asyncio.run(agent.run(scenario))
