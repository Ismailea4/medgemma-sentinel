"""
Medical AI Agent (True MCP Client + ReAct Loop)
Connects to a FastMCP server, discovers tools dynamically, 
and executes a Reason+Act loop using Ollama (MedGemma).
"""

import os
import sys
import re
import json
import asyncio
from dotenv import load_dotenv

# CRITICAL: Suppress TensorFlow logs from corrupting the MCP stdio stream
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# MCP Client SDK Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

load_dotenv()

class MedicalReActMCPClient:
    def __init__(self, model_name="amsaravi/medgemma-4b-it:q6"):
        self.model_name = model_name
        self.client = ollama.AsyncClient()
        
        # The core ReAct prompt
        self.system_prompt_template = """You are an autonomous Night Watch Medical AI.
You have access to the following tools via an MCP Server:
{tool_descriptions}

You MUST solve the user's request by following this EXACT format:
Question: the input goal you must achieve
Thought: you should always think about what to do next
Action: the exact name of the tool to use
Action Input: a valid JSON object containing the arguments (e.g. {{"room_number": "402"}} or {{"patient_id": "402"}})
Observation: the result of the action (provided by the system)
... (this Thought/Action/Action Input/Observation cycle can repeat N times)
Thought: I have enough information to form a clinical conclusion.
Final Answer: your final triage assessment and actionable instructions for the night nurse.

NEVER make up the Observation yourself. When you output 'Action Input:', STOP writing immediately.
"""

    async def _query_llm(self, prompt: str, system_prompt: str) -> str:
        """Sends the scratchpad to Ollama asynchronously."""
        response = await self.client.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            options={
                "temperature": 0.1, 
                "stop": ["Observation:"] # Forces MedGemma to stop so we can run the tool
            } 
        )
        return response['message']['content']

    async def run(self, user_request: str, max_iterations=8):
        """Connects to the MCP server and executes the Reason+Act Loop."""
        
        print("\n[System] Booting up MCP Client & Starting FastMCP Server...")
        
        # Configure the connection to the local FastMCP Server
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["medical_mcp_server.py"]
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 1. Dynamically Discover Tools from the MCP Server
                tools_response = await session.list_tools()
                available_tools = {t.name: t for t in tools_response.tools}
                
                tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in available_tools.values()])
                system_prompt = self.system_prompt_template.format(tool_descriptions=tool_descriptions)

                print(f"==============================================")
                print(f"🤖 NIGHT WATCH AGENT (MCP ReAct Mode)")
                print(f"🛠️  Discovered {len(available_tools)} tools: {list(available_tools.keys())}")
                print(f"🚨 GOAL: {user_request}")
                print(f"==============================================\n")
                
                # The scratchpad keeps track of the conversation history
                prompt_scratchpad = f"Question: {user_request}\n"
                
                # 2. Start the ReAct Loop
                for i in range(max_iterations):
                    print(f"\n[Loop {i+1}] MedGemma is thinking...")
                    llm_output = await self._query_llm(prompt_scratchpad, system_prompt)
                    
                    print(llm_output.strip())
                    prompt_scratchpad += llm_output + "\n"

                    # Check if the LLM has reached a final conclusion
                    if "Final Answer:" in llm_output:
                        print("\n✅ TASK COMPLETE.")
                        return
                    
                    # Parse the Tool Request using Regex
                    action_match = re.search(r"Action:\s*(.*?)\n", llm_output)
                    input_match = re.search(r"Action Input:\s*(.+?)(?:\n\s*\n|\Z)", llm_output, re.DOTALL)

                    if action_match and input_match:
                        action_name = action_match.group(1).strip()
                        raw_input = input_match.group(1).strip()

                        # Ensure the tool exists on the MCP Server
                        if action_name in available_tools:
                            print(f"  --> [MCP Client] Calling tool '{action_name}' on server (Please wait...)")
                            try:
                                # Clean up JSON parsing (LLMs sometimes add markdown formatting)
                                raw_input = raw_input.replace('```json', '').replace('```', '').strip()
                                
                                # Extract JSON if it's wrapped in code blocks
                                if raw_input.startswith('{'):
                                    # Find the last closing brace
                                    last_brace = raw_input.rfind('}')
                                    if last_brace != -1:
                                        raw_input = raw_input[:last_brace + 1]
                                
                                action_args = json.loads(raw_input)
                                
                                # Execute over the MCP Protocol
                                tool_result = await session.call_tool(action_name, arguments=action_args)
                                observation = tool_result.content[0].text if tool_result.content else "Success, but no output."
                                
                            except json.JSONDecodeError:
                                observation = f"Error: Action Input must be valid JSON. You provided: {raw_input}"
                            except Exception as e:
                                observation = f"Error calling MCP tool: {str(e)}"
                        else:
                            observation = f"Error: Tool '{action_name}' is not registered on the MCP Server."

                        print(f"👀 Observation: {observation}")
                        
                        # Append the real-world result back into the scratchpad
                        prompt_scratchpad += f"Observation: {observation}\n"
                    else:
                        print(f"⚠️ Formatting error. Nudging agent back on track...")
                        prompt_scratchpad += "Observation: Format error. You must provide an 'Action' and 'Action Input'.\n"
                        
                print("\n❌ Task aborted: Reached maximum iterations without a Final Answer.")

if __name__ == "__main__":
    
    # Hackathon Scenario: We give the agent a high-level goal, and it must figure out 
    # which MCP tools to call to assess the situation.
    
    scenario_goal = "A motion sensor alarm just triggered in Patient Room 402. Use your tools to investigate the patient's vitals, history, and room audio, then determine if a nurse needs to intervene."
    
    agent = MedicalReActMCPClient(model_name="amsaravi/medgemma-4b-it:q6")
    
    # Run the async loop
    asyncio.run(agent.run(scenario_goal))