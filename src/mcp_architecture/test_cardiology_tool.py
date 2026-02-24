"""
Test script to specifically demonstrate the cardiology sentinel tool
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MCP Client SDK Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama

load_dotenv()


async def test_cardiology_tool():
    """Test the cardiology sentinel tool directly"""
    
    print("\n[System] Testing Cardiology Sentinel Tool via MCP...\n")
    
    # Configure the connection to the local FastMCP Server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["medical_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Discover available tools
            tools_response = await session.list_tools()
            print(f"📋 Available tools: {[t.name for t in tools_response.tools]}\n")
            
            # Test data: simulated vitals showing cardiac distress
            vitals_data = """Time: 00:00 - HR: 115, SPO2: 88, RESP: 24
Time: 00:01 - HR: 118, SPO2: 87, RESP: 25
Time: 00:02 - HR: 120, SPO2: 86, RESP: 26
Time: 00:03 - HR: 122, SPO2: 85, RESP: 27
Time: 00:04 - HR: 125, SPO2: 84, RESP: 28"""
            
            print("🫀 Testing Cardiology Sentinel for Patient 402...")
            print(f"   Vitals data:\n{vitals_data}\n")
            
            # Call the cardiology sentinel tool
            result = await session.call_tool(
                "analyze_cardiology_sentinel",
                arguments={
                    "patient_id": "402",
                    "vitals_text": vitals_data
                }
            )
            
            output = result.content[0].text if result.content else "No output"
            print("=" * 80)
            print(output)
            print("=" * 80)
            print("\n✅ Cardiology Sentinel Tool Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_cardiology_tool())
