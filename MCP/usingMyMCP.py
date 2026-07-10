import asyncio
from dotenv import load_dotenv

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    StdioConnectionParams,
)
from mcp import StdioServerParameters

load_dotenv()

MODEL = "gemini-3.1-flash-lite"

profile_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=[
                "run",
                "server.py",      # your MCP server file
            ],
        ),
        timeout=30,
    ),
)

async def main():

    tools = await profile_server.get_tools()

    print("Available Tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    agent = LlmAgent(
        name="profile_agent",
        model=MODEL,
        instruction="Use the available tools to answer the user's questions.",
        tools=tools,
    )

    runner = InMemoryRunner(agent=agent)

    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="nasir",
    )

    async for event in runner.run_async(
        user_id="nasir",
        session_id=session.id,
        new_message=types.UserContent(
            parts=[
                types.Part(
                    text="Who is Nasir? Tell me about his profile."
                )
            ]
        ),
    ):

        for call in event.get_function_calls():
            print(f"\nTool Called: {call.name}")
            print(call.args)

        if event.is_final_response():
            print("\nFinal Response:\n")
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())