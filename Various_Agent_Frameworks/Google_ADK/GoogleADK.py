import asyncio
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    StdioConnectionParams,
)
from mcp import StdioServerParameters

# -----------------------------
# Environment
# -----------------------------
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
load_dotenv(override=True)

MODEL = "gemini-3.1-flash-lite"

# Folder the MCP filesystem server can access
workspace = Path.cwd()


# -----------------------------
# Custom Tool
# -----------------------------
def notifyUser(message: str) -> str:
    """Notify the user."""
    print(f"\n🔔 Notification: {message}")
    return "Notification sent."


# -----------------------------
# MCP Filesystem Tool
# -----------------------------
filesystem = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(workspace),
            ],
            cwd=str(workspace),
        ),
        timeout=60,
    ),
    errlog=subprocess.DEVNULL,
)

# -----------------------------
# Agent
# -----------------------------
worker = LlmAgent(
    name="task_worker",
    model=MODEL,
    instruction="""
You are a careful AI assistant.

You have access to filesystem tools.

Complete the user's task by reading and writing files only inside the provided workspace.
Use notifyUser after finishing.
""",
    tools=[
        filesystem,
        notifyUser,
    ],
)


async def main():
    runner = InMemoryRunner(agent=worker)

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
                    text="Read notes.txt, translate it into Hindi, save it as hindi.txt, and notify me when finished."
                )
            ]
        ),
    ):
        for call in event.get_function_calls():
            print(f"\nTool Call -> {call.name}")
            print(call.args)

        for response in event.get_function_responses():
            print(f"\nTool Response -> {response.name}")

        if event.is_final_response():
            print("\n===== FINAL RESPONSE =====")
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())