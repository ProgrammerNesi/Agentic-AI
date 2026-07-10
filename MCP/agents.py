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

load_dotenv()

MODEL = "gemini-3.1-flash-lite"

workspace = Path.cwd() / "workspace"
workspace.mkdir(exist_ok=True)

# ----------------------------
# Filesystem MCP
# ----------------------------

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

# ----------------------------
# Playwright MCP
# ----------------------------

browser = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "@playwright/mcp@latest",
            ],
        ),
        timeout=120,
    ),
    errlog=subprocess.DEVNULL,
)


async def main():

    print("=" * 60)
    print("Filesystem MCP Tools")
    print("=" * 60)

    fs_tools = await filesystem.get_tools()

    for tool in fs_tools:
        print(f"{tool.name}")
        print(f"  Description: {tool.description}\n")

    print()

    print("=" * 60)
    print("Playwright MCP Tools")
    print("=" * 60)

    browser_tools = await browser.get_tools()

    for tool in browser_tools:
        print(f"{tool.name}")
        print(f"  Description: {tool.description}\n")

    print()

    agent = LlmAgent(
        name="web_agent",
        model=MODEL,
        instruction="""
You are an expert web research assistant.

Use the Playwright tools to browse the web.

Find the about my the ficaso on the recent fifa match of egypt vs argentina and write description of it in fifa2.txt.

Then use the filesystem tools to create
workspace/certificates.md containing:
-list of certificates i have
Do not answer without writing the markdown file.
""",
        tools=[
            *fs_tools,
            *browser_tools,
        ],
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
                    text="Find the about recent event that occured in the fifa match of egypt vs argentina and write about what has happened in a file fifa2.txt."
                )
            ]
        ),
    ):

        for call in event.get_function_calls():
            print(f"\nTool Call: {call.name}")
            print(call.args)

        if event.is_final_response():
            print("\nFinal Response\n")
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())