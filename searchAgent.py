import os
import asyncio
from typing import Dict

from dotenv import load_dotenv
from ddgs import DDGS
from openai import AsyncOpenAI, api_key

from agents import (
    Agent,
    Runner,
    trace,
    function_tool,
    OpenAIChatCompletionsModel,
)
from agents.model_settings import ModelSettings

from IPython.display import display, Markdown

# Load environment variables
load_dotenv()
# Gemini client using OpenAI-compatible endpoint
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

INSTRUCTIONS = """
You are a research assistant.

Given a search term, search the web and produce a concise summary.

Requirements:
- 2-3 paragraphs
- Less than 300 words
- Capture the main points
- Ignore fluff
- No additional commentary
"""

@function_tool
def web_search(subject: str) -> Dict[str, str]:
    """
    Search the web and return top results.
    """
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(subject, max_results=5):
            results.append(
                f"- {r.get('title', 'No Title')}: "
                f"{r.get('body', '')}"
            )
        print(f"Search results for '{subject}':\n" + "\n".join(results))
    return {
        "summary": "\n".join(results)
    }

# Gemini model wrapped for OpenAI Agents SDK
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=gemini_client,
)

search_agent = Agent(
    name="Search Agent",
    instructions=INSTRUCTIONS,
    tools=[web_search],
    model=gemini_model,
    model_settings=ModelSettings(
        tool_choice="required"
    ),
)

async def main():
    query = "Latest AI Agent frameworks in 2025"

    with trace("Search"):
        result = await Runner.run(
            search_agent,
            query
        )
    print(result)
    print(result.final_output)
    display(Markdown(result.final_output))
    print("\n===== SUMMARY =====\n")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())