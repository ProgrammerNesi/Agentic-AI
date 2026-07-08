import asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


async def main():
    client = MultiServerMCPClient({
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest", "--isolated"],
        }
    })

    async with client.session("playwright") as session:
        all_tools = await load_mcp_tools(session)

        allowed = {"browser_navigate", "browser_snapshot", "browser_close"}
        browser_tools = [t for t in all_tools if t.name in allowed]

        print(f"Loaded {len(browser_tools)} browser tools:")
        for t in browser_tools:
            print(" -", t.name)

        browser_agent = create_agent(
            model=llm,
            tools=browser_tools,
            system_prompt=(
                "You are a web research assistant with browser tools. "
                "Take the minimum number of actions needed (usually: navigate, then snapshot). "
                "As soon as you have the information requested, respond with the final answer "
                "in plain text and do not call any more tools."
            ),
        )

        async for step in browser_agent.astream(
            {"messages": [{
                "role": "user",
                "content": "Go to https://news.ycombinator.com and give top 3 stories."
            }]},
            config={"recursion_limit": 15},
        ):
            print(step)


asyncio.run(main())