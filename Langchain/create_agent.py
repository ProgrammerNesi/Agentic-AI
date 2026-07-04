# from dotenv import load_dotenv
# from IPython.display import Image, display
# from pydantic import BaseModel, Field
# from langchain.agents import create_agent
# from langchain.agents.middleware import wrap_tool_call
# from langchain_core.tools import tool
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_mcp_adapters.client import MultiServerMCPClient

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.middleware import wrap_tool_call



from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv(override=True)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # free tier
    temperature=0,
)
# Creating an agent with tool with LangChain Create-Agent

# @tool
# def get_weather(city: str) -> str:
#     """Return today's weather for a city."""
#     pretend = {"London": "rainy, 14 degrees", "Rome": "sunny, 27 degrees"}
#     return pretend.get(city, "clear, 20 degrees")

# @tool
# def get_population(city: str) -> str:
#     """Return the population of a city."""
#     pretend = {"London": "8.9 million", "Rome": "2.8 million"}
#     return pretend.get(city, "unknown")

# agent = create_agent(
#     model="google_genai:gemini-3.1-flash-lite",
#     tools=[get_weather, get_population],
#     system_prompt="You are a travel assistant. Use your tools to answer questions about cities.",
# )

# result = agent.invoke({"messages": [{"role": "user", "content": "What is the weather and population of Rome?"}]})
# print(result["messages"][-1].content)

# #adding memory to the agent

# memory_agent = create_agent(
#     model="google_genai:gemini-3.1-flash-lite",
#     tools=[get_weather],
#     checkpointer=MemorySaver(),
# )

# config = {"configurable": {"thread_id": "trip-planning"}}
# memory_agent.invoke({"messages": [{"role": "user", "content": "I am planning a trip to London."}]}, config=config)
# result = memory_agent.invoke({"messages": [{"role": "user", "content": "What is the weather like where I am going on my trip?"}]}, config=config)
# print(result["messages"][-1].content)


# # Structured Output

# class CityReport(BaseModel):
#     city: str = Field(description="The city name")
#     weather: str = Field(description="A short weather description")
#     population: str = Field(description="The population")

# report_agent = create_agent(
#     model="google_genai:gemini-3.1-flash-lite",
#     tools=[get_weather, get_population],
#     response_format=CityReport,
# )

# result = report_agent.invoke({"messages": [{"role": "user", "content": "Give me a report on London."}]})
# report = result["structured_response"]
# print(report)
# print("Just the weather:", report.weather)


# # Middleware/GuardRails

# @wrap_tool_call
# def log_tool_calls(request, handler):
#     call = request.tool_call
#     print(f"  [middleware] calling {call['name']} with {call['args']}")
#     return handler(request)

# watched_agent = create_agent(
#     model="google_genai:gemini-3.1-flash-lite",
#     tools=[get_weather, get_population],
#     system_prompt="You are a travel assistant. Use your tools.",
#     middleware=[log_tool_calls],
# )

# result = watched_agent.invoke({"messages": [{"role": "user", "content": "Weather and population of London and Rome?"}]})
# print("\nFinal answer:", result["messages"][-1].content)

# Node and Playwright

async def main():
    client = MultiServerMCPClient({
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest", "--isolated"],
        }
    })

    browser_tools = await client.get_tools()

    print(f"Loaded {len(browser_tools)} browser tools:")
    for t in browser_tools:
        print(" -", t.name)

    browser_agent = create_agent(
        model=llm,
        tools=browser_tools,
        system_prompt="You are a web research assistant.",
    )

    result = await browser_agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "Go to https://news.ycombinator.com and give any text from it."
        }]
    })

    print(result["messages"][-1].content)

asyncio.run(main())

