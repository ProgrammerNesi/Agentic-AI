import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(override=True)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


@tool
def get_share_price(symbol: str) -> float:
    """Return the current share price for a given ticker symbol."""
    fake_prices = {"AAPL": 241.5, "GOOG": 168.2, "AMZN": 198.0}
    return fake_prices.get(symbol.upper(), 0.0)


print("name:", get_share_price.name)
print("description:", get_share_price.description)
print("args:", get_share_price.args)
print("called directly:", get_share_price.invoke({"symbol": "AAPL"}))


llm_with_tools = llm.bind_tools([get_share_price])

# Calling the LLM with a message that triggers the tool call
response = llm_with_tools.invoke("What is the share price of Amazon?")
print("content:", repr(response.content))
print("tool_calls:", response.tool_calls)


conversation = [HumanMessage("What is the share price of Amazon?")]
ai_message = llm_with_tools.invoke(conversation)
conversation.append(ai_message)


# Run each requested tool and add its result as a ToolMessage
for call in ai_message.tool_calls:
    if call["name"] == "get_share_price":
        result = get_share_price.invoke(call["args"])
        conversation.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

# Invoke again, now that the model can see the tool result
final = llm_with_tools.invoke(conversation)
print(final.content)


# Structured output example
class Company(BaseModel):
    name: str = Field(description="The company name")
    ticker: str = Field(description="The stock ticker symbol")
    founded_year: int = Field(description="The year the company was founded")

structured_llm = llm.with_structured_output(Company)

company = structured_llm.invoke("Tell me about Amazon the technology company")
print(company)
print("Company name:", company.name)
print("Ticker:", company.ticker)
print("Founded year:", company.founded_year)
