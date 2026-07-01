import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

load_dotenv(override=True)

message = "In 1 sentence, what does it mean for an AI Agent to be autonomous"

reply = llm.invoke(message)

print(reply.content)

for chunk in llm.stream("Tell me a two line poem about autonomous agents"):
    print(chunk.content, end="", flush=True)

openrouter_llm = ChatOpenAI(
    model="anthropic/claude-haiku-4.5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

reply = openrouter_llm.invoke("In one sentence, what is LangChain?")
print(reply.content)

messages = [
    SystemMessage("You are a terse assistant who answers in exactly five words."),
    HumanMessage("What is the capital of France?"),
]

print(llm.invoke(messages).content)

# The exact same call using plain dictionaries, the format you already know
messages_as_dicts = [
    {"role": "system", "content": "You are a terse assistant who answers in exactly five words."},
    {"role": "user", "content": "What is the capital of France?"},
]
print(llm.invoke(messages_as_dicts).content)

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

response = llm_with_tools.invoke("What is the share price of Amazon?")
print("content:", repr(response.content))
print("tool_calls:", response.tool_calls)

# Start the conversation and keep the model's tool request in the history
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

class Company(BaseModel):
    name: str = Field(description="The company name")
    ticker: str = Field(description="The stock ticker symbol")
    founded_year: int = Field(description="The year the company was founded")

structured_llm = llm.with_structured_output(Company)

company = structured_llm.invoke("Tell me about Amazon the technology company")
print(company)
print("Just the ticker:", company.ticker)