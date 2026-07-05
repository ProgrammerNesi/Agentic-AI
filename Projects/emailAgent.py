from typing import TypedDict, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class TriageState(TypedDict):
    ticket_text: str
    category: Optional[Literal["billing", "technical", "general", "spam"]]
    urgency: Optional[Literal["low", "medium", "high"]]
    draft_response: Optional[str]
    human_decision: Optional[Literal["approve", "edit", "reject"]]
    edited_response: Optional[str]
    revision_count: int
    final_status: Optional[str]


from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)

class Classification(BaseModel):
    category: Literal["billing", "technical", "general", "spam"]
    urgency: Literal["low", "medium", "high"]
    reasoning: str = Field(description="One sentence justification")

classifier = llm.with_structured_output(Classification)

def classify_ticket(state: TriageState) -> dict:
    result = classifier.invoke(
        f"""Classify the following customer support ticket.

Ticket:
{state['ticket_text']}
"""
    )

    print(
        f"[classify] category={result.category} "
        f"urgency={result.urgency} "
        f"({result.reasoning})"
    )

    return {
        "category": result.category,
        "urgency": result.urgency,
    }

def route_after_classify(state: TriageState) -> str:
    if state["category"] == "spam":
        return "ignore"
    if state["urgency"] == "high":
        return "escalate"
    return "draft_response"


def draft_response(state: TriageState) -> dict:
    prompt = (
        f"Write a short, professional customer support reply.\n"
        f"Category: {state['category']}\n"
        f"Ticket: {state['ticket_text']}\n"
        f"Modification: {state.get('edited_response', 'None')}"
    )
    response = llm.invoke(prompt)
    return {
        "draft_response": response.content,
        "revision_count": state.get("revision_count", 0) + 1,
    }


from langgraph.types import interrupt

def human_review(state: TriageState) -> dict:
    decision = interrupt({
        "ticket": state["ticket_text"],
        "category": state["category"],
        "urgency": state["urgency"],
        "draft": state["draft_response"],
        "instructions": "Type: approve | edit:<your replacement text> | reject",
    })
    if decision.startswith("edit:"):
        return {"human_decision": "edit", "edited_response": decision[5:].strip()}
    elif decision == "reject":
        return {"human_decision": "reject"}
    else:
        return {"human_decision": "approve"}
    
def route_after_review(state: TriageState) -> str:
    if state["human_decision"] in ("approve"):
        return "send"
    if state["revision_count"] >= 5:
        return "escalate"   # give up auto-drafting, hand to a person
    return "draft_response" # cycle back for a fresh draft


def send_email(state: TriageState) -> dict:
    final_text = state["draft_response"]
    print(f"\n📤 EMAIL SENT:\n{final_text}\n")
    return {"final_status": "sent"}

def escalate(state: TriageState) -> dict:
    print(f"\n🚨 ESCALATED TO HUMAN AGENT:\n{state['ticket_text']}\n")
    return {"final_status": "escalated"}

def ignore_ticket(state: TriageState) -> dict:
    print("\n🗑️ Marked as spam. Ignored.\n")
    return {"final_status": "ignored"}


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(TriageState)

builder.add_node("classify", classify_ticket)
builder.add_node("draft_response", draft_response)
builder.add_node("human_review", human_review)
builder.add_node("send", send_email)
builder.add_node("escalate", escalate)
builder.add_node("ignore", ignore_ticket)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_after_classify, {
    "draft_response": "draft_response",
    "escalate": "escalate",
    "ignore": "ignore",
})
builder.add_edge("draft_response", "human_review")
builder.add_conditional_edges("human_review", route_after_review, {
    "send": "send",
    "draft_response": "draft_response",  # the cycle
    "escalate": "escalate",
})
builder.add_edge("send", END)
builder.add_edge("escalate", END)
builder.add_edge("ignore", END)

checkpointer = MemorySaver()  # swap for SqliteSaver/PostgresSaver later
graph = builder.compile(checkpointer=checkpointer)

from langgraph.types import Command
import uuid

def run_ticket(ticket_text: str):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(
        {"ticket_text": ticket_text, "revision_count": 0},
        config=config,
    )
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("email_agent_graph.png", "wb") as f:
        f.write(png_bytes)
    print("Graph saved as email_agent_graph.png")
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n--- HUMAN REVIEW NEEDED ---")
        print(f"Category: {payload['category']} | Urgency: {payload['urgency']}")
        print(f"Ticket: {payload['ticket']}")
        print(f"Draft:\n{payload['draft']}")
        print(payload["instructions"])

        human_input = input("\nYour decision: ").strip()
        result = graph.invoke(Command(resume=human_input), config=config)

    print(f"\n✅ Final status: {result.get('final_status')}")

if __name__ == "__main__":
    ticket = input("Paste the customer ticket/email:\n> ")
    run_ticket(ticket)