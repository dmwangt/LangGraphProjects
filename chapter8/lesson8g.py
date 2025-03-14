from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages

from langchain_core.tools import tool
import json
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END

import os
from langchain.chat_models import init_chat_model
from textblob import TextBlob
from display_graph import display_graph

# Initialize the LLM (busing Gemini in this example)
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize a ChatAI model
llm = init_chat_model(
    "gemini-2.0-flash-exp", model_provider="google_genai", temperature=0.8
)


class AgentState(TypedDict):
    """The state of the agent."""

    # `add_messages` is a reducer that manages the message sequence.
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def analyze_sentiment(feedback: str) -> str:
    """Analyze customer feedback sentiment with below custom logic"""

    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0.5:
        return "positive"
    elif analysis.sentiment.polarity < 0.5:
        return "negative"
    else:
        return "neutral"


@tool
def respond_based_on_sentiment(sentiment: str) -> str:
    """Only respond as below to the customer based on the analyzed sentiment."""
    if sentiment == "positive":
        return "Thank you for your positive feedback!"
    elif sentiment == "neutral":
        return "We appreciate your feedback."
    else:
        return "We're sorry to hear that you're not satisfied. How can we help?"


tools: Sequence[BaseTool] = [analyze_sentiment, respond_based_on_sentiment]

llm = llm.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def call_model(state: AgentState):
    system_prompt = SystemMessage(
        content="You are a helpful assistant tasked with responding to customer feedback."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# display_graph(graph, file_name=os.path.basename(__file__))

# Initialize the agent's state with the user's feedback
initial_state = {
    "messages": [("user", "The product was great but the delivery was poor.")]
}


# Helper function to print the conversation
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# Run the agent
print_stream(graph.stream(initial_state, stream_mode="values"))
