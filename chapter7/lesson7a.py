from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

import os
from langchain.chat_models import init_chat_model

# Initialize the LLM (using OpenAI in this example)
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded successfully", api_key)

# Initialize a ChatAI model
llm = init_chat_model("gemini-2.0-flash-exp", model_provider="google_genai")


# Define a multiplication tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies two numbers.
    """
    return a * b


# Bind the LLM with the tool
llm_with_tools = llm.bind_tools([multiply])


# Node that calls the LLM with tools bound
def tool_calling_llm(state: MessagesState):
    """
    Node that calls the LLM with tools bound.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build the workflow
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node(
    "tools", ToolNode([multiply])
)  # Tool node for handling tool invocation

# Add the conditional edge based on tool usage
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,  # Condition to decide if the assistant should call the tool
)

# Define edges to connect the nodes
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tools", END)  # If tool is called, terminate after tool execution

# Compile the graph
graph = builder.compile()


# Simulate invoking the graph
def simulate():
    user_input = {"messages": [("human", "Can you multiply 3 by 5?")]}
    result = graph.invoke(user_input)
    return result["messages"][-1].pretty_print()


print(simulate())
