import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import requests
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# Initialize the LLM (using Gemini in this example)
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded successfully", api_key)

# Initialize a ChatAI model
llm = init_chat_model(
    "gemini-2.0-flash-exp", model_provider="google_genai", temperature=0.8
)


# Define the mock APIs for demand and competitor pricing
@tool
def get_demand_data(product_id: str) -> dict:
    """Mock demand API to get demand data for a product."""
    return {"product_id": product_id, "demand_level": "low"}


@tool
def get_competitor_pricing(product_id: str) -> dict:
    """Mock competitor pricing API."""
    return {"product_id": product_id, "competitor_price": 95.0}


# List of tools for the agent to call
tools = [get_demand_data, get_competitor_pricing]

# Create the ReAct agent with tools for demand and competitor pricing
graph = create_react_agent(llm, tools=tools)


# Define the initial state of the agent
initial_messages = [
    (
        "system",
        """You are an AI agent that dynamically adjusts product prices based on market demand 
        and competitor prices. You have access to tools like get_demand_data and get_competitor_
        pricing — use them directly when needed, and return the final price recommendation.
        """,
    ),
    ("user", "What should be the price for product ID '12345'? "),
]

# Stream agent responses and decisions
inputs = {"messages": initial_messages}

# Simulate the agent reasoning, acting (calling tools), and observing
for state in graph.stream(inputs, stream_mode="values"):
    message = state["messages"][-1]  # Get the latest message in the interaction
    if isinstance(message, tuple):
        print(message)
    else:
        print(message.content)  # Ensure proper output of the agent's response
