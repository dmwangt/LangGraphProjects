import os
from typing import Sequence
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool, tool

from langchain.chat_models import init_chat_model


# Initialize the LLM (using Gemini in this example)
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded successfully", api_key)

# Initialize a ChatAI model
llm = init_chat_model("gemini-2.0-flash-exp", model_provider="google_genai", temprature = 0.8)


# Define tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools= [add, multiply,divide]

# Create the ReAct agent
graph = create_react_agent(model=llm, tools=tools)


# User input
inputs = {
    "messages": [
        ("system", "you are an expert on accouting."),
       
        ("user", "Add 32 and 4. Multiply the result by 2 and divide by 4. Is it a odd number(anwser is yes or not)?")]
}

# Run the ReAct agent
messages = graph.invoke(inputs)
for message in messages["messages"]:
    message.pretty_print()
