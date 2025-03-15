from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import os
from langchain.chat_models import init_chat_model
from textblob import TextBlob

# Initialize the LLM (using Gemini in this example)
api_key = os.getenv("GOOGLE_API_KEY")

llm = init_chat_model(
    "gemini-2.0-flash-exp", model_provider="google_genai", temperature=0.8
)


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], ...]


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


tools = [analyze_sentiment, respond_based_on_sentiment]

agent = create_react_agent(llm, tools=tools)

# Initialize the agent's state with the user's feedback
initial_state = {
    "messages": [
        (
            "system",
            "You are a helpful assistant tasked with responding to customer feedback.",
        ),
        ("user", "The product was great but the delivery was poor."),
    ]
}

# Run the agent and print all messages in the result
response = agent.invoke(initial_state)
for msg in response["messages"]:
    if isinstance(msg, tuple):
        print(f"{msg[0].capitalize()}: {msg[1]}")
    else:
        print(msg.content)
