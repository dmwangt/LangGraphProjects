# Import necessary libraries
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

# Initialize the LLM (using Gemini in this example)
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded successfully", api_key)

# Step 2: Define the tool to get weather information
@tool
def get_weather(location: str):
    """Fetch the current weather for a specific location."""
    weather_data = {
        "San Francisco": "It's 60 degrees and foggy.",
        "New York": "It's 90 degrees and sunny.",
        "London": "It's 70 degrees and cloudy."
    }
    return weather_data.get(location, "Weather information is unavailable for this location.")

# Step 3: Initialize the LLM (Gemini's 2.0-flash-exp model) and bind the tool
tool_node = ToolNode([get_weather], handle_tool_errors=False)

# Initialize a ChatAI model
llm = init_chat_model("gemini-2.0-flash-exp", model_provider="google_genai")

model = llm.bind_tools([get_weather])

# Step 4: Function to handle user queries and process LLM
def call_llm(state: MessagesState):
    messages = state["messages"]
    # The LLM will decide if it should invoke a tool based on the user input
    response = model.invoke(messages[-1].content)
    return {"messages": [response]}

# Step 5: Create the LangGraph workflow
workflow = StateGraph(MessagesState)

# Step 6: Add the LLM node and tool node to the workflow
workflow.add_node("call_llm", call_llm)
workflow.add_node("tool_node", tool_node)

# Step 7: Define edges to control the flow
def should_use_tool(state):
    messages = state["messages"]
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tool_node"
    else:
        return END

workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_use_tool)
workflow.add_edge("tool_node", END)

# Step 8: Compile the workflow
app = workflow.compile()

# Step 9: Function to interact with the agent continuously
def interact_with_agent():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending the conversation.")
            break

        # Prepare the user input for processing
        input_message = {
            "messages": [("human", user_input)]
        }

        # Process the input through the workflow and return the response
        for chunk in app.stream(input_message, stream_mode="values"):
            if "messages" in chunk:
                chunk["messages"][-1].pretty_print()

# Step 10: Start interacting with the AI agent
interact_with_agent()