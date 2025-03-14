from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
import os
import uuid
from dotenv import load_dotenv

from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages

from langchain.chat_models import init_chat_model

load_dotenv()
# Initialize the LLM (using Gemini in this example)
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded successfully", api_key)

# Initialize a ChatAI model
llm = init_chat_model("gemini-2.0-flash-exp", model_provider="google_genai")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str


# Initialize an in-memory store to store user information across sessions
in_memory_store = InMemoryStore()


# Function to store user information across sessions
def store_user_info(state: State, config, *, store=in_memory_store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    # Store user's name in memory
    memory_id = str(uuid.uuid4())
    user_name = state["user_name"]
    memory = {"user_name": user_name}

    # Save the memory in the in-memory store
    store.put(namespace, memory_id, memory)

    return {"messages": ["User information saved."]}


# Function to retrieve stored user information
def retrieve_user_info(state: State, config, *, store=in_memory_store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    # Retrieve stored user info
    memories = store.search(namespace)
    if memories:
        info = f"Hello {memories[-1].value['user_name']}, welcome back!"
    else:
        info = "I don't have any information about you yet."

    return {"messages": [info]}


# Function to handle storing or retrieving user info based on input
def call_model(state: State, config):
    last_message = state["messages"][-1].content.lower()

    if "remember my name" in last_message:
        # Store user's name in state and in memory
        user_name = last_message.split("remember my name is")[-1].strip()
        state["user_name"] = user_name
        return store_user_info(state, config)

    if "what's my name" in last_message or "what is my name" in last_message:
        # Retrieve the user's name from memory
        return retrieve_user_info(state, config)

    # Default LLM response for other inputs
    return {"messages": ["I didn't understand your request."]}


# Build the LangGraph workflow
workflow = StateGraph(MessagesState)
workflow.add_node("call_model", call_model)
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)

# Compile the graph with memory management
app_with_memory = workflow.compile(checkpointer=MemorySaver(), store=in_memory_store)


# Simulate sessions
def simulate_sessions():
    # First session: store user's name
    config = {"configurable": {"thread_id": "session_1", "user_id": "user_123"}}
    input_message = {
        "messages": [{"type": "user", "content": "Remember my name is Alice"}]
    }
    for chunk in app_with_memory.stream(
        input_message, config=config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Second session: retrieve user's name
    config = {"configurable": {"thread_id": "session_2", "user_id": "user_123"}}
    input_message = {"messages": [{"type": "user", "content": "What's my name?"}]}
    for chunk in app_with_memory.stream(
        input_message, config=config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()


# Run the session simulations
simulate_sessions()
