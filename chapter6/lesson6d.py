from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
import uuid
from langchain.chat_models import init_chat_model

# Initialize the LLM (using OpenAI in this example)
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded successfully", api_key)


# Initialize a ChatAI model
model = init_chat_model("gemini-2.0-flash-exp", model_provider="google_genai")

# Initialize an in-memory store to store user information across sessions
in_memory_store = InMemoryStore()

# Function to call the LLM and respond
def call_llm(state: MessagesState, config, *, store=in_memory_store):
    messages = state["messages"]
    user_input = messages[-1].content

    # Check if the user provided their name
    if "my name is" in user_input.lower():
        name = user_input.split("my name is")[-1].strip().title()
        # state["user_name"] = name

        return {"messages": [f"Nice to meet you, {name}! Your name has been saved."]}

    
    # Default behavior: call the LLM to handle other inputs
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)

# Add the node to handle user input and call the LLM
workflow.add_node("call_llm", call_llm)

# Define the edges (start -> LLM -> end)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)

# Initialize the MemorySaver for short-term memory
checkpointer = MemorySaver()

# Compile the workflow with both the checkpointer and memory store
app_with_memory_store = workflow.compile(checkpointer=checkpointer, store=in_memory_store)

# Function to simulate interacting with the agent across sessions
def interact_with_agent_across_sessions():
    while True:
        # Simulate a new session by allowing the user to input a thread ID and user ID
        thread_id = input("Enter thread ID (or 'new' for a new session): ")
        if thread_id.lower() in ["exit", "quit"]:
            print("Ending the conversation.")
            break
        
        if thread_id.lower() == "new":
            thread_id = f"session_{os.urandom(4).hex()}"  # Generate a unique session ID
        
        user_id = input("Enter your user ID (or use a new one): ")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "end session"]:
                print(f"Ending session {thread_id}.")
                break
            
            input_message = {
                "messages": [("human", user_input)]
            }

            # Invoke the graph with the correct thread ID and user ID to maintain memory across sessions
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            for chunk in app_with_memory_store.stream(input_message, config=config, stream_mode="values"):
                chunk["messages"][-1].pretty_print()

# Start interacting with the memory-persistent agent
interact_with_agent_across_sessions()
