from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

# Initialize the LLM (using OpenAI in this example)
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded successfully", api_key)

# Initialize a ChatAI model
model = init_chat_model("gemini-2.0-flash-exp", model_provider="google_genai")


# Function to handle the user query and call the LLM
def call_llm(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages[-1].content)
    return {"messages": [response]}


# Define the graph
workflow = StateGraph(MessagesState)

# Add the node to call the LLM
workflow.add_node("call_llm", call_llm)

# Define the edges (start -> LLM -> end)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)

# Compile the workflow
app = workflow.compile()

# Example input message from the user
input_message = {
    "messages": [
        ("system", "you are a geography expert."),
        ("human", "What is the capital of Kenya?"),
    ]
}

# Run the workflow
for chunk in app.stream(input_message, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
