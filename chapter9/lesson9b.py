import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

# Set up your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI instance
llm = init_chat_model(
    "gemini-2.0-flash-exp", model_provider="google_genai", temperature=0.8
)

# Define the state structure
class State(TypedDict):
    input: str
    draft_content: str

# Define node functions
def create_draft(state: State):
    print("--- Generating Draft with ChatOpenAI ---")
    prompt = f"Write a blog post on the topic: {state['input']}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    state["draft_content"] = response.content
    print(f"Generated Draft:\n{state['draft_content']}")
    return state

def review_draft(state: State):
    print("--- Reviewing Draft ---")
    print(f"Draft for review:\n{state['draft_content']}")
    user_approval = input("Do you approve the draft for publishing? (yes/no/modification): ")

    if user_approval.lower() == "yes":
        return "publish"
    elif user_approval.lower() == "modification" or user_approval.lower() == "m":
        updated_draft = input("Please modify the draft content:\n")
        state["draft_content"] = updated_draft
        return "create_draft" # loop back to create_draft with the modified draft.
    else:
        return END

def publish_content(state: State):
    print("--- Publishing Content ---")
    print(f"Published Content:\n{state['draft_content']}")
    return END

# Build the graph
builder = StateGraph(State)
builder.add_node("create_draft", create_draft)
builder.add_node("review_draft", review_draft)
builder.add_node("publish_content", publish_content)

# Define flow
builder.add_edge(START, "create_draft")
builder.add_edge("create_draft", "review_draft")
# builder.add_conditional_edges(
#     "review_draft",
#     {
#         "publish": "publish_content",
#         "create_draft": "create_draft",
       
#     },
# )

# Compile the graph
graph = builder.compile()

# Run the graph
initial_input = {"input": "The importance of AI in modern content creation"}
thread_config = RunnableConfig(configurable={"thread_id": "1"})

for output in graph.stream(initial_input, thread_config, stream_mode="values"):
    print(output)