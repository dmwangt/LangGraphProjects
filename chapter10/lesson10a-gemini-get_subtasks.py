from langgraph.graph import StateGraph, END
from typing import List, Dict, TypedDict
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode

# Initialize the Gemini LLM
llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai", temperature=1)


# Define the state (shared data)
class GraphState(TypedDict):
    task: str
    plan: List[str]
    results: Dict[str, str]  # subtask:result
    current_subtask_index: int


# Define tools for planning and execution
@tool
def generate_plan(task: str):
    """Generates a plan (list of subtasks) based on the given task."""
    prompt = f"""Given the task: '{task}', create a plan with a list of subtasks. \
        Each task should be a simple step that can be followed to achieve the task.\
        The result of the list should be like this:\
            Step 1\
            Step 2\
            ..."""

    response = llm.invoke(prompt)
    plan = response.content.split("\n")
    # planx = [p.strip("- ").strip() for p in plan if p.strip()]  # clean up list
    return {"plan": plan, "current_subtask_index": 0}


# Create the LangGraph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("planx", generate_plan)

# Add edges
workflow.set_entry_point("planx")
workflow.set_finish_point("planx")

# Compile the graph
app = workflow.compile()

# Example usage
inputs = {
    "task": "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace weighs. What are their combined weights in pounds?",
    "plan": [],
    "results": {},
    "current_subtask_index": 0,
}
for output in app.stream(inputs, stream_mode="values"):
    print(output)
