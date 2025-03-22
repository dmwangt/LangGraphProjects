import os
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field
from langchain.tools import tool, BaseTool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import ToolExecutor
from display_graph import display_graph

# Initialize the Gemini LLM
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


# Define the state (shared data)
class GraphState(TypedDict):
    task: str
    plan: List[str]
    results: Dict[str, str] | None  ## subtask:result
    current_subtask_index: int


class ExecuteStepInput(BaseModel):
    subtask: str = Field(..., description="The subtask to execute")


# Define tools for planning and execution
def generate_plan(task: str):
    """Generates a plan (list of subtasks) based on the given task."""
    prompt = f"""Given the task: '{task}', create a detailed plan with a list of subtasks seperated by newlines. \
        Each task should be a simple step that can be followed to achieve the task. \
        Make sure to include all necessary steps and details. The result of the final step \
        should be the final answer. Do not skip any steps.
        the plan should be a list of subtasks seperated by newlines."""
    response = llm.invoke([HumanMessage(content=prompt)])

    plan = response.content.split("\n")  # Simple split by newline
    plan = [p.strip("- ").strip() for p in plan if p.strip()]  # clean up list
    return {"plan": plan}


@tool(args_schema=ExecuteStepInput)
def execute_step(subtask: str):
    """Executes a single step (subtask) and returns the result."""
    prompt = subtask
    print("---------------> step: ", prompt)
    response = llm.invoke(prompt)
    return response.content


# Create Tool Executors

tools: Dict[str, BaseTool] = {}
tools["execute_step"] = execute_step


def execute_subtask(state: GraphState):
    """Executes a subtask."""
    plan = state["plan"]
    index = state["current_subtask_index"]
    subtask = plan[index]
    print(f"Executing subtask: {subtask}")
    
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    if( state["results"] is None):
        results = []
    else:
        results = state["results"].values()
    # Format the past results
    past_results = "\n".join(f"{i+1}. {step}" for i, step in enumerate(results))
    task = plan[index]
    task_formatted = f"For the following plan of {state['task']}:\n{plan_str}\n\nWe have thoes data: {past_results}. \n\nYou are tasked with executing step {index}, {task} and provide the result."

    tool_name = "execute_step"

    tool = tools[tool_name]
    tool_input = task_formatted
    result = tool.invoke(tool_input)

    subtask_result = result
    results = state.get("results", {})
    
    results[subtask] =f"{subtask}: {subtask_result}"
    return {"results": results, "current_subtask_index": index + 1}


def should_continue(state: GraphState) -> str:
    """Determines if there are more subtasks to execute."""
    current_index = state["current_subtask_index"]
    plan = state["plan"]
    if current_index < len(plan):
        return "execute"
    else:
        return "end"


# Create the LangGraph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("planx", generate_plan)
workflow.add_node("execute", execute_subtask)

# Add edges
workflow.set_entry_point("planx")
workflow.add_edge("planx", "execute")
workflow.add_conditional_edges(
    "execute",
    should_continue,
    {
        "execute": "execute",
        "end": END,
    },
)

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
