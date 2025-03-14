import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

from display_graph import display_graph

# Define shared state for the agent and sub-graphs, including memory
class ReActAgentState(TypedDict):
    message: str      # Current user message
    action: str       # Action determined by reasoning
    sub_action: str   # Sub-action determined by reasoning
    memory: dict      # Memory of past interactions

# Reasoning node that adapts based on user input and past interactions
def reasoning_node(state: ReActAgentState):
    query = state["message"]
    
    # Check if there is any past context stored in memory
    past_interactions = state.get("memory", {})
    
    # Personalized decision-making based on past interactions
    if "weather" in query:
        return {"action": "fetch_weather"}
    elif "news" in query:
        return {"action": "fetch_news"}
    elif "recommend" in query:
        if past_interactions.get("favorite_genre") == "science":
            return {"action": "recommendation", "sub_action": "science_book"}
        else:
            return {"action": "recommendation", "sub_action": "general_book"}
    else:
        return {"action": "unknown"}

# Sub-graph for fetching weather information
def weather_subgraph_node(state: ReActAgentState):
    return {"message": "The weather today is sunny."}

# Sub-graph for fetching news information
def news_subgraph_node(state: ReActAgentState):
    return {"message": "Here are the latest news headlines."}

# Sub-graph for providing a general book recommendation
def general_recommendation_node(state: ReActAgentState):
    return {"message": "I recommend reading 'The Pragmatic Programmer'."}

# Sub-graph for providing a science book recommendation based on user preferences
def science_recommendation_node(state: ReActAgentState):
    return {"message": "Since you like science, I recommend 'A Brief History of Time' by Stephen Hawking."}

# Sub-graph for updating memory (e.g., user preference updates)
def update_memory_node(state: ReActAgentState):
    if "recommend" in state["message"]:
        # Example of updating user's favorite genre in memory
        state["memory"]["favorite_genre"] = "science"
    return state

# Build sub-graphs for actions and memory
weather_subgraph_builder = StateGraph(ReActAgentState)
weather_subgraph_builder.add_node("weather_action", weather_subgraph_node)
weather_subgraph_builder.set_entry_point("weather_action")
weather_subgraph = weather_subgraph_builder.compile()

news_subgraph_builder = StateGraph(ReActAgentState)
news_subgraph_builder.add_node("news_action", news_subgraph_node)
news_subgraph_builder.set_entry_point("news_action")
news_subgraph = news_subgraph_builder.compile()

general_recommendation_builder = StateGraph(ReActAgentState)
general_recommendation_builder.add_node("general_recommendation_action", general_recommendation_node)
general_recommendation_builder.set_entry_point("general_recommendation_action")
general_recommendation_subgraph = general_recommendation_builder.compile()

science_recommendation_builder = StateGraph(ReActAgentState)
science_recommendation_builder.add_node("science_recommendation_action", science_recommendation_node)
science_recommendation_builder.set_entry_point("science_recommendation_action")
science_recommendation_subgraph = science_recommendation_builder.compile()

# Memory update sub-graph
memory_update_builder = StateGraph(ReActAgentState)
memory_update_builder.add_node("update_memory_action", update_memory_node)
memory_update_builder.set_entry_point("update_memory_action")
memory_update_subgraph = memory_update_builder.compile()

# Define dynamic reasoning node in the parent graph to route to the correct sub-graph
def reasoning_state_manager(state: ReActAgentState):
    if state["action"] == "fetch_weather":
        return weather_subgraph
    elif state["action"] == "fetch_news":
        return news_subgraph
    elif state["action"] == "recommendation":
        if state["sub_action"] == "science_book":
            return science_recommendation_subgraph
        else:
            return general_recommendation_subgraph
    else:
        return None

# Create the parent graph
parent_builder = StateGraph(ReActAgentState)
parent_builder.add_node("reasoning", reasoning_node)
parent_builder.add_node("action_dispatch", reasoning_state_manager)
parent_builder.add_node("update_memory", memory_update_subgraph)

# Define edges in the parent graph
parent_builder.add_edge(START, "reasoning")
parent_builder.add_edge("reasoning", "action_dispatch")
parent_builder.add_edge("action_dispatch", "update_memory")
parent_builder.add_edge("update_memory", END)

# Compile the parent graph
react_agent_graph = parent_builder.compile()

#visualise the graph 
#display_graph(react_agent_graph,file_name= os.path.basename(__file__))

# Initialize memory
checkpointer = MemorySaver()

# Test the agent with a weather-related query (memory will not affect this)
inputs_weather = {"message": "What is the weather today?", "memory": {}}
result_weather = react_agent_graph.invoke(inputs_weather)
print(result_weather["message"])

# Test the agent with a recommendation query (first time, no memory)
inputs_recommendation_first = {"message": "Can you recommend a good book?", "memory": {}}
result_recommendation_first = react_agent_graph.invoke(inputs_recommendation_first)
print(result_recommendation_first["message"])

# Simulate memory update after the recommendation (user prefers science books)
inputs_recommendation_second = {"message": "Can you recommend another book?", "memory": {"favorite_genre": "science"}}
result_recommendation_second = react_agent_graph.invoke(inputs_recommendation_second)
print(result_recommendation_second["message"])

