import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from display_graph import display_graph
from langchain.chat_models import init_chat_model


# Initialize the LLM (using Gemini in this example)
api_key = os.getenv("GOOGLE_API_KEY")

print("API key loaded successfully", api_key)

# Initialize a ChatAI model
llm = init_chat_model(
    "gemini-2.0-flash-exp", model_provider="google_genai", temprature=0.8
)


# Define shared state for the agent and sub-graphs
class ReActAgentState(TypedDict):
    message: str  # Shared key between the parent graph and sub-graphs
    action: str  # Sub-graph specific key (what action to perform)
    sub_action: str  # Additional sub-action to perform in a more complex scenario


# Reasoning Node 1: Determines which action the agent should take based on the user query
def reasoning_node(state: ReActAgentState):
    query = state["message"]
    if "weather" in query:
        return {"action": "fetch_weather"}
    elif "news" in query:
        return {"action": "fetch_news"}
    elif "recommend" in query:
        return {"action": "recommendation", "sub_action": "book"}
    else:
        return {"action": "unknown"}


# Sub-graph for fetching weather information (acting phase)
def weather_subgraph_node(state: ReActAgentState):
    # Simulating a weather tool call
    return {"message": "The weather today is sunny."}


# Sub-graph for fetching news information (acting phase)
def news_subgraph_node(state: ReActAgentState):
    # Simulating a news tool call
    return {"message": "Here are the latest news headlines."}


# Sub-graph for providing a book recommendation (acting phase)
def recommendation_subgraph_node(state: ReActAgentState):
    if state.get("sub_action") == "book":
        return {"message": "I recommend reading 'The Pragmatic Programmer'."}
    else:
        return {"message": "I have no other recommendations at the moment."}

checkpointer = MemorySaver()

# Build sub-graph for fetching weather information
weather_subgraph_builder = StateGraph(ReActAgentState)
weather_subgraph_builder.add_node("weather_action", weather_subgraph_node)
weather_subgraph_builder.set_entry_point("weather_action")
weather_subgraph = weather_subgraph_builder.compile(checkpointer=checkpointer)

# Build sub-graph for fetching news information
news_subgraph_builder = StateGraph(ReActAgentState)
news_subgraph_builder.add_node("news_action", news_subgraph_node)
news_subgraph_builder.set_entry_point("news_action")
news_subgraph = news_subgraph_builder.compile(checkpointer=checkpointer)

# Build sub-graph for recommendations (e.g., book recommendation)
recommendation_subgraph_builder = StateGraph(ReActAgentState)
recommendation_subgraph_builder.add_node(
    "recommendation_action", recommendation_subgraph_node
)
recommendation_subgraph_builder.set_entry_point("recommendation_action")
recommendation_subgraph = recommendation_subgraph_builder.compile(checkpointer=checkpointer)


# Define dynamic reasoning node in the parent graph to route to the correct sub-graph
def reasoning_state_manager(state: ReActAgentState):
    if state["action"] == "fetch_weather":
        return weather_subgraph
    elif state["action"] == "fetch_news":
        return news_subgraph
    elif state["action"] == "recommendation":
        return recommendation_subgraph
    else:
        return None


# Create the parent graph
parent_builder = StateGraph(ReActAgentState)
parent_builder.add_node("reasoning", reasoning_node)
parent_builder.add_node("action_dispatch", reasoning_state_manager)

# Define edges in the parent graph
parent_builder.add_edge(START, "reasoning")
parent_builder.add_edge("reasoning", "action_dispatch")

# Compile the parent graph
react_agent_graph = parent_builder.compile(checkpointer=checkpointer)

# Visualise the graph
#display_graph(react_agent_graph, file_name=os.path.basename(__file__))
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
# Test the agent with a weather-related query
inputs_weather = {"message": "What is the weather today?"}
result_weather = react_agent_graph.invoke(inputs_weather, config)
print(result_weather["message"])

print("\n\n1. get state ",    react_agent_graph.get_state(config))


# Test the agent with a news-related query
inputs_news = {"message": "Give me the latest news."}
result_news = react_agent_graph.invoke(inputs_news, config)
print(result_news["message"])
print("\n\n2. get state ",    react_agent_graph.get_state(config))


# Test the agent with a recommendation-related query
inputs_recommendation = {"message": "Can you recommend a good book?"}
result_recommendation = react_agent_graph.invoke(inputs_recommendation, config)
print(result_recommendation["message"])
print("\n\n3. get state ",    react_agent_graph.get_state(config))
