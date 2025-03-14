from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

# Step 1: Define the tool
@tool
def get_user_profile(user_id: str):
    """Fetch the profile of a user by user ID."""
    user_data = {
        "101": {"name": "Alice", "age": 30, "location": "New York"},
        "102": {"name": "Bob", "age": 25, "location": "San Francisco"}
    }
    return user_data.get(user_id, "User profile not found.")

# Step 2: Set up the ToolNode with the get_user_profile tool
tools = [get_user_profile]
tool_node = ToolNode(tools)

# Step 3: Create an AIMessage for the tool call
message_with_tool_call = AIMessage(
    content="",
    tool_calls=[{
        "name": "get_user_profile",
        "args": {"user_id": "101"},
        "id": "tool_call_id",
        "type": "tool_call"
    }]
)

# Step 4: Set up the state with the messages key
state = {
    "messages": [message_with_tool_call]
}

# Step 5: Invoke the ToolNode with the state and get the result
result = tool_node.invoke(state)

# Output the result
print(result)



# {'messages': [ToolMessage(content="{'name': 'Alice', 'age': 30, 'location': 'New York'}", 
#     name='get_user_profile', tool_call_id='tool_call_id')]}

