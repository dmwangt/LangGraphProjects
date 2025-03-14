# lesson6f.py
from langgraph.store.memory import InMemoryStore
import uuid

import json

# Initialize the memory store
in_memory_store = InMemoryStore()
# Define a user namespace (user_id + "memories")
user_id = "1"
namespace_for_memory = (user_id, "memoriesxxxx","what ever you want")
# Store a memory (food preference)
memory_id = str(uuid.uuid4())
memory = {"food_preference": "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)
# Retrieve the stored memories
memories = in_memory_store.search(namespace_for_memory)

print(json.dumps(memories[-1].dict(),indent=4))  # {'value': {'food_preference': 'I like pizza'}, ...}

# Output:
# {
# 'value': {'food_preference': 'I like pizza'},
# 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
# 'namespace': ['1', 'memories'],
# 'created_at': '2024-10-02T17:22:31.590602+00:00',
# 'updated_at': '2024-10-02T17:22:31.590605+00:00'
# }
