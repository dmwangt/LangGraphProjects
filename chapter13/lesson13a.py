# pip install langchain_experimental

"""
Corrective Retrieval-Augmented Generation (CRAG) is an innovative
approach designed to enhance the robustness of standard Retrieval-
Augmented Generation (RAG) systems by addressing a crucial limitation:
what happens when retrieval returns irrelevant or incorrect results.
"""

from typing import Annotated, Hashable

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_experimental.tools.python.tool import PythonREPLTool

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from typing import Literal

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

from langchain_core.messages import HumanMessage


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }


members: list[str] = ["Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members


class routeResponse(BaseModel):
    next: Literal[*options]


prompt = ChatPromptTemplate.from_messages(
    [
        AIMessage(system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        AIMessage(
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=0)


def supervisor_agent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    msg = supervisor_chain.invoke(state)

    # It is the routeResponse model that defines the output
    # The output is a dictionary with the key 'next'
    # The value is the next worker to act
    # or FINISH
    return msg


import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# you can use different models for each agent for specific tasks
# or use the same model with different tools
research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_react_agent(llm, tools=[python_repl_tool])
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
from typing import Dict, Union

conditional_map: dict[Hashable, str] = {str(k): k for k in members}
conditional_map["FINISH"] = END

# since the supervisor's output is a dictionary with the key 'next'
# we can to use the key to access the value to get the next node to route to
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

for s in graph.stream(
    {
        "messages": [
              HumanMessage(content="my income is 40000, and eacher year, it will be increased by 2%. What will my income be in 10 years?")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")
        
"""
find out latest langgraph of human in loop funtion, and writ a code to demostrate that feature.


"""        

"""
for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="find out latest langgraph of human in loop funtion, and writ a code to demostrate that feature.")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")
        
{'supervisor': {'next': 'Researcher'}}
----
{'Researcher': {'messages': [HumanMessage(content='Recent advances in AI include:\n\n*   **Generative AI:** Explosive growth of generative AI tools, with one-third of organizations using them regularly in at least one business function.\n*   **AI in Cybersecurity:** AI is being used to enhance threat detection and improve cybersecurity measures.\n*   **AI in Healthcare:** AI is being used to augment AI training by computer-generated X-rays to create larger databases for identifying rare conditions.\n*   **AI in the Automobile Sector:** AI is being heavily invested in for autonomous driving.\n*   **Multi-modal AI:** Combining text, image, and speech processing.\n*   **Explainable AI:** Increasing transparency in decision-making.\n*   **Digital Twins:** Gaining prominence for simulating real-world objects digitally.\n*   **AI in Social Media:** AI algorithms enhance user experiences and drive engagement.', additional_kwargs={}, response_metadata={}, name='Researcher')]}}
----
{'supervisor': {'next': 'Coder'}}
----
{'Coder': {'messages': [HumanMessage(content="That's a great summary of recent AI advances! Is there anything specific you'd like me to elaborate on, or any particular area you'd like more information about? For example, I could try to find more details on a specific application of AI in healthcare, or delve deeper into the challenges of explainable AI.", additional_kwargs={}, response_metadata={}, name='Coder')]}}
----
{'supervisor': {'next': 'Researcher'}}
----
{'Researcher': {'messages': [HumanMessage(content='Please provide more details on AI in cybersecurity, including specific examples of its application and its effectiveness.', additional_kwargs={}, response_metadata={}, name='Researcher')]}}
----
{'supervisor': {'next': 'Researcher'}}
----
{'Researcher': {'messages': [HumanMessage(content='AI is being integrated into cybersecurity to enhance threat detection, adapt to new threats, and scale security measures efficiently. Unlike traditional cybersecurity tools that rely on predefined rules, AI-driven systems learn from experience, allowing them to predict, detect, and respond more effectively to both known and unknown threats. AI improves the speed and accuracy of threat detection, identifying potential risks before they escalate. AI can analyze large volumes of data, ensuring comprehensive monitoring as organizations grow, and continuously learns from new data to counteract emerging cyber threats. AI-based systems can also predict potential breach points, enabling better resource allocation and improved cyber resilience. Furthermore, AI can automate recommendations to configure and enhance security controls and processes. Some specific applications include:\n\n*   **Enhanced Threat Detection:** AI algorithms analyze network traffic, system logs, and user behavior to identify anomalies and potential threats that might be missed by traditional security systems.\n*   **Vulnerability Management:** AI can predict how and where you are most likely to be breached so that you can plan for resource and tool allocation toward areas of weakness.\n*   **Phishing Prevention:** AI systems help prevent phishing, malware, and other malicious activities.\n*   **User Authentication:** AI models analyze the risk of each login attempt and verify users through behavioral data, simplifying access for verified users and reducing the cost of fraud.', additional_kwargs={}, response_metadata={}, name='Researcher')]}}
----
{'supervisor': {'next': 'Researcher'}}
----
{'Researcher': {'messages': [HumanMessage(content="AI significantly enhances cybersecurity by improving threat detection, vulnerability management, and incident response. For example, AI algorithms can analyze network traffic and user behavior to identify anomalies indicative of cyberattacks, even those missed by traditional systems. AI can also predict potential breach points, enabling proactive resource allocation and improved cyber resilience. Some specific examples include:\n\n*   **Enhanced Threat Detection:** AI algorithms analyze network traffic, system logs, and user behavior to identify anomalies and potential threats. AI’s predictive capabilities allow organizations to anticipate and prevent cyberattacks before they occur through automated monitoring and analysis.\n*   **Vulnerability Management:** AI autonomously scans systems and networks for weaknesses, streamlining the identification of potential entry points for attackers. It can recommend and prioritize necessary security updates, reducing manual effort and minimizing vulnerability exposure. For example, IBM’s managed security services team used AI to automate 70% of alert closures and speed up their threat management timeline by more than 50% within the first year of use.\n*   **Phishing Prevention:** AI systems help prevent phishing, malware, and other malicious activities by tracking the evolution of malware and phishing techniques and adapting detection models accordingly.\n*   **Automated Incident Response:** AI automates actions to speed up incident response times. For example, AI can automate response processes to certain alerts, such as immediately shutting down a device's network connectivity upon detecting a known malware sample.\n*   **User Authentication:** AI models analyze the risk of each login attempt and verify users through behavioral data, simplifying access for verified users and reducing the cost of fraud.\n\nAI-based cybersecurity systems demonstrate enhanced efficacy. Deep Instinct's Chuck Everette reveals that AI models boast security rates of 80% to 92%, surpassing the 30% to 60% achieved by legacy signature-based malware detection systems.", additional_kwargs={}, response_metadata={}, name='Researcher')]}}
----
{'supervisor': {'next': 'FINISH'}}
----
"""


# for s in graph.stream(
#     {"messages": [HumanMess"""age(content="Write a brief pdf research report on pikas.")]},
#     {"recursion_limit": 100},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")
