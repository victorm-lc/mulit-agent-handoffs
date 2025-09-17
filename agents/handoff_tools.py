"""
HANDOFFS PATTERN - TOOLS IMPLEMENTATION (LangChain Multi-Agent)

This implements the "Handoffs" pattern from LangChain's multi-agent documentation:
https://docs.langchain.com/oss/python/langchain/multi-agent.md#handoffs

In handoffs, agents can directly pass control to each other. The "active" agent changes, 
and the user interacts with whichever agent currently has control.

Flow:
1. The current agent decides it needs help from another agent
2. It passes control (and state) to the next agent  
3. The new agent interacts directly with the user until it decides to hand off again or finish

This specific implementation uses "handoffs as tools" - a LangGraph pattern where:
- Handoff tools return Command objects that specify navigation to different agent nodes
- Tools can update state and navigate simultaneously 
- Each agent exists as a separate node in the graph (not just tool calls)

Key characteristics:
- Decentralized control: agents can change who is active
- Agents can interact directly with the user
- Complex, human-like conversation between specialists
- Tools return Command objects instead of simple strings
- Command objects specify destination nodes and state updates

Benefits:
- Agents interact directly with users
- Support for complex, multi-domain conversations  
- Specialist takeover capabilities
- More flexible routing than simple tool returns
- Native LangGraph handoff pattern with explicit control

When to use (from LangChain docs):
- Need centralized control over workflow? ❌ No
- Want agents to interact directly with the user? ✅ Yes
- Complex, human-like conversation between specialists? ✅ Strong

Perfect for: Multi-domain conversations, specialist takeover, explicit agent navigation control
Note: You can mix patterns - use handoffs for agent switching, and have each agent call subagents as tools.
"""

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from agents.invoice_agent import graph as invoice_information_subagent
from agents.music_agent import graph as music_catalog_subagent
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated, NotRequired
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class State(InputState):
    customer_id: NotRequired[str]  # Optional context for agents
    loaded_memory: NotRequired[str]  # Optional memory state
    remaining_steps: NotRequired[RemainingSteps]  # Execution control

model = ChatOpenAI(model="o3-mini")

# HANDOFFS PATTERN - TOOLS IMPLEMENTATION  
# From LangChain docs: "Agents can directly pass control to each other. The 'active' agent changes,
# and the user interacts with whichever agent currently has control."
# These tools demonstrate the "handoffs as tools" pattern where tools return Command objects
# instead of simple strings. This enables decentralized control and agent-to-agent navigation.

@tool("transfer-to-invoice-agent")
def transfer_to_invoice_agent(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    reason: str = "Invoice-related inquiry",
    context: str = "Context for the invoice agent"
) -> Command:
    """Transfer control to the invoice agent for invoice-related questions.
    
    HANDOFFS PATTERN: This demonstrates how the current agent passes control to another agent.
    From LangChain docs: "The current agent decides to transfer control to another agent.
    The active agent changes, and the user may continue interacting directly with the new agent."
    
    Flow:
    1. Current agent (via tool) decides it needs help from invoice agent
    2. Tool receives current state and creates handoff representation  
    3. Returns Command object that passes control (and state) to target agent
    4. Invoice agent becomes active and can interact directly with user
    
    Args:
        state: Current graph state (injected automatically)
        tool_call_id: Tool call identifier (injected automatically)
        reason: Reason for the transfer
        context: Context information for the target agent
    """
    agent_name = "invoice_information_subagent"
    
    # Create a ToolMessage to represent the handoff in the conversation history
    # This follows LangGraph docs recommendation to add handoff representation
    tool_message = ToolMessage(
        content=f"Successfully transferred to {agent_name}. Reason: {reason}. Context: {context}",
        name="transfer-to-invoice-agent",
        tool_call_id=tool_call_id,
    )
    
    # Return Command object that specifies:
    # - goto: which node to navigate to
    # - update: how to update the state (add the handoff message)
    return Command(goto=agent_name, update={"messages": state["messages"] + [tool_message]})

@tool("transfer-to-music-catalog-agent")
def transfer_to_music_catalog_agent(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    reason: str = "Music catalog inquiry",
    context: str = "Context for the music catalog agent"
) -> Command:
    """Transfer control to the music catalog agent for music-related questions.
    
    Same pattern as invoice transfer but targeting the music catalog agent.
    Demonstrates how multiple handoff tools can coexist in the same graph.
    """
    agent_name = "music_catalog_subagent"
    
    tool_message = ToolMessage(
        content=f"Successfully transferred to {agent_name}. Reason: {reason}. Context: {context}",
        name="transfer-to-music-catalog-agent",
        tool_call_id=tool_call_id,
    )

    return Command(goto=agent_name, update={"messages": state["messages"] + [tool_message]})

# SUPERVISOR SETUP
tools = [transfer_to_invoice_agent, transfer_to_music_catalog_agent]

supervisor_prompt = """You are an expert customer support assistant for a digital music store. You can handle music catalog or invoice related questions regarding past purchases, song or album availabilities.

Your primary role is to serve as a supervisor for this multi-agent team that helps answer queries from customers.

Your team is composed of two subagents:
1. music_catalog_subagent: Has access to user's saved music preferences and can retrieve information about the digital music store's music catalog (albums, tracks, songs, etc.) from the database.
2. invoice_information_subagent: Can retrieve information about a customer's past purchases or invoices from the database.

DECISION LOGIC:
- If the user's question needs specialist help AND no subagent has responded yet, use the appropriate handoff tool
- If a subagent has already provided a response, DO NOT call tools again - instead, synthesize and present their response to the user in a helpful, conversational way
- If the question is unrelated to music or invoices, respond directly without using tools

IMPORTANT: After a subagent completes their task, if there's no more help needed from your team, your job is to act as the final interface to the user - synthesize the subagent's response and present it conversationally, don't just add generic follow-ups."""

supervisor = create_react_agent(model, tools=tools, prompt=supervisor_prompt, state_schema=State, name="supervisor")

# GRAPH CONSTRUCTION
# This pattern requires adding the subagent graphs as separate nodes
supervisor_workflow = StateGraph(State)

# Add all nodes: supervisor, and the actual subagent graphs
supervisor_workflow.add_node("supervisor", supervisor, destinations=["music_catalog_subagent", "invoice_information_subagent"])
supervisor_workflow.add_node("music_catalog_subagent", music_catalog_subagent)  # Actual subagent graph
supervisor_workflow.add_node("invoice_information_subagent", invoice_information_subagent)  # Actual subagent graph

supervisor_workflow.add_edge(START, "supervisor")
supervisor_workflow.add_edge("music_catalog_subagent", "supervisor")  # Subagents return to supervisor
supervisor_workflow.add_edge("invoice_information_subagent", "supervisor")

graph = supervisor_workflow.compile(name="supervisor")