"""
SUBAGENTS AS TOOLS PATTERN (ASYNC/PARALLEL)

This pattern implements the "Supervisor (tool-calling)" architecture from LangGraph docs
with async execution to enable parallel subagent processing.

Key characteristics:
- Subagents are wrapped as @tool decorated async functions
- Supervisor uses standard tool-calling to invoke subagents
- Multiple subagents can execute in parallel when called simultaneously
- Subagents return results as strings after async execution
- Simple, clean handoff pattern with parallel processing capability

Benefits:
- Easy to understand and implement
- Standard LangGraph tool execution flow
- Async execution enables parallel processing of multiple subagents
- Maintains predictable behavior while improving performance
- Good for supervisor-worker patterns where subagents complete discrete tasks

When to use:
- When you want simple handoffs with parallel execution capability
- When subagents should complete their work and return control
- When you prefer standard tool-calling patterns over Command objects
- When you need multiple subagents to work simultaneously
- For supervisor-worker architectures that benefit from parallelism
"""

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from agents.invoice_agent import graph as invoice_subagent_graph
from agents.music_agent import graph as music_subagent_graph
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

# Standard LangGraph state schema - uses 'messages' field for compatibility with built-in tools
class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Standard message list for tool execution
    customer_id: int  # Shared context that can be injected into tools
    remaining_steps: RemainingSteps  # Standard LangGraph execution control

model = ChatOpenAI(model="gpt-4o")

# ASYNC SUBAGENT TOOL DEFINITIONS
# These functions are decorated with @tool and use async execution to enable parallel processing.
# When multiple tools are called simultaneously, they can execute in parallel.
@tool
async def invoice_agent(task: str, customer_id: Annotated[int, InjectedState("customer_id")]):
    """Handle invoice-related queries by processing customer requests about past purchases, billing information, and invoice details.
    
    This tool demonstrates async execution for parallel processing:
    1. Use InjectedState to access shared context (customer_id)
    2. Invoke subagent graph asynchronously with ainvoke()
    3. Can run in parallel with other async tools
    4. Extract and return the final response as a string
    
    Args:
        task: The specific invoice-related task or question to handle
        customer_id: Automatically injected from the supervisor's state
    """
    # Create a focused prompt for the invoice agent with the specific task
    invoice_message = f"""You are an invoice specialist. Handle this customer request: {task}
    
    Focus on providing accurate information about invoices, billing, and purchase history."""
    
    # Invoke the subagent graph asynchronously - this enables parallel execution
    # The subagent will execute completely and return its final state
    input = {"messages": [HumanMessage(content=invoice_message)], "customer_id": customer_id}
    invocation = await invoice_subagent_graph.ainvoke(input)
    
    # Extract the final response content and return as string
    # This becomes the tool's response in the supervisor's conversation
    response = invocation["messages"][-1].content
    return response

@tool
async def music_catalog_agent(task: str):
    """Handle music catalog queries by processing customer requests about songs, albums, artists, and music recommendations.
    
    This tool demonstrates async execution for parallel processing:
    1. No additional state injection needed
    2. Subagent is invoked asynchronously with ainvoke()
    3. Can run in parallel with other async tools
    4. Result is returned as string
    
    Args:
        task: The specific music catalog task or question to handle
    """
    # Create a focused prompt for the music catalog agent with the specific task
    music_prompt = f"""You are a music catalog specialist. Handle this customer request: {task}
    
    Focus on providing information about songs, albums, artists, and music recommendations from our catalog."""
    
    # Invoke the subagent graph asynchronously - enables parallel execution
    input = {"messages": [HumanMessage(content=music_prompt)]}
    invocation = await music_subagent_graph.ainvoke(input)
    
    # Return the subagent's final response
    response = invocation["messages"][-1].content
    return response

# SUPERVISOR SETUP
# Standard tool-calling pattern using LangGraph's built-in components
tools = [invoice_agent, music_catalog_agent]
supervisor_model = model.bind_tools(tools)  # Bind tools to the LLM for tool-calling
tool_node = ToolNode(tools)  # Built-in node that executes tools and handles responses

supervisor_prompt = """You are an expert customer support assistant for a digital music store. You can handle music catalog or invoice related questions regarding past purchases, song or album availabilities. 
Your primary role is to serve as a supervisor/planner for this multi-agent team that helps answer queries from customers.

You have access to two specialist tools that can execute in parallel:
1. invoice_agent: Use this for questions about past purchases, billing information, invoice details, or payment history
2. music_catalog_agent: Use this for questions about songs, albums, artists, music recommendations, or catalog availability

When using these tools:
- Pass the specific task/question as the 'task' parameter
- Be clear and specific about what you want the specialist to handle
- You can break down complex questions into multiple tool calls if needed
- IMPORTANT: If a question involves both music and invoice aspects, call BOTH tools simultaneously - they will execute in parallel for faster response
- For example, if asked "What music did I buy last month?", call both invoice_agent and music_catalog_agent at the same time

If a question is unrelated to music or invoices, answer it directly without using the specialist tools.
"""

async def supervisor(state: SupervisorState, config: RunnableConfig):
    """
    Async supervisor node that decides whether to call tools or end the conversation.
    
    Uses async LangGraph tool-calling flow for parallel execution:
    1. LLM decides whether to call tools based on the user's request
    2. If multiple tools are called, they execute in parallel (thanks to async)
    3. After tools execute, control returns to supervisor
    4. Process repeats until no more tools are needed
    """
    response = await supervisor_model.ainvoke([SystemMessage(content=supervisor_prompt)] + state["messages"])
    return {"messages": [response]}

# GRAPH CONSTRUCTION
# Async supervisor + tool-calling pattern with parallel execution capability
supervisor_workflow = StateGraph(SupervisorState)

# Add the async supervisor and tool execution nodes
supervisor_workflow.add_node("supervisor", supervisor)
supervisor_workflow.add_node("tool_node", tool_node)  # ToolNode automatically handles async tools

# Define the flow:
# 1. Start with supervisor
# 2. Supervisor decides whether to call tools or end
# 3. If multiple tools are called, they execute in parallel (async)
# 4. After all tools complete, return to supervisor
supervisor_workflow.add_edge(START, "supervisor")
supervisor_workflow.add_edge("tool_node", "supervisor")  # Return to supervisor after tool execution
supervisor_workflow.add_conditional_edges("supervisor", tools_condition, {
    "tools": "tool_node",  # Route to tools if supervisor makes tool calls
    "__end__": "__end__",  # End if no tool calls
})

# Compile the graph - LangGraph automatically handles async execution
graph = supervisor_workflow.compile(name="supervisor")