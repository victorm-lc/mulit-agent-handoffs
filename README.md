# Multi-Agent Patterns

This repository demonstrates the two main multi-agent patterns from LangChain's [multi-agent documentation](https://docs.langchain.com/oss/python/langchain/multi-agent.md), implemented in supervisor-style architectures.

> **Key Focus**: Context engineering - controlling what information each agent sees for optimal performance.

## Patterns Implemented

### 1. **Tool Calling Pattern** (`subagents_as_tools.py`)
- **Centralized control**: Controller agent calls other agents as tools
- **Use when**: Need centralized workflow control, structured task orchestration
- Tool agents perform tasks and return results to the controller

### 2. **Handoffs Pattern** - Two Implementations:

#### **Command + Send** (`command_send.py`)
- **Decentralized control**: Agents transfer control using structured routing
- Uses Command + Send objects for flexible state management

#### **Handoff Tools** (`handoff_tools.py`) 
- **Decentralized control**: Agents transfer control using handoff tools
- Tools return Command objects for explicit navigation control

**Use handoffs when**: Want agents to interact directly with users, need complex multi-domain conversations

## Running the Examples

Each file contains a complete graph implementation. Install dependencies:

```bash
uv sync
```

Run the LangGraph dev server to test any pattern:

```bash
langgraph dev
```

## References

- [LangGraph Multi-Agent Patterns](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Agent Supervisor Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
