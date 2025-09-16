# Multi-Agent Handoff Patterns

This repository demonstrates different patterns for implementing multi-agent handoffs in LangGraph, based on the [LangGraph multi-agent documentation](https://docs.langchain.com/oss/python/langchain/multi-agent).

## Patterns Implemented

### 1. **Subagents as Tools** (`subagents_as_tools.py`)

### 2. **Handoffs using Handoff tools (Command-based)** (`handoff_tools.py`)

### 3. **Command + Send Pattern** (`command_send.py`)

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
