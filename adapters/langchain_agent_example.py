"""
LangChain AgentExecutor Example (Demo Only)
============================================
Demonstrates using the VerifiedSQLTool within a LangChain AgentExecutor.

This is a DEMO FILE showing interoperability - not production code.
The core verification logic remains in nl_to_sql_agent.py.

Requirements:
    pip install langchain langchain-core langchain-openai

Usage:
    # Set OPENAI_API_KEY environment variable, then:
    python -m adapters.langchain_agent_example
"""

import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_core_agent_demo():
    """
    Fallback demo when LangChain is not installed.
    Shows the core agent works independently.
    """
    from nl_to_sql_agent import NLToSQLAgent, MockLLM

    mock_responses = {
        "customers": ["SELECT * FROM customers"],
        "orders": ["SELECT * FROM orders WHERE status = 'pending'"],
    }
    mock_llm = MockLLM(responses=mock_responses)
    agent = NLToSQLAgent(llm=mock_llm, max_retries=2)

    print("--- Core Agent Demo (No LangChain) ---")
    result = agent.process("Show me all customers")
    print(f"Success: {result.success}")
    print(f"SQL: {result.sql}")
    print(f"Attempts: {result.attempts}")

    print("\nThe core agent works without LangChain.")
    print("LangChain adapters are optional wrappers only.")


def run_demo_with_mock():
    """
    Demo using MockLLM (no API keys required).
    Shows the integration pattern without external dependencies.
    """
    print("=" * 60)
    print("LangChain Integration Demo (Mock LLM)")
    print("=" * 60)

    # Check if LangChain is available
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        print("\nLangChain is not installed. Install with:")
        print("  pip install langchain-core")
        print("\nRunning core agent demo instead...\n")
        run_core_agent_demo()
        return

    from nl_to_sql_agent import NLToSQLAgent, MockLLM
    from adapters.langchain_tool import VerifiedSQLTool
    from adapters.langchain_runnable import VerifiedSQLRunnable

    # Configure mock responses
    mock_responses = {
        "customers": ["SELECT * FROM customers"],
        "orders": ["SELECT * FROM orders WHERE status = 'pending'"],
        "revenue": ["SELECT SUM(amount) FROM orders"],
    }
    mock_llm = MockLLM(responses=mock_responses)
    agent = NLToSQLAgent(llm=mock_llm, max_retries=2)

    # --- Demo 1: Tool Adapter ---
    print("\n--- Tool Adapter Demo ---")
    tool = VerifiedSQLTool(agent=agent, include_audit=True)

    print(f"Tool name: {tool.name}")
    print(f"Tool description: {tool.description[:80]}...")

    result = tool.invoke({"query": "Show me all customers"})
    print(f"Result: {result}")

    # --- Demo 2: Runnable Adapter ---
    print("\n--- Runnable (LCEL) Adapter Demo ---")

    # Reset agent for fresh demo
    agent2 = NLToSQLAgent(llm=MockLLM(responses=mock_responses), max_retries=2)
    runnable = VerifiedSQLRunnable(agent=agent2, output_sql_only=False)

    # Direct invocation
    result = runnable.invoke("What is the total revenue?")
    print(f"Success: {result.success}")
    print(f"SQL: {result.sql}")
    print(f"Attempts: {result.attempts}")

    # --- Demo 3: Streaming Audit Trail ---
    print("\n--- Streaming Demo ---")
    agent3 = NLToSQLAgent(llm=MockLLM(responses=mock_responses), max_retries=2)
    runnable_stream = VerifiedSQLRunnable(agent=agent3)

    print("Streaming verification steps:")
    for chunk in runnable_stream.stream("Show pending orders"):
        if chunk["type"] == "audit_entry" and chunk.get("verifications"):
            for v in chunk["verifications"]:
                status = "✓" if v["status"] == "passed" else "✗"
                print(f"  {status} {v['name']}: {v['message']}")
        elif chunk["type"] == "final_result":
            print(f"  → Final: {chunk['sql']}")


def run_demo_with_openai():
    """
    Demo using real OpenAI LLM (requires OPENAI_API_KEY).
    Uncomment and run if you have API access.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        print("OpenAI demo requires: pip install langchain langchain-openai")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable for OpenAI demo")
        return

    print("\n" + "=" * 60)
    print("LangChain AgentExecutor Demo (OpenAI)")
    print("=" * 60)

    from nl_to_sql_agent import NLToSQLAgent, LLMInterface, LLMResponse

    # Create a real LLM adapter
    class OpenAIAdapter(LLMInterface):
        def __init__(self):
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
            messages = []
            if system_prompt:
                messages.append(("system", system_prompt))
            messages.append(("human", prompt))
            response = self.llm.invoke(messages)
            return LLMResponse(content=response.content, model="gpt-4o-mini")

    # Create verified SQL agent with real LLM
    openai_adapter = OpenAIAdapter()
    sql_agent = NLToSQLAgent(llm=openai_adapter, max_retries=2)

    # Wrap as LangChain tool
    from adapters.langchain_tool import VerifiedSQLTool
    sql_tool = VerifiedSQLTool(agent=sql_agent, include_audit=True)

    # Create a simple agent with the tool
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the verified_sql_generator "
                   "tool to convert natural language questions into SQL queries."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the agent
    agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_tool_calling_agent(agent_llm, [sql_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[sql_tool], verbose=True)

    # Run a query
    result = agent_executor.invoke({
        "input": "Generate SQL to find the top 5 customers by order amount"
    })
    print(f"\nFinal output: {result['output']}")

    # Show audit trail
    print("\nAudit trail from verification:")
    for entry in sql_tool.get_last_audit_trail():
        if entry.verification_results:
            for vr in entry.verification_results:
                status = "✓" if vr.status.value == "passed" else "✗"
                print(f"  {status} {vr.verifier_name}")


if __name__ == "__main__":
    # Always run mock demo (no dependencies)
    run_demo_with_mock()

    # Optionally run OpenAI demo if configured
    if os.getenv("OPENAI_API_KEY"):
        run_demo_with_openai()
    else:
        print("\n(Set OPENAI_API_KEY to run the OpenAI AgentExecutor demo)")
