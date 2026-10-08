"""
================================================================================
LANGCHAIN MULTI-TASK ReAct AGENT (PARALLEL TOOL CALLING)
================================================================================
Demonstrates executing multiple tasks simultaneously using LangChain/LangGraph
and Google Gemini's parallel tool-calling capabilities.

Key Features:
- Executes multiple tools (calculator, web_search, datetime) in parallel
  within a single turn.
- Streams live Thought -> Actions -> Observations -> Final Answer.
- Framework: Modern LangChain / LangGraph (create_react_agent).
================================================================================
"""

import json
import os
import sys

# Modern LangChain & LangGraph Imports
try:
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent
except Exception as err:
    print(f"\n❌ [IMPORT ERROR]: {type(err).__name__}: {err}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Reuse our existing tool implementations
from calculator import execute_calculator
from web_tool import execute_web_search
from datetime_tool import execute_get_datetime
from document_tool import execute_document_search


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
GEMINI_API_KEY = "AQ.Ab8RN6JtYeJd21sqsCJCbqZPwIhohV0vRew1FLpbz2czkxmDbg"
MODEL_ID = "gemini-3.8-flash"


def get_api_key() -> str:
    """Retrieves API key from script variable, environment, or terminal input."""
    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("YOUR_"):
        return GEMINI_API_KEY.strip()

    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    print("\n🔑 GEMINI API KEY REQUIRED")
    user_key = input("Paste your Gemini API Key: ").strip()
    if user_key:
        return user_key

    print("No API key provided. Exiting.")
    sys.exit(1)


# ------------------------------------------------------------------------------
# DEFINE LANGCHAIN TOOLS
# ------------------------------------------------------------------------------
@tool
def calculator_tool(operation: str, a: float, b: float) -> str:
    """Performs arithmetic calculations ('add', 'subtract', 'multiply', 'divide', 'power') on numbers a and b."""
    result = execute_calculator(operation, a, b)
    return json.dumps(result)


@tool
def web_search_tool(query: str, max_results: int = 3) -> str:
    """Searches the live web for facts, information, news, current statistics, or external knowledge."""
    result = execute_web_search(query, max_results=max_results)
    return json.dumps(result)


@tool
def datetime_tool(timezone: str = "Local") -> str:
    """Returns the current live date, time, day of the week, year, and timezone."""
    tz = None if timezone.lower() == "local" else timezone
    result = execute_get_datetime(tz)
    return json.dumps(result)


@tool
def document_search_tool(query: str, filename: str = "") -> str:
    """Searches local files in the documents/ folder (e.g. company handbooks, policies, notes). Use this for policy, leave, or company questions."""
    target = filename.strip() if filename else None
    result = execute_document_search(query, target)
    return json.dumps(result)


TOOLS = [calculator_tool, web_search_tool, datetime_tool, document_search_tool]


# ------------------------------------------------------------------------------
# BUILD LANGCHAIN MULTI-TASK AGENT
# ------------------------------------------------------------------------------
def build_agent(api_key: str, model_id: str = MODEL_ID):
    """Initializes the Gemini LLM with LangGraph ReAct agent."""
    llm = ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=api_key,
        temperature=0.0,
    )

    system_prompt = (
        "You are an advanced multi-tasking AI Assistant built with LangChain.\n\n"
        "Your available tools:\n"
        "- calculator_tool: Performs numeric calculations.\n"
        "- web_search_tool: Searches the web for facts and information.\n"
        "- datetime_tool: Retrieves the current live date, time, weekday, and year.\n"
        "- document_search_tool: Searches local documents in the documents/ folder (e.g. company policies, employee handbook, notes).\n\n"
        "PARALLEL EXECUTION RULE:\n"
        "When the user asks multiple questions or a multi-part query, YOU MUST CALL ALL "
        "RELEVANT TOOLS IN PARALLEL AT THE SAME TIME in a single step.\n"
        "After receiving all observations, deliver a comprehensive, friendly final answer."
    )

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=system_prompt
    )
    return agent


# ------------------------------------------------------------------------------
# INTERACTIVE CLI
# ------------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("   🦜️🔗 LANGCHAIN MULTI-TASK AGENT (PARALLEL TOOL CALLING) 🦜️🔗")
    print("=" * 70)
    print("Architecture: Modern LangChain / LangGraph ReAct Agent")
    print("Capable of executing multiple tasks at the exact same time!")
    print("Tools: 1. Calculator | 2. Web Search | 3. Date & Time | 4. Doc Search")

    api_key = get_api_key()
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key

    agent = build_agent(api_key)
    conversation_history = []

    print("\n✅ LangChain Agent is ready!")
    print("Commands:")
    print("  - 'reset' : Clears conversation memory")
    print("  - 'quit' or 'exit': Closes the agent")
    print("\n💡 Try asking multiple tasks at once:")
    print("  👉 'What is 45 * 12, what day of the week is it today, and who is the CEO of Google?'\n")

    while True:
        try:
            user_input = input("\n👉 Enter multi-task prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent. Goodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ["quit", "exit", "q"]:
            print("Exiting agent. Goodbye!")
            break
        elif lower == "reset":
            conversation_history = []
            print("🔄 Conversation memory reset.")
            continue

        print("\n" + "─" * 65)
        print("⚡ LANGCHAIN AGENT EXECUTING MULTIPLE TASKS IN PARALLEL...")
        print("─" * 65)

        conversation_history.append(HumanMessage(content=user_input))

        try:
            # Stream the execution to observe parallel tool calling live
            final_content = ""
            for chunk in agent.stream({"messages": conversation_history}, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                print(f"\n🛠️ [LANGCHAIN PARALLEL ACTION]: Invoking {len(msg.tool_calls)} tool(s) simultaneously!")
                                for tc in msg.tool_calls:
                                    print(f"   • Tool: {tc['name']}")
                                    print(f"     Args: {json.dumps(tc['args'])}")
                            if msg.content:
                                final_content = msg.content
                        elif isinstance(msg, ToolMessage):
                            print(f"\n👁️ [OBSERVATION]: {msg.name}")
                            print(f"   Result: {msg.content}")

            if final_content:
                conversation_history.append(AIMessage(content=final_content))
                print("\n" + "─" * 65)
                print("💡 [FINAL SYNTHESIZED ANSWER]:")
                print(final_content)
                print("─" * 65)

        except Exception as err:
            err_str = str(err)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                print(f"\n⚠️  [DAILY QUOTA LIMIT REACHED (20 requests/day)]")
                print("   Your Google Cloud project reached the daily limit on 'gemini-3.8-flash'.")
                print("   Create a key in a new project at https://aistudio.google.com/app/apikey")
            else:
                print(f"\n❌ [ERROR DURING EXECUTION]: {err}")


if __name__ == "__main__":
    main()
