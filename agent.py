"""
================================================================================
AGENTIC AI CALCULATOR (NO FRAMEWORKS)
================================================================================
A pure implementation of an Agentic AI Tool-Calling system using Google's
official `google-genai` SDK and a JSON-defined tool schema.

How Agentic AI Works Here:
1. JSON Tool Schema: We declare a calculator tool in JSON schema format.
2. Model Reasoning: The LLM reads the user prompt + the JSON tool schema.
3. Tool Call Generation: If math is needed, the LLM outputs a structured
   JSON function call with arguments (operation, a, b).
4. Agent Execution: Our Python code intercepts the LLM's request, executes
   the local calculator function, and packages the result into a JSON response.
5. Observation & Final Answer: The LLM reads the tool output and provides the
   final answer to the user.
================================================================================
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Official Google GenAI SDK (No LangChain, No CrewAI, No heavy wrappers)
from google import genai
from google.genai import types

# Import our local tool logic
from calculator import execute_calculator
from web_tool import execute_web_search
from datetime_tool import execute_get_datetime
from document_tool import execute_document_search


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
# You can paste your Gemini API key directly here, or set the GEMINI_API_KEY
# environment variable.
GEMINI_API_KEY = "AQ.Ab8RN6JtYeJd21sqsCJCbqZPwIhohV0vRew1FLpbz2czkxmDbg"

# Active model required by Google AI Studio
MODEL_ID = "gemini-3.8-flash"

# Paths to the JSON tool definitions
CALCULATOR_SCHEMA_FILE = Path(__file__).parent / "calculator_tool.json"
WEB_SCHEMA_FILE = Path(__file__).parent / "web_tool.json"
DATETIME_SCHEMA_FILE = Path(__file__).parent / "datetime_tool.json"
DOCUMENT_SCHEMA_FILE = Path(__file__).parent / "document_tool.json"


def get_api_key() -> str:
    """Retrieves the API key from variable, environment, or terminal input."""
    # 1. Check hardcoded variable if user replaced the placeholder
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        return GEMINI_API_KEY.strip()

    # 2. Check environment variable
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    # 3. Prompt user interactively
    print("\n" + "=" * 60)
    print("🔑 GEMINI API KEY REQUIRED")
    print("=" * 60)
    print("No API key was detected.")
    print("You can either:")
    print("  1. Edit 'agent.py' and set GEMINI_API_KEY = 'your_key'")
    print("  2. Set the environment variable: set GEMINI_API_KEY=your_key")
    print("  3. Or paste it directly below right now:\n")
    try:
        user_key = input("Paste your Gemini API Key: ").strip()
        if user_key:
            return user_key
    except (KeyboardInterrupt, EOFError):
        pass

    print("\n[ERROR] No API key provided. Exiting.")
    sys.exit(1)


def load_tool_declarations() -> List[Dict[str, Any]]:
    """Loads calculator, web search, datetime, and document search tool JSON schemas from disk."""
    schemas = []
    for filepath in [CALCULATOR_SCHEMA_FILE, WEB_SCHEMA_FILE, DATETIME_SCHEMA_FILE, DOCUMENT_SCHEMA_FILE]:
        if not filepath.exists():
            raise FileNotFoundError(f"Tool schema not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            schemas.append(json.load(f))
    return schemas


def print_json_box(title: str, data: Any, prefix: str = "  "):
    """Nicely formats and prints JSON data with indentation."""
    print(f"\n{prefix}┌─ {title} " + "─" * max(10, 50 - len(title)))
    formatted = json.dumps(data, indent=2)
    for line in formatted.splitlines():
        print(f"{prefix}│ {line}")
    print(f"{prefix}└" + "─" * 52)


import time


def safe_send_message(chat: Any, message: Any, max_retries: int = 3, initial_delay: float = 2.5) -> Any:
    """
    Sends a message with automatic retry on transient Google API errors
    (503 UNAVAILABLE, Deadline expired, 429 Rate Limit, High Demand).
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return chat.send_message(message)
        except Exception as err:
            err_str = str(err)
            # Daily quota limit should fail fast to trigger model fallback
            if "RESOURCE_EXHAUSTED" in err_str or "quota exceeded" in err_str.lower():
                raise err

            is_transient = any(
                keyword in err_str
                for keyword in [
                    "503", "UNAVAILABLE", "Deadline", "expired",
                    "timeout", "high demand", "demand"
                ]
            )
            if is_transient and attempt < max_retries:
                print(f"\n⚠️  [HIGH DEMAND / 503]: Google server is busy. Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
            else:
                raise err


def process_agent_turn(chat: Any, user_prompt: str, max_iterations: int = 5):
    """
    Executes one turn of the Agentic ReAct (Reason + Act) loop:
    
    1. THOUGHT (Reasoning) : The LLM reasons about what tool to use.
    2. ACTION (Acting)      : The LLM issues a structured tool call in JSON.
    3. OBSERVATION          : Our agent executes the tool and captures the output.
    4. REPEAT               : LLM observes the result, reasons again, and acts until done.
    5. FINAL ANSWER         : LLM delivers the final synthesized response.
    """
    print(f"\n💬 [USER]: {user_prompt}")
    print("\n" + "─" * 60)
    print("🤖 [ReAct AGENT LOOP INITIALIZED]")
    print("─" * 60)
    
    # 1. Send the user's message to the Gemini chat session
    response = safe_send_message(chat, user_prompt)
    
    iteration = 0
    while response.function_calls and iteration < max_iterations:
        iteration += 1
        print(f"\n📍 ─── STEP {iteration} ───")

        # Extract any Thought / Reasoning text generated by the model before the action
        has_thought = False
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text and part.text.strip():
                    print(f"\n🧠 [THOUGHT / REASON]:")
                    print(f"   {part.text.strip()}")
                    has_thought = True

        if not has_thought:
            print(f"\n🧠 [THOUGHT / REASON]:")
            print(f"   Evaluating query and determining appropriate tool to invoke...")

        # 2. Extract and display the ACTION (Tool Call)
        tool_responses: List[types.Part] = []
        for call in response.function_calls:
            tool_name = call.name
            tool_args = dict(call.args) if call.args else {}

            print(f"\n🛠️ [ACTION (Tool Call)]:")
            print(f"   Tool Name : {tool_name}")
            print_json_box(f"Action Input for '{tool_name}' (JSON)", tool_args, prefix="   ")

            # 3. Execute the local tool with the LLM's arguments
            if tool_name == "calculator":
                tool_result = execute_calculator(
                    operation=tool_args.get("operation", ""),
                    a=tool_args.get("a", 0),
                    b=tool_args.get("b", 0)
                )
            elif tool_name == "web_search":
                tool_result = execute_web_search(
                    query=tool_args.get("query", ""),
                    max_results=tool_args.get("max_results", 3)
                )
            elif tool_name == "get_current_datetime":
                tool_result = execute_get_datetime(
                    timezone=tool_args.get("timezone", None)
                )
            elif tool_name == "document_search":
                tool_result = execute_document_search(
                    query=tool_args.get("query", ""),
                    filename=tool_args.get("filename", None)
                )
            else:
                tool_result = {
                    "status": "error",
                    "error": f"Unknown tool '{tool_name}' requested by LLM."
                }

            # Display the OBSERVATION (Tool Output)
            print(f"\n👁️ [OBSERVATION (Tool Output)]:")
            print_json_box(f"Observation for '{tool_name}' (JSON)", tool_result, prefix="   ")

            # Wrap the result into a Gemini FunctionResponse part
            part = types.Part.from_function_response(
                name=tool_name,
                response=tool_result
            )
            tool_responses.append(part)

        # Send tool results back to the LLM so it can reason over the observation
        print(f"\n🔄 [FEEDING OBSERVATION TO LLM FOR NEXT REASONING STEP]...")
        response = safe_send_message(chat, tool_responses)

    # 4. Extract any final Thought and the Final Answer
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.text and part.text.strip():
                print(f"\n💡 [FINAL ANSWER]:")
                print(f"{part.text.strip()}")
                return

    if response.text:
        print(f"\n💡 [FINAL ANSWER]:")
        print(response.text)
    else:
        print("\n⚠️ [AGENT]: Finished without text response.")


def main():
    print("=" * 70)
    print(" 🤖 ReAct AGENT: CALC + WEB + DATE/TIME + DOCUMENT SEARCH 🤖")
    print("=" * 70)
    print("Architecture: Pure ReAct (Reasoning -> Acting -> Observation -> Answer)")
    print("Tools Available: 1. Calculator | 2. Web | 3. Date & Time | 4. Doc Search")
    print("Frameworks used: None (Pure Python + official google-genai SDK).")

    # Retrieve API key
    api_key = get_api_key()
    
    # Initialize the official Gemini Client
    client = genai.Client(api_key=api_key)

    # Load tool JSON schemas
    tool_schemas = load_tool_declarations()
    print("\n📋 [INITIALIZATION] Loaded Tool JSON Schemas:")
    for schema in tool_schemas:
        print_json_box(f"{schema.get('name')} Schema", schema)

    # Wrap the JSON declarations into Gemini Tools
    tools = [types.Tool(function_declarations=tool_schemas)]

    # ReAct Agent system instruction
    system_instruction = (
        "You are an advanced AI Agent that operates using the ReAct (Reasoning + Acting) framework.\n\n"
        "Your available tools:\n"
        "1. 'calculator': Performs arithmetic calculations ('add', 'subtract', 'multiply', 'divide', 'power'). "
        "Never calculate math mentally; ALWAYS call 'calculator'.\n"
        "2. 'web_search': Searches the live web for facts, information, news, current statistics, or external knowledge.\n"
        "3. 'get_current_datetime': Returns current live date, time, weekday, year, and timezone.\n"
        "4. 'document_search': Searches local documents in the documents/ folder (e.g. company handbooks, policies, notes). "
        "ALWAYS use 'document_search' when asked about company policies, leave, office budgets, internal notes, or local files.\n\n"
        "STRICT ReAct WORKFLOW INSTRUCTIONS:\n"
        "Whenever given a user query, you MUST work in steps:\n"
        "- STEP 1 (THOUGHT): First, explicitly state your Thought/Reasoning. Explain what the user is asking, what information you need, and which tool you plan to call.\n"
        "- STEP 2 (ACTION): Call the tool with the appropriate JSON arguments.\n"
        "- STEP 3 (OBSERVATION): When you receive the tool's output, analyze it.\n"
        "- STEP 4 (NEXT THOUGHT/ACTION): If additional steps are required (such as doing math with information found from search, docs, or date), state your new Thought and call the next tool.\n"
        "- STEP 5 (FINAL ANSWER): Once all needed actions are complete, synthesize your final comprehensive answer for the user."
    )

    # Configure the model
    config = types.GenerateContentConfig(
        tools=tools,
        temperature=0.0,  # Deterministic output for tool calling
        system_instruction=system_instruction
    )

    current_model = MODEL_ID

    def create_chat_session(model_name=None):
        m = model_name or current_model
        return client.chats.create(model=m, config=config)

    # Create active chat session
    chat = create_chat_session()

    print("\n✅ Multi-Tool Agent is ready!")
    print(f"Active Model: {current_model}")
    print("Commands:")
    print("  - Ask any math or web questions (e.g. '5 * 5', 'Who won the 2024 ICC World Cup?', 'Search distance to moon and divide by 2')")
    print("  - 'model <name>' : Switch model (e.g. 'model gemini-2.0-flash' or 'model gemini-3.8-flash')")
    print("  - 'reset' : Clears conversation history and starts fresh")
    print("  - 'schema': Re-prints the JSON tool schemas")
    print("  - 'quit' or 'exit': Closes the agent\n")

    # Interactive loop
    while True:
        try:
            user_input = input("\n👉 Enter calculation prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent. Goodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ["quit", "exit", "q"]:
            print("Exiting agent. Goodbye!")
            break
        elif lower.startswith("model"):
            parts = user_input.split()
            if len(parts) > 1:
                current_model = parts[1].strip()
                try:
                    chat = create_chat_session(current_model)
                    print(f"🔄 Switched active model to: {current_model}")
                except Exception as ex:
                    print(f"❌ Failed to switch model: {ex}")
            else:
                print(f"Current model: {current_model}. Use 'model <name>' to switch (e.g. 'model gemini-2.0-flash').")
            continue
        elif lower.startswith("key"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                new_key = parts[1].strip()
                try:
                    client = genai.Client(api_key=new_key)
                    chat = create_chat_session(current_model)
                    print("\n🔑 Successfully updated Gemini API key and re-initialized session!")
                except Exception as k_err:
                    print(f"❌ Failed to initialize with new key: {k_err}")
            else:
                print("Usage: key <your_new_gemini_api_key>")
            continue
        elif lower == "reset":
            chat = create_chat_session()
            print("🔄 Conversation memory reset to fresh state.")
            continue
        elif lower == "schema":
            for schema in tool_schemas:
                print_json_box(f"{schema.get('name')} Schema", schema)
            continue
        elif lower == "help":
            print("\nTry asking:")
            print("  • Math (Calculator) : 'What is 12345 multiplied by 678?'")
            print("  • Web (Search)      : 'Who won the 2024 ICC T20 World Cup?'")
            print("  • Date & Time       : 'What is today\'s date and current time?'")
            print("  • Doc Search        : 'What is the annual paid leave policy in the company handbook?'")
            print("  • Doc Search        : 'What is the home office hardware allowance?'")
            print("  • Chaining Multi    : 'Search annual leave days from handbook, and multiply by 8 to get total hours'")
            continue

        try:
            process_agent_turn(chat, user_input)
        except Exception as err:
            err_str = str(err)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                print(f"\n⚠️  [DAILY QUOTA LIMIT REACHED (20 requests/day)]")
                print("   Google AI Studio allows 20 free requests per day per project on 'gemini-3.8-flash'.")
                print("   To continue immediately without waiting:")
                print("   1. Go to: https://aistudio.google.com/app/apikey")
                print("   2. Click 'Create API key' -> 'Create API key in new project'")
                print("   3. Type here: key <your_new_api_key> (or update line 41 in agent.py)")
            else:
                print(f"\n❌ [ERROR DURING EXECUTION]: {err}")


if __name__ == "__main__":
    main()
