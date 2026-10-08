# ReAct AI Agent (Reason + Act)

A pure, framework-free implementation of an **Agentic AI ReAct (Reasoning + Acting) Agent** using Google's official `google-genai` SDK and Python.

---

## 🎯 How the ReAct Pattern Works Here

The agent executes the formal **Thought → Action → Observation → Answer** cycle:

1. 🧠 **THOUGHT (Reasoning)**: The LLM explicitly explains its thought process before executing any tool. It reasons about what information it has, what it still needs, and which tool to call.
2. 🛠️ **ACTION (Tool Invocation in JSON)**: The LLM outputs a structured JSON function call (e.g. `calculator`, `web_search`, or `get_current_datetime`).
3. 👁️ **OBSERVATION (Tool Output)**: Python executes the local tool and feeds the structured JSON output back to the model as an observation.
4. 🔄 **MULTI-STEP CHAINING**: If a task requires multiple steps (e.g. searching the web for data, then doing math with that data), the LLM loops through another Thought → Action → Observation step.
5. 💡 **FINAL ANSWER**: Once all observations are gathered, the LLM synthesizes the final comprehensive answer.

---

## 📂 Project Structure

```
calculator_agent/
├── documents/                 # 📂 Local documents folder
│   ├── students.csv           # 🎓 25 Student Database (from PDF)
│   ├── student_database.md    # 🎓 Markdown semantic chunks of students
│   ├── company_handbook.txt
│   └── ai_notes.md
├── student_rag.py             # 📚 Dedicated Student Database RAG System
├── calculator_tool.json       # Calculator tool JSON schema
├── calculator.py              # Python math calculation logic
├── web_tool.json              # Web Search tool JSON schema
├── web_tool.py                # Live web search logic (DuckDuckGo + Wikipedia)
├── datetime_tool.json         # Date & Time tool JSON schema
├── datetime_tool.py           # Live date, time, weekday, and timezone logic
├── document_tool.json         # Document Search tool JSON schema
├── document_tool.py           # Local file & paragraph relevance search logic
├── agent.py                   # Pure Python ReAct Agent (No Frameworks)
├── langchain_agent.py         # 🦜️🔗 LangChain Multi-Task Agent (Parallel Tool Calling)
├── requirements.txt           # Dependencies for LangChain
└── README.md                  # Instructions and walkthrough
```

---

## 📚 Dedicated Student Database RAG System (`student_rag.py`)

Implements the complete Retrieval-Augmented Generation (RAG) pipeline for the **25 Dummy Student Database**:

### Run RAG QA:
```powershell
python student_rag.py
```
You can type numbers `1` to `10` to immediately test all 10 suggested questions from the PDF, or ask any custom question!

---

## 🛠️ Available Tools (4 Tools)

1. **`calculator`**:
   - Operations: `add`, `subtract`, `multiply`, `divide`, `power`
   - Triggered for: Any numeric arithmetic calculation.
2. **`web_search`**:
   - Arguments: `query`, `max_results`
   - Triggered for: Real-time facts, current news, sports results, and external data.
3. **`get_current_datetime`**:
   - Arguments: `timezone` (optional, e.g. `'UTC'`, `'Asia/Kolkata'`)
   - Triggered for: Current date, time, day of the week, year, and timezone.
4. **`document_search`**:
   - Arguments: `query`, `filename` (optional)
   - Triggered for: Internal policies, company handbook, notes, and local files in `documents/`.

---

## 💡 Example Queries to Test

- **Document Search**:
  > *"What is the annual paid vacation policy in the employee handbook?"*
  > *"What is the home office hardware allowance budget?"*

- **Multi-Tool Chaining (Doc Search + Calculator)**:
  > *"Search the annual vacation days from the handbook, and multiply by 8 to get total leave hours."*

- **LangChain Parallel Execution (4 Tools in 1 Turn)**:
  > *"What is 25 * 14, what day of the week is it, who is the CEO of Google, and what is our annual vacation leave?"*

---

## 💡 Example Queries to Test

- **Math (Calculator)**:
  > *"What is 1234 multiplied by 56?"*
  > *"Calculate 2 to the power of 12"*

- **Web Search**:
  > *"Who won the latest ICC T20 World Cup?"*
  > *"What is the capital of Australia and what is its population?"*

- **Date & Time**:
  > *"What is today's date and what time is it?"*
  > *"What day of the week is it today?"*
  > *"What is the current time in Tokyo vs London?"*

- **Multi-Tool Chaining (Combining Tools)**:
  > *"What year is it right now, and calculate how many years are left until 2050?"*
  > *"Look up the height of Mount Everest and Mount Kilimanjaro, and calculate the difference"*

---

## 🛠️ In-Chat Commands

- `reset` — Clears conversation memory to start a fresh calculation session.
- `schema` — Prints the raw JSON schema of the calculator tool.
- `help` — Displays prompt suggestions.
- `quit` or `exit` — Closes the agent.
