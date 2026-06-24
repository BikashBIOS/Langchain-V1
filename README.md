# 🦜🔗 LangChain V1 — A Hands-On Learning Repository

> A structured, notebook-driven guide to mastering **LangChain V1** — covering agents, multi-provider model integration, tools, messages, structured output, and middleware.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Setup & Installation](#setup--installation)
- [Module Breakdown](#module-breakdown)
  - [1. LangChain Introduction & Agents](#1-langchain-introduction--agents)
  - [2. Multi-Provider Model Integration](#2-multi-provider-model-integration)
  - [3. Tools & Tool Execution Loops](#3-tools--tool-execution-loops)
  - [4. Messages & Conversation Context](#4-messages--conversation-context)
  - [5. Structured Output](#5-structured-output)
  - [6. Middleware](#6-middleware)
- [Key Concepts Covered](#key-concepts-covered)
- [Architecture & Flow](#architecture--flow)
- [Environment Variables](#environment-variables)

---

## Overview

This repository is a comprehensive, practical introduction to **LangChain V1**, the latest major version of the LangChain framework for building LLM-powered applications and agents. Each notebook in this repo corresponds to a distinct concept, progressing from foundational agent creation to advanced patterns like middleware-controlled agent pipelines.

The project is built for learners, developers, and practitioners who want to understand how LangChain V1's new primitives — `create_agent`, `init_chat_model`, `with_structured_output`, and `middleware` — work together to build robust AI applications.

---

## Repository Structure

```
Langchain-V1/
│
├── 1-langchainintro.ipynb       # LangChain V1 agents and basic invocation
├── 2-modelintegration.ipynb     # OpenAI, Google Gemini, and GROQ model integration
├── 3-tools.ipynb                # Tool definition, binding, and execution loops
├── 4-messages.ipynb             # Message types, conversation history, and context
├── 5-structuredoutput.ipynb     # Pydantic, TypedDict, and Dataclass structured output
├── 6-middleware.ipynb           # Summarization and Human-in-the-Loop middleware
├── main.py                      # Entry point placeholder
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Tech Stack & Dependencies

| Library | Purpose |
|---|---|
| `langchain` | Core framework — agents, chains, messages, tools |
| `langchain-openai` | OpenAI GPT model integration |
| `langchain-groq` | GROQ model integration (Qwen, LLaMA, etc.) |
| `langchain-google-genai` | Google Gemini model integration |
| `langchain_community` | Community integrations and utilities |
| `langgraph` | Agent state graph, checkpointing, and middleware runtime |
| `pydantic` | Data validation and structured output schemas |
| `python-dotenv` | API key management via `.env` files |

**Python version:** 3.13+

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/BikashBIOS/Langchain-V1.git
cd Langchain-V1
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure your API keys** — create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

**5. Launch Jupyter**
```bash
jupyter notebook
```

---

## Module Breakdown

### 1. LangChain Introduction & Agents
**File:** `1-langchainintro.ipynb`

The entry point to the repository. Introduces LangChain V1's most important new primitive: `create_agent` — a high-level, opinionated way to construct agents with tools, system prompts, and model selection in a single call.

**What's covered:**
- Verifying the LangChain V1 version
- Defining a Python function as a tool (`get_weather`)
- Creating a fully functional agent using `create_agent` with `gpt-5`
- Invoking the agent with natural-language queries
- Inspecting agent message outputs

**Key code pattern:**
```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)
response = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in New York?"}]})
```

---

### 2. Multi-Provider Model Integration
**File:** `2-modelintegration.ipynb`

Demonstrates how LangChain V1's `init_chat_model` provides a **unified interface** to swap between LLM providers without changing the rest of your code. Covers three major providers and three invocation patterns.

**Providers covered:**

| Provider | Model Used | Integration Class |
|---|---|---|
| OpenAI | `gpt-4.1` | `init_chat_model` / `ChatOpenAI` |
| Google Gemini | `gemini-2.5-flash`, `gemini-2.5-flash-lite` | `init_chat_model` / `ChatGoogleGenerativeAI` |
| GROQ | `qwen/qwen3-32b` | `init_chat_model` / `ChatGroq` |

**Invocation patterns covered:**

- `model.invoke()` — Single, synchronous response
- `model.stream()` — Real-time token-by-token streaming with `flush=True`
- `model.batch()` — Parallel processing of multiple independent prompts with concurrency control (`max_concurrency`)

**Streaming example:**
```python
for chunk in model.stream("Write me a paragraph on Artificial Intelligence"):
    print(chunk.text, end="|", flush=True)
```

**Batch example:**
```python
responses = model.batch(
    ["Why do parrots have colorful feathers?", "How do airplanes fly?", "What is quantum computing?"],
    config={"max_concurrency": 5}
)
```

---

### 3. Tools & Tool Execution Loops
**File:** `3-tools.ipynb`

Covers how LLMs use tools to fetch external data or perform actions, and how to build the full tool execution loop — from tool call generation to result injection and final response.

**What's covered:**
- Defining tools using the `@tool` decorator
- Binding tools to a model with `model.bind_tools()`
- Inspecting tool call metadata (name, args, call ID)
- Implementing a 3-step tool execution loop manually

**The Tool Execution Loop:**

```
Step 1 → User sends a message
Step 2 → Model generates a tool call (name + args)
Step 3 → Tool is executed and result is appended to messages
Step 4 → Model receives the tool result and generates a final response
```

**Code example:**
```python
@tool
def get_weather(location: str) -> str:
    """Get the weather at a location"""
    return f"It's sunny in {location}"

model_with_tools = model.bind_tools([get_weather])

# Full loop
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

final_response = model_with_tools.invoke(messages)
```

---

### 4. Messages & Conversation Context
**File:** `4-messages.ipynb`

A deep dive into LangChain's message system — the fundamental unit through which all LLM interactions are structured. Covers all four message types and how to build multi-turn conversation history.

**Message types:**

| Type | Class | Purpose |
|---|---|---|
| System | `SystemMessage` | Sets model behavior, role, and tone |
| Human | `HumanMessage` | Represents user input (text, images, files) |
| AI | `AIMessage` | Model-generated responses and tool calls |
| Tool | `ToolMessage` | Tool execution results passed back to the model |

**What's covered:**
- Text prompts vs. message list prompts and when to use each
- Injecting rich system instructions to shape model behavior
- Using message metadata (`name`, `id`) for tracing and multi-user contexts
- Manually constructing conversation history with `AIMessage`
- The full tool call + `ToolMessage` flow with matched `tool_call_id`
- Accessing `response.usage_metadata` for token tracking

**System message example:**
```python
from langchain.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("""
        You are a senior Python developer with expertise in web frameworks.
        Always provide code examples and explain your reasoning.
    """),
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
```

---

### 5. Structured Output
**File:** `5-structuredoutput.ipynb`

Shows how to force LLMs to return data in a precise, machine-readable schema using `model.with_structured_output()`. Covers three schema definition approaches and their trade-offs.

**Schema approaches:**

| Approach | Best For | Validation |
|---|---|---|
| `Pydantic BaseModel` | Full feature set, field descriptions, nested structures | Runtime validation ✅ |
| `TypedDict` | Lightweight, no runtime validation needed | Static only |
| `@dataclass` | Standard Python data classes with agent `response_format` | Static only |

**What's covered:**
- Basic Pydantic schema → structured model response
- `include_raw=True` to capture both parsed output and raw message
- Nested schemas (e.g., `MovieDetails` containing a list of `Actor` objects)
- Using `TypedDict` with `Annotated` fields
- Using `create_agent` with `response_format` for structured agent output
- Extracting contact information (name, email, phone) from unstructured text

**Pydantic example:**
```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
# → Movie(title='Inception', year=2010, director='Christopher Nolan', rating=8.8)
```

**Nested schema example:**
```python
class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")
```

---

### 6. Middleware
**File:** `6-middleware.ipynb`

The most advanced module — covers LangChain V1's **Middleware** system, which intercepts and controls agent behavior at runtime. Two middleware types are explored in depth.

#### 6a. Summarization Middleware (`SummarizationMiddleware`)

Automatically compresses conversation history when it approaches token or message limits, preserving context without overflowing the context window.

**Trigger modes:**

| Mode | Example | Meaning |
|---|---|---|
| Message count | `trigger=("messages", 10)` | Summarize after 10 messages |
| Token count | `trigger=("tokens", 550)` | Summarize after 550 tokens |
| Context fraction | `trigger=("fraction", 0.005)` | Summarize at 0.5% of context window |

```python
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages", 10),
            keep=("messages", 4)
        )
    ]
)
```

#### 6b. Human-in-the-Loop Middleware (`HumanInTheLoopMiddleware`)

Pauses agent execution before executing specified tools, requiring explicit human approval before proceeding. Supports three decision types:

| Decision | Behavior |
|---|---|
| `approve` | Tool executes as-is |
| `reject` | Tool call is cancelled, agent notified |
| `edit` | Tool arguments are modified before execution |

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False  # No approval needed for reads
            }
        )
    ]
)
```

The agent pauses with an `__interrupt__` signal; a `Command(resume={"decisions": [...]})` is sent to resume execution with the human's decision.

---

## Key Concepts Covered

| Concept | Notebooks |
|---|---|
| Agent creation with `create_agent` | 1, 5, 6 |
| Provider-agnostic model interface | 2 |
| Streaming & batch invocation | 2 |
| Tool definition with `@tool` decorator | 3 |
| Tool execution loop (3-step) | 3 |
| Message types (System, Human, AI, Tool) | 4 |
| Conversation history management | 4 |
| Structured output (Pydantic / TypedDict / Dataclass) | 5 |
| Nested schema extraction | 5 |
| Summarization middleware | 6 |
| Human-in-the-loop with approve / edit / reject | 6 |
| Thread-based memory with `InMemorySaver` | 6 |

---

## Architecture & Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           LangChain Agent               │
│  (create_agent / init_chat_model)       │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │  Middleware   │  │   LLM Provider  │  │
│  │ (Summarize / │◄─►│ OpenAI / Gemini │  │
│  │  HITL)       │  │   / GROQ        │  │
│  └──────────────┘  └─────────────────┘  │
│           │                │            │
│           ▼                ▼            │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │   Memory     │  │     Tools       │  │
│  │(InMemorySaver│  │  (@tool defs)   │  │
│  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────┘
    │
    ▼
Structured Response / Text / Tool Result
```

---

## Environment Variables

The project requires the following API keys stored in a `.env` file at the root:

```env
OPENAI_API_KEY=sk-...          # Required for notebooks 1, 5, 6
GOOGLE_API_KEY=AIza...         # Required for notebook 2 (Gemini)
GROQ_API_KEY=gsk_...           # Required for notebooks 2, 3, 4, 5
```

All notebooks load these automatically using:
```python
from dotenv import load_dotenv
load_dotenv()
```

> **Note:** Never commit your `.env` file to version control. Add it to `.gitignore`.

---

## Author

**Bikash Ranjan Ojha** — [@BikashBIOS]
---

## License

This project is open-source and available under the [MIT License](LICENSE).
