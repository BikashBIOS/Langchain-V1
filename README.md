# 🤖 GenAI Projects — LangChain Learning Series

> A structured collection of Generative AI applications built with LangChain, Groq, HuggingFace, ChromaDB, and FastAPI. This repository progressively teaches core GenAI engineering patterns — from a single LLM call all the way to a full conversational RAG pipeline with persistent memory.

---

## Table of Contents

- [Repository Overview](#repository-overview)
- [Project Map](#project-map)
- [Prerequisites & Setup](#prerequisites--setup)
- [requirements.txt — Every Package Explained](#requirementstxt--every-package-explained)
- [Core Concepts Primer](#core-concepts-primer)
  - [What is LCEL?](#what-is-lcel)
  - [What is RAG?](#what-is-rag)
  - [What are Embeddings & Vector Stores?](#what-are-embeddings--vector-stores)
  - [What is Conversation Memory?](#what-is-conversation-memory)
- [Project 1 — Simple LLM with LCEL (`simplellmLCEL.ipynb`)](#project-1--simple-llm-with-lcel-simplellmlcelipynb)
- [Project 2 — LCEL App as a REST API (`serve.py`)](#project-2--lcel-app-as-a-rest-api-servepy)
- [Project 3 — Stateful Chatbot with Memory (`1-chatbots.ipynb`)](#project-3--stateful-chatbot-with-memory-1-chatbotsipynb)
- [Project 4 — Vector Store & Retriever (`vectorretriever.ipynb`)](#project-4--vector-store--retriever-vectorretrieveripynb)
- [Project 5 — Conversational RAG Q&A (`conversationqa.ipynb`)](#project-5--conversational-rag-qa-conversationqaipynb)
- [How the Projects Build on Each Other](#how-the-projects-build-on-each-other)
- [Architecture Diagrams](#architecture-diagrams)

---

## Repository Overview

This repo is a hands-on learning series for building GenAI applications. Each file introduces new concepts that build directly on the previous one:

| File | Concept Introduced | Complexity |
|---|---|---|
| `simplellmLCEL.ipynb` | Single LLM call, prompt templates, LCEL pipe chains | ⭐ Beginner |
| `serve.py` | Serving an LCEL chain as a REST API with FastAPI + LangServe | ⭐⭐ Beginner+ |
| `1-chatbots.ipynb` | Stateful multi-turn chatbot, session memory, token trimming | ⭐⭐ Intermediate |
| `vectorretriever.ipynb` | Document embeddings, ChromaDB vector store, retrievers | ⭐⭐⭐ Intermediate |
| `conversationqa.ipynb` | Full RAG pipeline + conversation history + automated session store | ⭐⭐⭐⭐ Advanced |

---

## Project Map

```
GenAI-Projects/
├── simplellmLCEL.ipynb     # Project 1 — basic LLM + LCEL translation app
├── serve.py                # Project 2 — same app served as REST API
├── 1-chatbots.ipynb        # Project 3 — stateful chatbot with memory
├── vectorretriever.ipynb   # Project 4 — embeddings, Chroma, RAG retriever
├── conversationqa.ipynb    # Project 5 — conversational RAG Q&A bot
├── requirements.txt        # All Python dependencies
└── .gitignore
```

---

## Prerequisites & Setup

```bash
# 1. Clone the repo
git clone https://github.com/BikashBIOS/GenAI-Projects.git
cd GenAI-Projects

# 2. Create and activate a virtual environment
python -m venv genaienv
source genaienv/bin/activate       # Linux/Mac
# genaienv\Scripts\activate        # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Create your .env file
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=GenAI-Projects
LANGCHAIN_TRACING_V2=true
EOF

# 5. Open Jupyter to run the notebooks
jupyter lab
```

**Where to get API keys:**
- **Groq:** https://console.groq.com — fast, free LLM inference (LLaMA, Mixtral etc.)
- **HuggingFace:** https://huggingface.co/settings/tokens — needed for HuggingFace embeddings
- **LangSmith:** https://smith.langchain.com — optional tracing/observability for LangChain

---

## requirements.txt — Every Package Explained

```
pandas              # Data manipulation — used for tabular data handling
numpy               # Numerical computing — vector math underpinning embeddings
ipykernel           # Jupyter notebook kernel — allows notebooks to run
langchain           # Core LangChain framework — chains, prompts, runnables
langsmith           # LangChain's observability/tracing tool — traces chains
dotenv              # Load .env files into os.environ at runtime
langchain_groq      # LangChain adapter for Groq's API (fast LLaMA inference)
langchain_core      # Core LangChain primitives: messages, runnables, parsers
fastapi             # Modern async Python web framework (serves the LCEL chain)
uvicorn             # ASGI server — runs the FastAPI app
langserve           # LangChain's built-in FastAPI integration to expose chains
sse_starlette       # Server-Sent Events support for streaming LangServe responses
langchain_community # Community integrations: WebBaseLoader, ChatMessageHistory
langchain_chroma    # LangChain adapter for ChromaDB vector database
langchain_huggingface # LangChain adapter for HuggingFace embedding models
sentence-transformers # The library powering the all-MiniLM-L6-v2 embedding model
torch               # PyTorch — required runtime for sentence-transformers
bs4                 # BeautifulSoup4 — HTML parsing used by WebBaseLoader
langchain_classic   # Classic LangChain chain builders (RAG, history-aware retriever)
```

---

## Core Concepts Primer

Before diving into the code, here are the foundational ideas powering every project in this repo.

### What is LCEL?

**LangChain Expression Language (LCEL)** is a declarative way to compose LangChain components using the `|` (pipe) operator. Each component must implement the `Runnable` interface, which provides `.invoke()`, `.batch()`, `.stream()` methods.

```
prompt | model | parser
  ↓         ↓        ↓
formats    calls    extracts
input      LLM      text
```

The pipe passes the output of the left side as input to the right side. This means you can build complex chains in a single readable line — and every piece is swappable.

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that gives the LLM access to your own documents/data by:
1. Breaking your documents into chunks
2. Embedding each chunk as a vector (a list of numbers representing meaning)
3. Storing those vectors in a database
4. At query time: embedding the user's question, finding the most similar document chunks, and injecting them into the LLM prompt as context

Without RAG, the LLM only knows what it was trained on. With RAG, it can answer questions about your specific data.

```
User Question
     ↓
Embed question → [0.1, 0.8, 0.3, ...]
     ↓
Search vector DB for similar chunks
     ↓
Inject found chunks into prompt
     ↓
LLM answers using that context
```

### What are Embeddings & Vector Stores?

An **embedding** is a mathematical transformation of text into a dense vector of numbers, where semantically similar texts produce similar vectors. The `all-MiniLM-L6-v2` model used in this project maps any text to a 384-dimensional vector.

A **vector store** (like ChromaDB) is a database optimized for storing and searching these vectors by cosine similarity or L2 distance — finding the documents that mean the same thing as your query, not just those that share keywords.

### What is Conversation Memory?

By default, each LLM call is completely stateless — it has no knowledge of what was said before. **Conversation memory** solutions in LangChain solve this by maintaining a list of past `HumanMessage` and `AIMessage` objects and injecting them into every new prompt. The challenge is that token limits mean you can't keep the full history forever — which is why token trimming becomes important as conversations grow.

---

## Project 1 — Simple LLM with LCEL (`simplellmLCEL.ipynb`)

**Goal:** Build a text translation app using a single LCEL chain.

**Concepts introduced:** LLM setup, direct model invocation, output parsing, prompt templates, LCEL pipe operator.

---

### Cell 0 — Environment Setup

```python
import os
from dotenv import load_dotenv
load_dotenv()
```

`load_dotenv()` scans for a `.env` file in the current directory and loads all key=value pairs as environment variables. This means `os.getenv("GROQ_API_KEY")` will work in subsequent cells without hardcoding secrets into your notebook.

---

### Cell 1 — Initialize the LLM

```python
from langchain_groq import ChatGroq
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
```

`ChatGroq` is LangChain's adapter class for Groq's API. **Groq** is an AI inference company that runs open-source models (LLaMA, Mixtral etc.) on custom hardware called LPUs — much faster than GPU-based providers. `llama-3.1-8b-instant` is Meta's 8-billion-parameter LLaMA 3.1 model, optimized for fast responses.

The `model` object is a `Runnable` — it implements `.invoke()` and can be used inside LCEL chains.

---

### Cell 2 — Direct Model Invocation

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English to French"),
    HumanMessage(content="Hello How are You?")
]

result = model.invoke(messages)
```

`SystemMessage` sets the model's role and instructions — it's the first item in the prompt, telling the model *how to behave*. `HumanMessage` represents the user's input. Passing a list of messages to `model.invoke()` sends them all as a structured conversation to the LLM. The returned `result` is an `AIMessage` object containing the model's raw response, including metadata like token usage.

---

### Cell 3 — Output Parser

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
parser.invoke(result)
```

The raw `AIMessage` from the model contains not just the text but also metadata (token counts, model name, stop reason etc.). `StrOutputParser` extracts just the text content from the `AIMessage` and returns a clean Python string. This is the most basic parser — LangChain also provides `JsonOutputParser`, `PydanticOutputParser`, and others.

---

### Cell 4 — First LCEL Chain (model → parser)

```python
chain = model | parser
chain.invoke(messages)
```

This is LCEL in action. The `|` operator connects two Runnables into a pipeline:
1. `messages` are passed to `model.invoke()` → produces an `AIMessage`
2. That `AIMessage` is passed to `parser.invoke()` → produces a clean string

The chain object itself is also a Runnable, so `.invoke()`, `.batch()`, and `.stream()` all work on it.

---

### Cell 5 — Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

generic_template = "Translate the Following into {language}:"

prompt = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")]
)
```

`ChatPromptTemplate` is a reusable template with variable placeholders using `{curly_brace}` syntax. `from_messages()` takes a list of `(role, template_string)` tuples. The `{language}` and `{text}` slots are filled at runtime when you call `.invoke()`. This separates the structure of the prompt from its dynamic content — you can reuse the same template for French, Spanish, Japanese etc. just by changing the input dict.

---

### Cell 6 & 7 — Invoking the Prompt

```python
result = prompt.invoke({"language": "French", "text": "Hello"})
result.to_messages()
```

`prompt.invoke()` fills the template variables and returns a `ChatPromptValue` object. `.to_messages()` converts it to the list-of-messages format that the model expects. This intermediate step is useful for debugging — you can inspect the exact messages being sent to the LLM before chaining.

---

### Cell 8 — Full LCEL Chain (prompt → model → parser)

```python
chain = prompt | model | parser
chain.invoke({"language": "French", "text": "Hello"})
```

This is the complete translation pipeline as a single LCEL chain:
1. `prompt` formats `{"language": "French", "text": "Hello"}` into a list of messages
2. `model` calls the Groq API and returns an `AIMessage`
3. `parser` extracts the translated string from the `AIMessage`

The result is `"Bonjour"` (or equivalent). This same three-component pattern (`prompt | model | parser`) is the fundamental building block of nearly every LangChain application.

---

## Project 2 — LCEL App as a REST API (`serve.py`)

**Goal:** Turn the LCEL translation chain into a deployable REST API with auto-generated documentation.

**Concepts introduced:** FastAPI, LangServe, `add_routes`, Uvicorn.

---

### Full Code

```python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
import uvicorn
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

system_template = "Translate the Following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template),
     ("user", "{text}")]
)

parser = StrOutputParser()

## Create chain
chain = prompt_template | model | parser

## App definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Simple API server using Langchain runnable Interfaces"
)

## Adding Chain Routes
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

---

### Line-by-Line Walkthrough

**Lines 1–8 — Imports:** `FastAPI` is the web framework. `langserve.add_routes` is the key function that automatically exposes an LCEL chain as REST endpoints. `uvicorn` is the ASGI server that runs FastAPI.

**Lines 10–12 — LLM Setup:** Identical to the notebook — load the API key from `.env` and initialize the Groq model.

**Lines 14–18 — Prompt Template:** Same `ChatPromptTemplate` as in the notebook. The template has two inputs: `{language}` (which language to translate to) and `{text}` (what to translate).

**Line 21 — Parser:** `StrOutputParser()` strips the `AIMessage` wrapper and returns a plain string.

**Line 24 — The Chain:** `prompt_template | model | parser` — exactly the same three-step pipe chain as the notebook, now ready to be served.

**Lines 27–31 — FastAPI App:**
```python
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Simple API server using Langchain runnable Interfaces"
)
```
This creates the web application instance. The `title`, `version`, and `description` appear in the auto-generated Swagger UI at `http://127.0.0.1:8000/docs`.

**Lines 34–35 — Adding Routes:**
```python
add_routes(app, chain, path="/chain")
```
This is the magic line. `add_routes` from LangServe automatically creates **five endpoints** on `/chain`:
- `POST /chain/invoke` — call the chain with a single input
- `POST /chain/batch` — call the chain with multiple inputs in parallel
- `POST /chain/stream` — call the chain and stream back the response token by token
- `GET  /chain/input_schema` — get the JSON schema of valid inputs
- `GET  /chain/output_schema` — get the JSON schema of the output

You also get a built-in playground at `http://127.0.0.1:8000/chain/playground/`.

**Lines 37–38 — Server Launch:**
```python
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```
`if __name__ == "__main__"` ensures this only runs when you execute `python serve.py` directly, not when the file is imported as a module. `uvicorn` is the production-grade ASGI server that handles concurrent HTTP requests.

**To run:**
```bash
python serve.py
# Then visit: http://127.0.0.1:8000/chain/playground/
```

---

## Project 3 — Stateful Chatbot with Memory (`1-chatbots.ipynb`)

**Goal:** Build a chatbot that remembers what was said earlier in the conversation, handles multi-language responses, and stays within token limits.

**Concepts introduced:** Multi-turn conversation state, `ChatMessageHistory`, `RunnableWithMessageHistory`, `MessagesPlaceholder`, token trimming with `trim_messages`, session isolation.

---

### Section 1 — Naive Multi-Turn (No Memory)

```python
from langchain_core.messages import HumanMessage
model.invoke([HumanMessage(content="Hi, my name is Bikash and I am a functional consultant")])
```

This works for one message. But the model has no memory — every call is independent.

```python
from langchain_core.messages import AIMessage
model.invoke([
    HumanMessage(content="Hi, my name is Bikash and I am a functional consultant"),
    AIMessage(content="Nice to meet you, Bikash..."),
    HumanMessage(content="Hey what's my name and what do I do?")
])
```

The manual workaround — pass the entire conversation history as a list on every call. The LLM processes the full history and can answer "Your name is Bikash and you're a functional consultant." This works but becomes unmanageable as conversations grow. The next section automates it.

---

### Section 2 — Automated Session Memory

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)
```

**Breaking this down:**

`store = {}` — A simple Python dictionary acting as an in-memory session database, mapping session IDs to their history objects.

`get_session_history(session_id)` — A lookup function. Given a session ID string, it returns the `ChatMessageHistory` for that session, creating a new one if this is the first message in that session. This is the key abstraction: LangChain will call this function automatically before every model invocation.

`ChatMessageHistory` — An in-memory list of `HumanMessage` and `AIMessage` objects. It auto-appends every input and output to maintain the full conversation log.

`RunnableWithMessageHistory(model, get_session_history)` — Wraps the model in a stateful shell. Before each call, it calls `get_session_history(session_id)` to get the history, prepends it to the input messages, and after the call, saves the new exchange into that history. The user never has to manage any of this manually.

```python
config = {"configurable": {"session_id": "chat1"}}

response = with_message_history.invoke(
    [HumanMessage(content="Hi, My favorite food is Paneer and I love it with Maggi masala.")],
    config=config
)
```

`config = {"configurable": {"session_id": "chat1"}}` — The session identifier. This is passed to `get_session_history` so it knows which conversation bucket to use. Changing to `"chat2"` starts a completely fresh, isolated conversation.

```python
config1 = {"configurable": {"session_id": "chat1"}}
response = with_message_history.invoke(
    [HumanMessage(content="What's my favorite food?")],
    config=config1
)
```

Because this uses the same `"chat1"` session ID, the previous message is already in the history. The model correctly responds that the favorite food is Paneer.

---

### Section 3 — Prompt Templates with Memory

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant and answer all the question with the best of your ability in {language}"),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt | model
```

`MessagesPlaceholder(variable_name="messages")` — This is the critical piece. It inserts the entire conversation history at this position in the prompt template. When `RunnableWithMessageHistory` prepopulates the history, it goes into this slot. This lets you combine a static system prompt (persona, language settings) with the dynamic, growing conversation history.

The chain is now `prompt | model` — no parser yet, because the next step wraps it in memory management.

```python
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)
```

`input_messages_key="messages"` — Tells `RunnableWithMessageHistory` which key in the input dict contains the messages to track. The chain now expects a dict like `{"messages": [...], "language": "Hindi"}`.

```python
config = {"configurable": {"session_id": "chat4"}}
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="I love paneer")], "language": "Hindi"},
    config=config
)
```

The model now responds in Hindi (controlled by the `language` variable), and subsequent messages will also be in Hindi for this session.

---

### Section 4 — Token Trimming

As conversations grow, they eventually exceed the model's context window limit. Token trimming prevents this.

```python
from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=70,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)
```

**Each parameter explained:**

- `max_tokens=70` — Hard cap: keep the total token count at or below 70. In production you'd set this much higher (e.g. 4000 for a 4096-token model).
- `strategy="last"` — When trimming, keep the *most recent* messages. Older messages get dropped first. (Alternative: `"first"` keeps the oldest messages.)
- `token_counter=model` — Use the actual model to count tokens. This ensures accurate counting for the specific model being used, since different tokenizers count differently.
- `include_system=True` — Never trim the `SystemMessage` — always keep the persona/instructions.
- `allow_partial=False` — Don't cut a message halfway through. If a message would be partially trimmed, drop it entirely.
- `start_on="human"` — After trimming, make sure the first remaining message is a `HumanMessage`. Avoids the LLM seeing an AI response without the prior human turn that prompted it.

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
```

This is a more advanced LCEL pattern:

- `itemgetter("messages")` — Extracts just the `"messages"` key from the input dict
- `| trimmer` — Runs the message list through the trimmer to reduce it
- `RunnablePassthrough.assign(messages=...)` — Takes the full input dict and replaces just the `"messages"` key with the trimmed version, passing everything else unchanged
- The result flows into `prompt` then `model`

The trimmer runs *before* the prompt is formatted, so the formatted prompt never exceeds the token limit.

```python
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)
```

The final version wraps the trimming-enabled chain in session memory. Every turn now: loads history → trims to fit budget → formats prompt → calls model → saves result.

---

## Project 4 — Vector Store & Retriever (`vectorretriever.ipynb`)

**Goal:** Store documents as vectors, query them by semantic meaning, and build a RAG pipeline.

**Concepts introduced:** `Document`, `HuggingFaceEmbeddings`, `Chroma` vector store, `similarity_search`, retrievers, LCEL RAG chain.

---

### Section 1 — Creating Documents

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
```

`Document` is LangChain's standard data class for a piece of text. It has exactly two fields:

- `page_content` — the actual text content to be embedded and searched
- `metadata` — arbitrary key-value pairs for filtering and provenance tracking. Here, `"source"` identifies which document category the text came from.

In real applications these documents come from PDFs, web pages, databases etc. Here they're hardcoded for demonstration.

---

### Section 2 — Embeddings

```python
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
```

Note `os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")` — this re-exports the HuggingFace token into the environment so the `langchain_huggingface` library can find it.

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

`HuggingFaceEmbeddings` loads the `all-MiniLM-L6-v2` model from HuggingFace and runs it locally to generate embeddings. This model:
- Converts any text into a 384-dimensional vector
- Is small (~80MB) and fast
- Has very good semantic understanding despite its size
- Is free and runs on CPU without a GPU

All documents and queries will pass through this model to get their vector representations before any similarity comparison.

---

### Section 3 — Building the Vector Store

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(documents, embedding=embeddings)
```

`Chroma.from_documents()` does three things in one call:
1. Passes each `Document.page_content` through the `embeddings` model to get a 384-dim vector
2. Stores those vectors (and the original text + metadata) in ChromaDB
3. Returns a `Chroma` object ready for querying

ChromaDB stores everything in-memory by default. For persistence, you'd add `persist_directory="./chroma_db"`.

---

### Section 4 — Querying the Vector Store

```python
vectorstore.similarity_search("cat")
```

**What happens inside:** The query `"cat"` is passed through the same `all-MiniLM-L6-v2` embedding model to get its vector. ChromaDB then computes the cosine similarity between that query vector and all stored document vectors and returns the top 4 most similar documents (default `k=4`).

The result includes the Cats document (most similar) and likely the Dogs and Rabbits documents (mammal pets — semantically related), even though neither "dog" nor "rabbit" contains the word "cat". This is the power of semantic search over keyword search.

```python
await vectorstore.asimilarity_search("cat")
```

The async version — identical behavior but non-blocking, useful in async web frameworks.

---

### Section 5 — Retrievers (Two Approaches)

VectorStore objects have powerful search methods, but they can't be plugged directly into LCEL chains because they don't implement the `Runnable` interface. Retrievers wrap them so they can.

**Approach A — Manual Lambda Wrapper:**

```python
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
retriever.batch(["cat", "dog"])
```

`RunnableLambda` wraps any Python callable as a LangChain Runnable. `.bind(k=1)` permanently attaches `k=1` as a parameter, so every call will only return the single best match. `.batch()` runs multiple queries in parallel.

**Approach B — Native Conversion (Recommended):**

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

retriever.batch(["cat", "dog"])
```

`vectorstore.as_retriever()` is the official, idiomatic way. It creates a `VectorStoreRetriever` object that implements the full `Runnable` interface. `search_type="similarity"` uses cosine similarity (the default and most common). `search_kwargs={"k": 1}` again caps results at 1 document per query. This approach is preferred because it's cleaner and supports more search strategies (e.g., `"mmr"` for Maximal Marginal Relevance).

---

### Section 6 — Building the RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about dogs")
print(response.content)
```

**Dissecting the chain's entry point:**

```python
{"context": retriever, "question": RunnablePassthrough()}
```

This is a dictionary of Runnables — LCEL's way of routing one input to multiple processors simultaneously. When `"tell me about dogs"` is passed to `.invoke()`:

- `retriever` receives it and returns the relevant `Document` objects (fetched from ChromaDB)
- `RunnablePassthrough()` receives it and passes it through unchanged

So the dictionary produces: `{"context": [Document(...)], "question": "tell me about dogs"}`. This flows into:
- `prompt` — formats the question and context into a human message
- `llm` — generates the answer based only on the provided context

This is a minimal but complete RAG pipeline.

---

## Project 5 — Conversational RAG Q&A (`conversationqa.ipynb`)

**Goal:** Build a full RAG chatbot that loads real web content, answers questions based on that content, and remembers the entire conversation history.

**Concepts introduced:** `WebBaseLoader`, `RecursiveCharacterTextSplitter`, `create_retrieval_chain`, `create_stuff_documents_chain`, `create_history_aware_retriever`, `RunnableWithMessageHistory` for RAG, automated session store.

---

### Section 1 — Setup

```python
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

All the imports gathered at the top. This notebook uses `langchain_classic` (the older chain-building API) alongside `langchain_core` (the newer LCEL API).

---

### Section 2 — Loading Web Content

```python
import bs4
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
```

`WebBaseLoader` fetches the webpage at the URL and uses BeautifulSoup4 to parse the HTML. The important parameters:

- `web_paths=("https://...",)` — a tuple of URLs to scrape (can be multiple)
- `bs_kwargs=dict(parse_only=...)` — passes arguments directly to BeautifulSoup. This is a filter
- `bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))` — tells BeautifulSoup to only extract HTML elements that have these CSS class names

This selective extraction is crucial — without it, you'd get navigation bars, footers, ads etc. mixed into your documents, degrading RAG quality. Only the actual blog post content is extracted.

The target URL is Lilian Weng's famous "LLM Powered Autonomous Agents" blog post — a 5000+ word article about AI agent architectures.

---

### Section 3 — Chunking & Indexing

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

**Why chunking is necessary:** The blog post is ~5000 words — far too long to embed as a single vector or inject wholesale into an LLM prompt. It must be split into smaller, semantically coherent pieces.

`RecursiveCharacterTextSplitter`:
- `chunk_size=1000` — each chunk is at most 1000 characters
- `chunk_overlap=200` — consecutive chunks share 200 characters of overlap

The overlap prevents key concepts from being split across chunks where neither chunk contains enough context on its own. The "recursive" part means it tries to split on natural boundaries in order: paragraphs (`\n\n`), then newlines (`\n`), then sentences (`. `), then words (` `), then characters. It only moves to smaller splits if the chunk would otherwise exceed `chunk_size`.

After splitting, all chunks are embedded and stored in ChromaDB, and a retriever is created in one line.

---

### Section 4 — Basic RAG Chain

```python
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
```

The system prompt explicitly instructs the model to:
- Only use the retrieved context (not its own training knowledge)
- Admit ignorance if the context doesn't contain the answer
- Keep answers short (3 sentences max)

The `{context}` placeholder will be filled with the retrieved document chunks. The `{input}` placeholder will receive the user's question.

```python
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

`create_stuff_documents_chain` — Creates a chain that "stuffs" (concatenates) the retrieved documents into the `{context}` slot of the prompt. All retrieved chunks are joined into one big context string and sent to the LLM.

`create_retrieval_chain` — Combines the retriever and the QA chain. When you invoke it with `{"input": "question"}`, it:
1. Passes the question to the retriever → gets relevant document chunks
2. Passes the question + chunks to the QA chain → LLM generates answer
3. Returns `{"input": ..., "context": ..., "answer": ...}`

```python
response = rag_chain.invoke({"input": "what is self reflection?"})
response['answer']
```

The response dict has `"answer"` (the LLM's text) and `"context"` (the documents that were retrieved). This RAG chain has no memory — each invocation is completely independent.

---

### Section 5 — History-Aware Retriever

The basic RAG chain fails for follow-up questions. If the user asks "What is self-reflection?" and then asks "How do we achieve it?" — the word "it" has no standalone meaning for the retriever. It needs to see the previous question to understand what "it" refers to.

```python
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

This prompt tells the LLM to act as a question rewriter. Given the chat history and the current question, it should rephrase the question to be self-contained. For example:
- History: Q: "What is self-reflection?" A: "Self-reflection is..."
- Current: "How do we achieve it?"
- Rewritten: "How do we achieve self-reflection?"

The rewritten standalone question is then what gets sent to the vector store retriever.

```python
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

`create_history_aware_retriever` chains together: LLM (question rewriter) → original retriever. If there's no chat history, the question is passed directly to the retriever unchanged. If there IS history, the LLM first rewrites it, then the retriever uses the rewritten question.

---

### Section 6 — Full Conversational RAG Chain

```python
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

The QA prompt is rebuilt to include `MessagesPlaceholder("chat_history")`. This means the LLM sees the full conversation history when generating its answer — it can refer back to previous answers for coherence.

```python
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

Identical structure to before, but now uses the `history_aware_retriever` and the conversation-aware `qa_prompt`.

---

### Section 7 — Manual History Management

```python
from langchain_core.messages import AIMessage, HumanMessage

chat_history = []
question = "What is Self-Reflection"
response1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

chat_history.extend([
    HumanMessage(content=question),
    AIMessage(content=response1["answer"])
])

question2 = "Tell me more about it?"
response2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})
print(response2['answer'])
```

Here the history is managed manually — the developer is responsible for appending each exchange to `chat_history`. The second question "Tell me more about it?" works correctly because `chat_history` contains the context about Self-Reflection from the first turn. This is explicit but tedious for real applications.

---

### Section 8 — Automated Session History (Production Pattern)

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```

This is the same session store pattern from the chatbot notebook, now applied to the full RAG chain.

The three key parameters:
- `input_messages_key="input"` — which key in the input dict is the user's current message
- `history_messages_key="chat_history"` — which key the history should be injected into (matches the `MessagesPlaceholder("chat_history")` in the prompts)
- `output_messages_key="answer"` — which key of the output dict contains the assistant's response (for saving to history)

```python
conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={"configurable": {"session_id": "abc123"}}
)["answer"]
```

First question in session `"abc123"`. The `store["abc123"]` is empty initially, so the history-aware retriever passes the question directly to the vector store.

```python
conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}}
)["answer"]
```

Second question. The wrapper automatically:
1. Loads the history from `store["abc123"]` (contains previous Q&A)
2. Passes it to the `history_aware_retriever`, which rewrites "What are common ways of doing it?" → "What are common ways of achieving task decomposition?"
3. Retrieves relevant chunks from the vector store using the standalone question
4. Injects the chunks + full history into the QA prompt
5. LLM generates a contextually coherent answer
6. Saves the new Q&A pair to `store["abc123"]`

```python
store
```

At this point `store["abc123"]` contains the full conversation log as a `ChatMessageHistory` object. Multiple sessions can run simultaneously with complete isolation.

---

## How the Projects Build on Each Other

```
simplellmLCEL.ipynb
   └── LLM + prompt + parser → LCEL pipe chain
            ↓
serve.py
   └── Adds: FastAPI + LangServe → REST API from LCEL chain
            ↓
1-chatbots.ipynb
   └── Adds: ChatMessageHistory + RunnableWithMessageHistory → stateful memory
       Adds: MessagesPlaceholder → dynamic history injection into prompts
       Adds: trim_messages → token budget management
            ↓
vectorretriever.ipynb
   └── Adds: Document + HuggingFaceEmbeddings → semantic vectors
       Adds: ChromaDB → vector storage and similarity search
       Adds: as_retriever() → LCEL-compatible retrieval
       Adds: RAG chain pattern → context-aware answering
            ↓
conversationqa.ipynb
   └── Adds: WebBaseLoader + RecursiveCharacterTextSplitter → real document ingestion
       Adds: create_history_aware_retriever → context-aware query rewriting
       Adds: RunnableWithMessageHistory + RAG → full conversational RAG system
```

---

## Architecture Diagrams

### Complete Conversational RAG Architecture

```
USER QUESTION: "What are common ways of doing it?"
                    │
                    ▼
         ┌─────────────────────────┐
         │  RunnableWithMessage    │  ← loads session history from store{}
         │  History Wrapper        │
         └──────────┬──────────────┘
                    │
                    ▼ {input, chat_history}
    ┌───────────────────────────────────────┐
    │     History-Aware Retriever           │
    │  ┌──────────────────────────────────┐ │
    │  │ If chat_history not empty:       │ │
    │  │   LLM rewrites question to       │ │
    │  │   standalone form:               │ │
    │  │   "What are common ways of       │ │
    │  │    achieving task decomposition?"│ │
    │  └──────────────────────────────────┘ │
    │                    │                  │
    │                    ▼                  │
    │  ┌──────────────────────────────────┐ │
    │  │   ChromaDB Vector Store          │ │
    │  │   similarity_search(rewritten q) │ │
    │  │   → [chunk1, chunk2, chunk3]     │ │
    │  └──────────────────────────────────┘ │
    └───────────────────────────────────────┘
                    │
                    ▼ {input, chat_history, context=[chunks]}
    ┌───────────────────────────────────────┐
    │   create_stuff_documents_chain        │
    │   Prompt:                             │
    │     system: "Use context to answer.  │
    │              {context}"               │
    │     chat_history: [Q1, A1, ...]      │
    │     human: "What are common ways...?"│
    │                    │                  │
    │                    ▼                  │
    │            Groq LLaMA 3.1 8B         │
    └───────────────────────────────────────┘
                    │
                    ▼
              FINAL ANSWER
                    │
                    ▼
         store["abc123"].add_messages(Q, A)  ← auto-saved
```

### LCEL Pipe Chain Data Flow

```
prompt | model | parser

Input dict: {"language": "French", "text": "Hello"}
    │
    ▼
ChatPromptTemplate
    Fills {language} and {text} slots
    Produces: ChatPromptValue
    [SystemMessage("Translate to French:"), HumanMessage("Hello")]
    │
    ▼
ChatGroq (LLaMA 3.1 8B via Groq API)
    Sends messages to API
    Produces: AIMessage(content="Bonjour", usage_metadata={...})
    │
    ▼
StrOutputParser
    Extracts .content from AIMessage
    Produces: "Bonjour"
```

---

*Built with [LangChain](https://python.langchain.com/), [Groq](https://console.groq.com/), [HuggingFace](https://huggingface.co/), [ChromaDB](https://www.trychroma.com/), and [FastAPI](https://fastapi.tiangolo.com/).*
