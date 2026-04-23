# DEUS Bank — Customer Support AI

An AI-powered customer support system for DEUS Bank, built with LangGraph and Gemini. Handles customer authentication, account queries, complaint logging, and specialist routing through a multi-agent state machine exposed via a REST API.

---

## Features

- **Multi-agent state machine** — Greeter, Bouncer, and Specialist agents orchestrated via LangGraph
- **2-of-3 authentication** — customers authenticate with any two of: full name, phone number, IBAN
- **Secret question verification** — second factor after identity match
- **Account queries** — balance, IBAN, account number (responds only to what was asked)
- **Specialist routing** — routes to human specialists for insurance, investments, or accounts topics
- **Complaint logging** — logs complaints to `src/data/complaints.json`
- **Conversational memory** — agent recalls previous turns within a session
- **Prompt injection hardening** — sanitises history to block system-override attacks
- **Text-to-speech** — agent responses spoken aloud (macOS, `pyttsx3`)
- **REST API** — FastAPI with API key auth and CORS support
- **Docker** — multi-stage image, secrets never baked in

---

## Project Structure

```
customer_support/
├── src/
│   ├── api.py                  # FastAPI application
│   ├── graph.py                # LangGraph state machine
│   ├── main.py                 # CLI chat interface
│   ├── agents/
│   │   ├── greeter.py          # Authentication agent
│   │   ├── bouncer.py          # Account type verification agent
│   │   ├── specialist.py       # Service routing agent
│   │   └── tools.py            # LangChain tools
│   ├── ai/
│   │   ├── llm.py              # Gemini model setup
│   │   └── prompt.yaml         # All agent prompts
│   ├── config/
│   │   └── models.py           # Pydantic models
│   ├── data/
│   │   ├── users.json          # Customer records
│   │   ├── accounts.json       # Account data (balance, IBAN, etc.)
│   │   ├── employees.json      # Specialist employees
│   │   └── complaints.json     # Logged complaints
│   └── utils/
│       ├── database_queries.py # Async JSON data access
│       ├── get_prompts.py      # Prompt loader
│       └── tts.py              # Text-to-speech module
├── docs/
│   └── notes.md                # Technical development notes
├── logs/                       # Chat session logs
├── Dockerfile
├── .dockerignore
├── .env.example
└── pyproject.toml
```

---

## Quickstart

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- Docker (optional)

### Setup

```bash
# Clone and enter the project
cd customer_support

# Install dependencies
uv sync

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY and API_KEY
```

### Run CLI (with TTS)

```bash
PYTHONPATH=src /path/to/.venv/bin/python src/main.py
```

### Run API server

```bash
PYTHONPATH=src /path/to/.venv/bin/python -m uvicorn api:app \
  --host 127.0.0.1 --port 8000 --app-dir src
```

### Run with Docker

```bash
docker build -t deus-bank-api .
docker run -d --name deus-bank -p 8000:8000 --env-file .env deus-bank-api
```

---

## API

All session endpoints require the header `X-API-Key: <your API_KEY>`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check (no auth) |
| `POST` | `/sessions` | Start a new chat session |
| `POST` | `/sessions/{session_id}/message` | Send a message, receive agent reply |

**Response shape:**
```json
{
  "session_id": "uuid",
  "messages": ["Agent reply..."],
  "done": false
}
```

`done: true` means the conversation ended — no further messages can be sent on this session.

**Frontend integration example:**
```js
// 1. Open a session (user opens the chatbot)
const { session_id, messages } = await fetch('/sessions', {
  method: 'POST',
  headers: { 'X-API-Key': API_KEY }
}).then(r => r.json())

// 2. Send each user message
const res = await fetch(`/sessions/${session_id}/message`, {
  method: 'POST',
  headers: { 'X-API-Key': API_KEY, 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: userInput })
}).then(r => r.json())
// res.messages → agent replies to display
// res.done → disable input when true
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ | Gemini API key |
| `API_KEY` | ✅ | Secret key for API authentication |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing key |
| `LANGSMITH_TRACING` | Optional | Enable tracing (`true`/`false`) |
| `ALLOWED_ORIGINS` | Optional | CORS origins (comma-separated). Default: `*` — restrict in production |

---

## Tech Stack

| | |
|---|---|
| LLM | Gemini 2.5 Flash Lite via `langchain-google-genai` |
| Orchestration | LangGraph `StateGraph` + `InMemorySaver` |
| API | FastAPI + uvicorn |
| TTS | pyttsx3 (macOS `nsss`, Samantha voice) |
| Package manager | uv |
| Containerisation | Docker (multi-stage build) |