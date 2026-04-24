# DEUS Bank — Customer Support AI

An AI-powered customer support system for DEUS Bank, built with **LangGraph** and **Google Gemini 2.5 Flash Lite**. Handles customer authentication, account queries, complaint logging, and specialist routing through a multi-agent state machine exposed via a REST API with full voice support.

---

## Features

- **Multi-agent state machine** — Greeter, Bouncer, and Specialist agents orchestrated via LangGraph
- **2-of-3 authentication** — customers authenticate with any two of: full name, phone number, IBAN
- **Secret question verification** — second security factor after identity match, with automatic rejection on failure
- **Details retry loop** — up to 3 authentication attempts before the conversation is rejected
- **Account queries** — balance, IBAN, account number, account type (responds only to what was asked)
- **Specialist routing** — routes to human specialists for insurance, investments, or accounts topics
- **Complaint logging** — logs complaints to `src/data/complaints.json` with timestamp
- **Conversational memory** — agent recalls previous turns within a session
- **Prompt injection hardening** — sanitises history to block system-override attacks; IDENTITY block in specialist prompt
- **Voice round-trip** — STT (Whisper `small`) → agent → TTS → PCM WAV reply, all in one request
- **Text-to-speech** — macOS: native `say` + ffmpeg → PCM WAV; Linux (Docker): `espeak-ng`
- **Centralised logging** — rotating file + console at `logs/deus_bank.log`, controlled by `LOG_LEVEL` env var
- **Session logs** — each conversation saved to `logs/chat_<timestamp>_<session>.md`, written after every turn
- **REST API** — FastAPI with API key auth (`X-API-Key`) and CORS support
- **HTML voice client** — `test/voice_client.html` with Text and Voice tabs
- **Docker** — multi-stage image, `ffmpeg` + `espeak-ng` pre-installed, secrets never baked in

---

## Project Structure

```
customer_support/
├── src/
│   ├── api.py                  # FastAPI application — all REST endpoints
│   ├── graph.py                # LangGraph StateGraph definition and all nodes
│   ├── main.py                 # CLI chat interface (with TTS)
│   ├── agents/
│   │   ├── greeter.py          # Authentication agent (verify_identity tool)
│   │   ├── bouncer.py          # Account verification agent (verify_account_type tool)
│   │   ├── specialist.py       # Service routing agent (4 tools)
│   │   └── tools.py            # All LangChain tools
│   ├── ai/
│   │   ├── llm.py              # Gemini 2.5 Flash Lite model setup
│   │   └── prompt.yaml         # All agent prompts (single source of truth)
│   ├── config/
│   │   ├── logger.py           # Centralised logging configuration
│   │   └── models.py           # All Pydantic models and enums
│   ├── data/
│   │   ├── users.json          # Customer records (name, phone, IBAN, NIF, secret)
│   │   ├── accounts.json       # Account data (balance, IBAN, premium flag)
│   │   ├── employees.json      # Specialist employees and their skills
│   │   └── complaints.json     # Logged complaints (auto-created)
│   └── utils/
│       ├── database_queries.py # Async JSON data access layer
│       ├── get_prompts.py      # YAML prompt loader
│       ├── stt.py              # Speech-to-text (Whisper)
│       └── tts.py              # Text-to-speech (say/espeak-ng + ffmpeg)
├── test/
│   ├── test_state_machine.py   # End-to-end conversation scenario tests
│   ├── test_prompt_injection.py# Security tests for injection attacks
│   ├── test_tts.py             # TTS pipeline tests
│   └── voice_client.html       # Browser-based voice/text chat client
├── logs/                       # Chat session logs + rotating app log
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
- ffmpeg (`brew install ffmpeg` on macOS)
- Docker (optional)

### Setup

```bash
cd customer_support

# Install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env — required keys:
#   GOOGLE_API_KEY=<your Gemini API key>
#   API_KEY=<your chosen API key for the REST API>
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ | — | Google Gemini API key |
| `API_KEY` | ✅ | — | API key for `X-API-Key` header auth |
| `ALLOWED_ORIGINS` | — | `*` | Comma-separated CORS origins |
| `LOG_LEVEL` | — | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `STT_MODEL` | — | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |

### Run the API server

```bash
cd customer_support
/path/to/.venv/bin/uvicorn api:app --app-dir src --port 8000 --reload
```

Or with `PYTHONPATH`:
```bash
PYTHONPATH=src /path/to/.venv/bin/python -m uvicorn api:app --app-dir src --port 8000
```

### Run the CLI (text + TTS)

```bash
PYTHONPATH=src /path/to/.venv/bin/python src/main.py
```

### Run the HTML voice client

Start the API server, then open `test/voice_client.html` in a browser. Enter the API URL and key, click **Connect**, then switch between Text and Voice tabs.

## User for easy testing using voice 
    {
    "name": "John",
    "phone": "123",
    "nif": "123",
    "iban": "ABC",
    "account_number": "AC-123",
    "secret": "What was the name of your first school?",
    "answer": "Testing"
  }

### Run with Docker

```bash
docker build -t deus-bank-api .
docker run -d --name deus-bank -p 8000:8000 --env-file .env deus-bank-api
```

---

## API Reference

All session endpoints require the header `X-API-Key: <your key>`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — no auth required |
| `POST` | `/sessions` | Start a new session, returns greeting |
| `POST` | `/sessions/{id}/message` | Send text, receive JSON reply |
| `POST` | `/sessions/{id}/message/audio` | Send text, receive WAV audio reply |
| `POST` | `/sessions/{id}/voice` | Send audio, receive audio reply (full voice round-trip) |

### Response headers for audio/voice endpoints

| Header | Values | Description |
|---|---|---|
| `X-Session-Done` | `true` / `false` | Whether the conversation has ended |
| `X-Transcript` | string | Transcribed user speech (voice endpoint only) |
| `X-Messages` | JSON array | Agent reply text for display |

---

## Running Tests

```bash

# End-to-end scenario tests (11 scenarios, ~2 min)
/path/to/.venv/bin/python test/test_state_machine.py

# Prompt injection security tests
/path/to/.venv/bin/python test/test_prompt_injection.py

# TTS pipeline tests
/path/to/.venv/bin/python test/test_tts.py
```

---

## Logs

| File | Description |
|---|---|
| `logs/deus_bank.log` | Application log (rotating, 5 MB × 3 backups) |
| `logs/chat_<date>_<session>.md` | Per-session conversation transcript |
| `logs/test_run_<date>.md` | State machine test run report |

---

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