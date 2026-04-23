# Development Notes

## Architecture

- **LangGraph StateGraph** with `InMemorySaver` checkpointing
- **Gemini 2.5 Flash Lite** via `langchain_google_genai`
- **pyttsx3** for TTS (macOS `nsss` driver), voice set to `com.apple.voice.compact.en-US.Samantha`
- **FastAPI** REST API with API key authentication and CORS middleware
- **uv** as package manager

---

## State Machine Flow

```
START → greeting → details_input*
                     └─ greeter_verify [verify_identity 2-of-3]
                             ├─ no match → reject → END
                             └─ match → secret_question_node → secret_input*
                                                                  ├─ wrong → reject_secret → END
                                                                  └─ ok → bouncer [verify_account_type]
                                                                              ├─ not found → reject → END
                                                                              └─ found → request_welcome → request_input*
                                                                                            ├─ exit → farewell → END
                                                                                            └─ request → specialist → request_input* (loop)
```
`*` = interrupt node (waits for human input)

---

## Agents

| Agent | Role |
|---|---|
| Greeter | Collects 2-of-3 auth details (name, phone, IBAN), calls `verify_identity` |
| Bouncer | Calls `verify_account_type` to confirm customer exists and get account tier |
| Specialist | Routes service requests, calls tools, answers balance/account queries, logs complaints |

---

## Tools (specialist_agent)

| Tool | Trigger |
|---|---|
| `get_account_summary(nif)` | Customer asks about balance, IBAN, account number or account details |
| `get_available_specialists()` | Customer asks who can help or which specialist covers a topic |
| `log_complaint(nif, complaint)` | Customer explicitly wants to file a complaint |
| `delegate_hitl(skill)` | Request maps to insurance / investments / accounts and needs a human specialist |

---

## Data Models

### AccountSummary (Pydantic)
Returned by `get_account_summary`. Fields: `account_number`, `iban`, `account_type` (Premium/Regular), `balance` (EUR float).

### IdentityResult (Pydantic)
Output of identity verification. Fields: `verified` (bool), `secret_question` (str), `matched_nif` (str).

---

## Security

- **Prompt injection hardening**: `_sanitize_user_line()` in `graph.py` drops history lines containing `[system`, `ignore all`, `ignore previous`, `new rule:`, `override:`, `context update`, `you are now`, `act as`
- **API key**: `secrets.compare_digest` (constant-time) in `api.py`
- **CORS**: configurable via `ALLOWED_ORIGINS` env var; defaults to `*` (restrict in production)
- **Secrets never baked into Docker image**: injected at runtime via `--env-file`

---

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness check |
| `POST` | `/sessions` | X-API-Key | Start session, returns greeting + `session_id` |
| `POST` | `/sessions/{session_id}/message` | X-API-Key | Send message, returns agent reply + `done` flag |

`done: true` signals the conversation has ended (farewell or rejection).

---

## TTS

- Module: `src/utils/tts.py`
- `speak(text)` — plays audio via shared engine with threading lock
- `save_speech_to_file(text, path)` — creates a fresh engine per call (macOS nsss singleton workaround)
- Voice: `com.apple.voice.compact.en-US.Samantha`
- Tests: `src/test_tts.py` — 10 phrases, subprocess per WAV to isolate nsss driver; `--no-audio` flag skips playback

---

## Known Issues / Decisions

- **macOS nsss singleton bug**: pyttsx3's `save_to_file` only writes audio on the first call per process. Fixed by creating a new engine per `save_speech_to_file` call.
- **uv + uvicorn**: `uv run uvicorn` doesn't work on this machine; use `python -m uvicorn` via the venv directly.
- **CWD-relative paths**: `get_prompts.py` and `database_queries.py` previously used hardcoded `src/...` paths. Fixed to use `Path(__file__)`-relative paths so the server can be started from any directory.

---

## Running the Server

```bash
# Local (development)
PYTHONPATH=src /path/to/.venv/bin/python -m uvicorn api:app \
  --host 127.0.0.1 --port 8000 --app-dir src

# Docker
docker build -t deus-bank-api .
docker run -d --name deus-bank -p 8000:8000 --env-file .env deus-bank-api
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ | Gemini API key |
| `API_KEY` | ✅ | Secret key for API authentication |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | Enable tracing (`true`/`false`) |
| `ALLOWED_ORIGINS` | Optional | CORS origins (comma-separated). Default: `*` |
