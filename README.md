# DEUS Bank — AI Customer Support

A multi-agent customer support system built on **LangGraph** with a **FastAPI**
backend and a **React admin UI**. Callers are greeted, identified (≥ 2 of 3:
name / phone / IBAN, plus a secret-question challenge), tier-classified, and
routed to the right specialist — over text, push-to-talk voice, or streaming
voice (Deepgram STT/TTS).

> See [`docs/architecture.md`](docs/architecture.md) for the design decisions,
> state machine, security model, and known limitations.
> The rendered graph lives at [`docs/graph.png`](docs/graph.png).

## Quick start (zero external services)

Requires **Python 3.12+**, [`uv`](https://github.com/astral-sh/uv), and
**Node 20+** for the admin UI.

```bash
# 1. Backend
make install
echo "GOOGLE_API_KEY=your-google-ai-studio-key" > .env
make run                       # FastAPI on http://localhost:8004
```

```bash
# 2. Admin UI (in another terminal)
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

The Vite dev server proxies `/api/*`, `/chat`, and `/voice` to port 8004,
so both layers share the same origin during development.

**Only `GOOGLE_API_KEY` is required.** Everything else (MongoDB, Deepgram)
is opt-in.

## Configuration

Create a `.env` file at the repo root. Only the first variable is required.

```bash
GOOGLE_API_KEY=...                          # REQUIRED — Google AI Studio key

# Optional — voice (STT + TTS)
DEEPGRAM_API_KEY=...
DEEPGRAM_STT_MODEL=nova-2-general
DEEPGRAM_TTS_MODEL=aura-2-thalia-en

# Optional — persistent state + summaries
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=deus_bank
```

### What each toggle changes

| Variable | Unset (default) | Set |
|---|---|---|
| `GOOGLE_API_KEY` | App fails to start | Gemini powers all LLM nodes |
| `MONGODB_URL` | LangGraph uses `InMemorySaver` (state lost on restart); summaries fall back to `data/summaries.jsonl` | LangGraph checkpoints to Mongo; summaries land in the `summaries` collection |
| `DEEPGRAM_API_KEY` | `/chat` ignores `audio_base64`; `/voice` accepts the WS then closes with 1011 ("Deepgram not configured"); admin UI Voice tab is text-only | `/chat` accepts and returns audio; `/voice` PTT works end-to-end |

The architecture is intentionally tiered so you can run *any* combination —
text-only with no infra, or full voice + persistence with everything wired.

### With everything (Mongo + Deepgram)

Easiest path is `docker compose up --build`, which brings up a local Mongo
plus the backend together. Or run Mongo separately and:

```bash
DEEPGRAM_API_KEY=... MONGODB_URL=mongodb://localhost:27017 make run
```

## Admin UI

Open <http://localhost:5173> when both servers are running:

- **Test chat** — full text conversation against the agent graph, with a
  debug pane showing the live `AgentState` (stage, identity, routing,
  retries, latency).
- **Voice** — Push-to-talk mode (working). Streaming mode is wired but
  marked *(preview)* — see Voice limitations in
  [`docs/architecture.md`](docs/architecture.md).
- **Configure** — edit routing services, phrase templates (grouped by
  conversation stage), and post-call summary metrics live; changes take
  effect immediately.
- **Summaries** — paginated list + iMessage-style transcript detail for
  every completed call.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | One conversation turn (text, optionally with `audio_base64`) |
| `WS` | `/voice` | Streaming voice conversation (Deepgram Flux) |
| `GET` | `/api/health` | Status + feature flags (`deepgram`, `mongo`) |
| `GET / PUT` | `/api/config/routing` | Service routing rules |
| `GET / PUT` | `/api/config/phrases` | Phrase templates per stage |
| `GET / PUT` | `/api/config/metrics` | Post-call summary metrics schema |
| `GET` | `/api/summaries` | Paginated post-call summaries |
| `GET` | `/api/summaries/{id}` | Single summary detail |
| `GET` | `/api/sessions/{id}/state` | Current `AgentState` snapshot |

## Make targets

| Command | What it does |
|---|---|
| `make install` | Install backend deps via `uv sync --all-extras` |
| `make run` | uvicorn on port 8004 with auto-reload |
| `make test` | Run the pytest suite |
| `make check` | Lint + format check + tests |
| `make simulate` | Run one canned scripted call against `/chat` |
| `make simulate-all` | Run every canned scenario back-to-back |
| `make simulate-interactive` | REPL session against `/chat` |
| `make simulate-voice AUDIO=path/to.wav` | Send a recorded audio file through `/chat` |
| `make simulate-call-live` | Live-mic streaming test against `/voice` |
| `make simulate-call-ptt` | Local PTT loop against `/chat` |
| `make draw-graph` | Re-render `docs/graph.png` |
| `make docker-build` / `make docker-up` | Container path |

## Tests

```bash
make test                      # backend (pytest, ~250 tests)
cd frontend && npm test        # admin UI (vitest, ~20 tests)
```

CI-style one-shot:

```bash
make check                     # ruff lint + format + pytest
cd frontend && npx tsc --noEmit && npx vitest run
```

---

<details>
<summary><strong>Original challenge brief</strong></summary>

A customer calls the bank, hoping to get help, but instead, they get lost in
an endless phone menu maze. Build an **AI-powered customer support system**
where multiple agents work together to identify the customer and route them
to the right place.

- **The Greeter** — friendly face of the bank; opens the conversation, asks
  for identification, makes sure the customer is legitimate.
- **The Bouncer** — once identified, decides regular / premium / non-customer.
- **The Specialist** — for specific high-value requests, ensures they reach
  the right expert.
- **Guardrails** — keeps everything safe, professional, and policy-compliant.

Verification requires matching **at least two of three** details
(`name`, `phone`, `iban`) before asking the secret question.

```python
example_of_user = {
  "name": "Lisa",
  "phone": "+1122334455",
  "iban": "DE89370400440532013000",
  "secret": "Which is the name of my dog?",
  "answer": "Yoda",
}

example_of_account = {
  "iban": "DE89370400440532013000",
  "premium": True,
}
```

</details>
