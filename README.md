# AI Persona — "Digital Me"

A zero-human-in-the-loop AI persona of **Bhuvanesh**, grounded in his real resume
and GitHub repos. Recruiters can chat with it (and later, call it) and book an
interview. Built for the SCALER AI Engineer screening assignment.

> Full design rationale, stack comparison, and architecture diagrams: see [PRD.md](PRD.md).

**Current status:** Stage 1 — **Ingestion + RAG core** (chat track). Voice + booking come next.

---

## Architecture (this stage)

```
Resume (PDF) ┐
             ├─→ ingest ─→ chunk ─→ embed (voyage-3) ─→ Supabase pgvector
GitHub repos ┘   (26 original + 10 forks, READMEs · source · commit logs)
                                                              │
Recruiter ──→ FastAPI /chat ──→ hybrid retrieve (vector+FTS, RRF) ──┘
                                        │
                                        └─→ Claude Opus 4.8 (grounded, cited,
                                            fork-honest, injection-resistant)
```

- **`is_fork` tagging** — forked OSS repos (hyperswitch, apidash, forge, mcp-gateway, …)
  are marked so the persona never claims it built them.
- **Commit logs** are ingested so questions answerable only from commit history work.

---

## Prerequisites

- Python 3.11+
- A Supabase project (free tier) — gives you a Postgres DB with `pgvector`.
- API keys (see below).

## Setup

```bash
# 1. install deps (a virtualenv is recommended)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# 2. configure secrets
copy .env.example .env             # Windows  (cp on macOS/Linux)
#   …then fill in .env (see "How to get each key" below)

# 3. add your resume
#   drop your resume at  data/resume.pdf   (or data/resume.txt)

# 4. ingest the corpus  (resume + all GitHub repos + commits → pgvector)
python -m app.ingest.run_ingest --fresh

# 5. test the RAG core from the terminal
python -m app.ask "Why are you a good fit for an AI engineer role?"
python -m app.ask "Did you build hyperswitch?"          # should NOT claim authorship
python -m app.ask "What does your rag-chatbot repo do?"

# 6. run the chat API
uvicorn app.api:app --reload
#   GET  http://localhost:8000/health
#   POST http://localhost:8000/chat        {"message": "..."}
#   POST http://localhost:8000/chat/stream (SSE)
```

---

## How to get each key

Fill these into `.env`.

### 1. `ANTHROPIC_API_KEY` (the LLM brain)
- Go to **console.anthropic.com** → sign in → **Settings → API Keys → Create Key**.
- Copy the `sk-ant-…` value. Add a few dollars of credit under **Billing**.

### 2. `GITHUB_TOKEN` (corpus ingestion)
- **github.com → Settings → Developer settings → Personal access tokens**.
- Either a **fine-grained** token (Repository access: *Public repositories (read-only)*)
  or a **classic** token with the `public_repo` scope is enough.
- Without a token you're limited to 60 requests/hour (not enough for a full ingest);
  with one you get 5000/hour.

### 3. Embeddings — pick ONE (set `EMBEDDING_PROVIDER` accordingly)
- **`voyage`** (default, recommended): **dashboard.voyageai.com** → API Keys → create →
  copy `pa-…` into `VOYAGE_API_KEY`. Model used: `voyage-3` (1024-dim).
- **`openai`** (alternative): **platform.openai.com → API keys** → create → copy `sk-…`
  into `OPENAI_API_KEY`. Model used: `text-embedding-3-large` (forced to 1024-dim).

### 4. `DATABASE_URL` (Supabase Postgres + pgvector)
- Create a project at **supabase.com** (free tier).
- **Project Settings → Database → Connection string → URI**. Copy it and replace
  `[YOUR-PASSWORD]` with your DB password.
- Format: `postgresql://postgres:PASSWORD@db.xxxx.supabase.co:5432/postgres`
- `pgvector` is auto-enabled by the app (`CREATE EXTENSION IF NOT EXISTS vector`).

---

## Project layout

```
app/
├── config.py            # env-driven settings
├── db.py                # pgvector connection + schema
├── embeddings.py        # voyage / openai (pluggable, 1024-dim)
├── ingest/
│   ├── github_loader.py # repos, READMEs, source files, commit logs, fork tagging
│   ├── resume_loader.py # PDF/txt resume
│   ├── chunker.py       # structure-aware chunking
│   └── run_ingest.py    # orchestrator (python -m app.ingest.run_ingest)
├── rag/
│   ├── retriever.py     # hybrid vector + full-text, RRF fusion
│   └── answer.py        # grounded answer w/ citations + fork-honesty + injection guard
├── ask.py               # terminal test harness
└── api.py               # FastAPI: /health, /chat, /chat/stream
```

### 5. Cal.com (`CALCOM_API_KEY` + `CALCOM_EVENT_TYPE_ID`)
- Create an account at **cal.com** (free tier).
- **Settings → Security → API Keys → Add** — copy the `cal_live_…` key.
- Create a new event type (e.g. "30-min Intro Call"), connect it to your Google Calendar.
- Open the event type; the URL ends with `/event-types/<ID>` — that integer is your `CALCOM_EVENT_TYPE_ID`.

---

## Voice track (Part A)

After filling in all keys and running ingest, the voice webhook is live at:

```
POST http://localhost:8000/voice/retell
```

**Retell setup** (do after deploying to cloud with a public URL):
1. Sign up at **retell.ai** → **Create Agent → Custom LLM**.
2. Set **LLM WebSocket URL** → `https://your-app.onrender.com/voice/retell`.
3. Under **Phone Numbers → Import** → connect a Twilio number (or buy one in Retell directly).
4. Attach the phone number to the agent. That number is your callable line.

**What happens on a call:**
```
Caller dials → Retell (STT + turn-taking) → POST /voice/retell
    → RAG retrieve → Haiku 4.5 (+ Cal.com tools) → SSE text back
    → Retell (TTS) → Caller hears spoken answer
```

---

## Cost note (this stage)
One full ingest embeds a few hundred–thousand short chunks (cents). Each chat turn is a
single Opus 4.8 call with a cached system prompt (~$0.02–0.10). See [PRD.md §10](PRD.md).
