# Jarvis Research Agent

Fully local, open-source AI research agent. No proprietary APIs.

## Stack
- **Model**: Ollama (Gemma 3 4B Q4 — fits 4 GB VRAM)
- **Search**: SearXNG (self-hosted Docker)
- **Extraction**: Crawl4AI + Playwright fallback
- **PDF**: PyMuPDF
- **Backend**: FastAPI (SSE streaming)
- **DB**: SQLite + filesystem cache

## Quick start

```bash
# 1. Clone and enter the project
cd jarvis

# 2. Install Python deps
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/playwright install chromium

# 3. Copy and edit env
cp .env.example .env

# 4. Start SearXNG
docker compose up -d

# 5. Start Ollama and pull model
# (Assumes Ollama is installed)
ollama pull gemma3:4b-it-q4_K_M

# 6. Create data dirs
mkdir -p data/cache data/chroma

# 7. Run the agent
./venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## Changing the model

Edit `.env`:
```
OLLAMA_MODEL=phi3:mini-4k-instruct-q4_K_M
```
Then restart the server. No code changes needed.

## Token budget (4 GB VRAM)

| Parameter | Value | Why |
|---|---|---|
| MAX_TOKENS_PER_SOURCE | 1500 | Fits 4 sources in context |
| MAX_SOURCES_PER_ROUND | 4 | Protects RAM |
| MAX_REACT_ITERATIONS | 8 | Prevents infinite loops |
| MAX_CONTEXT_TOKENS | 6000 | Safe for 4B models |

## Project structure

```
jarvis/
├── agent/          # Orchestrator, planner, synthesizer, memory
├── tools/          # Search, crawl, browser, PDF
├── llm/            # Ollama client
├── store/          # SQLite + filesystem cache
├── api/            # FastAPI app
├── frontend/       # Single-file HTML UI
└── data/           # Runtime data (gitignored)
```
