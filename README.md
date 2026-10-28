---
title: RAG Architect
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
short_description: Find the best RAG stack for your documents
python_version: "3.12"
---

# RAG Architect

**Upload your documents and get the best RAG architecture for them, reviewed by a human before it is final.**

Choosing a RAG stack (embedding model, chunking, tokenizer, index) is guesswork. This app reads a
document, detects its domain, has four specialist LLM agents score the options in parallel, lets a human
approve or reject each agent, and returns the Top-2 stacks. An optional RAGAS check then *measures* the
embedding and chunking picks on the same document.

## The flow

```
Upload (PDF / CSV / TXT / image)
  → domain detection
  → 4 agents in parallel: Embedding | Chunking | Tokenizer | Indexing      (LangGraph)
  → human Approve / Reject per agent     (a rejection re-runs only that agent, with the reviewer's note)
  → consensus score over all 81 combinations → Top-2 stacks
  → optional: RAGAS validation of the Embedding and Chunking picks
```

`final_score = 0.4·retrieval + 0.2·latency + 0.2·human approval + 0.2·accuracy`, with a deterministic
tiebreak so identical inputs always rank identically.

## What to look at

| Design choice | Where |
|---|---|
| **Parallel agents** via LangGraph fan-out, joined before the human step | `agents.py` (`build_graph`) |
| **Human-in-the-loop gate**: no recommendation until all four agents are approved; rejection feedback goes into a targeted re-run | `app.py`, `agents.py` |
| **Reliable LLM calls**: retry with backoff, model fallback, JSON validation that re-asks with the error, response cache, token counts. A failed agent degrades to flagged placeholder scores instead of crashing the run | `agents.py` |
| **Free filtering**: lock Chunking / Tokenizer / Indexing and re-rank with zero LLM calls, and browse all 81 combinations | `consensus.py` |
| **Honest scoring**: the agents' scores are LLM estimates, so RAGAS measures the Embedding and Chunking picks and reports whether the agents agree | `evaluation.py` |
| **107 unit tests** that use fakes, so they need no API key or network | `tests/` |

Full reasoning, trade-offs and known gaps: [docs/RAG_ARCHITECT_DESIGN_DOCUMENT.md](docs/RAG_ARCHITECT_DESIGN_DOCUMENT.md).

## Run it

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then set GROQ_API_KEY
python app.py
```

Upload a file from `dataset/` and click **Run**. Image OCR needs the Tesseract binary installed.
For the RAGAS check: `pip install -r requirements-eval.txt`. Tests: `pytest`.

## Deploy to Hugging Face Spaces

1. Create a new Space with SDK **Gradio** and push this folder (the README front matter above configures it).
   `.gitignore` keeps `.env` and `venv/` out of the push.
2. In **Settings → Variables and secrets**, add the secret `GROQ_API_KEY`. Optionally set `GROQ_MODEL`,
   `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` (see `.env.example`).
3. `packages.txt` installs the Tesseract binary for image OCR; `requirements.txt` installs the Python dependencies.
   The RAGAS check needs `requirements-eval.txt` (heavy), so it is off on the default Space; copy its
   lines into `requirements.txt` to enable it.

## Honest limitations

- Retrieval, latency and accuracy scores are LLM judgement, not benchmarks. Only Embedding and Chunking can be measured (RAGAS).
- No hard per-call LLM timeout yet, and HITL decisions are not logged to LangSmith.
- The API key is server-side (`.env` or Space secret), so on a public Space every visitor uses that quota.
