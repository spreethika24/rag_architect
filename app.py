"""
RAG Architect — Multi-agent RAG architecture recommender with human-in-the-loop review and RAGAS validation.
Gradio front-end. Run locally with `python app.py` or deploy to HF Spaces.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, List

import gradio as gr
import pandas as pd

try:  # load backend config from an untracked .env next to this file, if present
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from ingestion import ingest_files, build_corpus_summary
from agents import (
    run_pipeline, node_embedding, node_chunking, node_tokenizer, node_indexing,
    GROQ_MODELS, DOMAINS, CHUNKING_OPTIONS, TOKENIZER_OPTIONS, INDEX_OPTIONS,
)
from consensus import build_recommendations, build_comparison_table

MAX_EVAL_CHARS = 200_000  # upper bound on document text kept for the RAGAS check

# ---------------------------------------------------------------------------
# Environment-based configuration
#
# Keys are read from environment variables ONLY -- never hardcode real keys
# into a source file that gets pushed to the Space repo. On Hugging Face
# Spaces, set these under Settings -> "Repository secrets": they're
# encrypted, never shown in the Files tab, and never committed to git.
# Locally, set them in your shell (export GROQ_API_KEY=...) or an
# untracked .env file loaded before launch -- never commit that file either.
#
#   GROQ_API_KEY          (required for agent recommendations)
#   LANGCHAIN_API_KEY      or  LANGSMITH_API_KEY   (either name works)
#   LANGCHAIN_TRACING_V2  ("true" to enable tracing by default)
#   LANGCHAIN_PROJECT     (defaults to "agentbench-ai" if unset)
# ---------------------------------------------------------------------------
ENV_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
ENV_LANGCHAIN_API_KEY = (os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY", "")).strip()
ENV_LANGCHAIN_TRACING_DEFAULT = (
    os.environ.get("LANGCHAIN_TRACING_V2", "") or os.environ.get("LANGSMITH_TRACING", "")
).lower() == "true"
ENV_LANGCHAIN_PROJECT = (os.environ.get("LANGCHAIN_PROJECT") or os.environ.get("LANGSMITH_PROJECT") or "agentbench-ai").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "").strip() or GROQ_MODELS[0]
LANGSMITH_ON = ENV_LANGCHAIN_TRACING_DEFAULT and bool(ENV_LANGCHAIN_API_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec_to_df(rec: Dict[str, Any]) -> pd.DataFrame:
    if not rec:
        return pd.DataFrame()
    rows = []
    for c in rec["candidates"]:
        rows.append({
            "Candidate": c["name"],
            "Retrieval": round(c["retrieval_score"], 2),
            "Latency": round(c["latency_score"], 2),
            "Accuracy": round(c["accuracy_score"], 2),
            "Top Choice": "⭐" if c["name"] == rec["top_choice"] else "",
            "Rationale": c["rationale"],
        })
    return pd.DataFrame(rows)


def _usage_text(usage: Dict[str, int]) -> str:
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    parts = [f"{tokens:,} tokens"]
    if usage.get("cache_hits"):
        parts.append(f"{usage['cache_hits']} answer(s) reused from cache")
    if usage.get("retries"):
        parts.append(f"{usage['retries']} retry(ies)")
    if usage.get("fallbacks"):
        parts.append(f"switched model {usage['fallbacks']}×")
    return " · ".join(parts)


def _degraded_warning(degraded: List[str]) -> str:
    if not degraded:
        return ""
    return (f"⚠️ **{', '.join(degraded)} agent(s) could not get a usable answer from the LLM** "
            "and show placeholder 0.5 scores. Reject them below and use **Re-run Rejected Agents**.")


def _maybe_enable_langsmith(enable: bool, api_key: str, project: str):
    if enable and api_key:
        api_key = api_key.strip()
        project = (project or "agentbench-ai").strip()
        for ns in ("LANGCHAIN", "LANGSMITH"):
            os.environ[f"{ns}_API_KEY"] = api_key
            os.environ[f"{ns}_PROJECT"] = project
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
    else:
        for name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
            os.environ.pop(name, None)
    # langsmith memoizes env lookups; without this, a value read before the
    # checkbox was ticked (or before the key was entered) keeps being used.
    try:
        from langsmith.utils import get_env_var
        get_env_var.cache_clear()
    except Exception:
        pass


def _format_recommendations(result: Dict[str, Any]) -> "tuple[pd.DataFrame, str]":
    df = pd.DataFrame(result["recommendations"])
    if not df.empty:
        df.index = [f"Recommendation {i + 1}" for i in range(len(df))]

    summary_lines = []
    for i, rec in enumerate(result["recommendations"], start=1):
        summary_lines.append(
            f"**Recommendation {i}** — Score: {rec['final_score']:.3f}  \n"
            f"Embedding: `{rec['Embedding']}` | Chunking: `{rec['Chunking']}` | "
            f"Tokenizer: `{rec['Tokenizer']}` | Indexing: `{rec['Indexing']}`"
        )
    final_md = "\n\n".join(summary_lines)
    final_md += f"\n\n_Considered {result['all_combos_considered']} combinations._"
    return df, final_md


def _comparison_df(state: Dict[str, Any], chk_lock: str, tok_lock: str, idx_lock: str) -> pd.DataFrame:
    locks = {}
    if chk_lock and chk_lock != "All":
        locks["Chunking"] = chk_lock
    if tok_lock and tok_lock != "All":
        locks["Tokenizer"] = tok_lock
    if idx_lock and idx_lock != "All":
        locks["Indexing"] = idx_lock

    table = build_comparison_table(
        state["embedding_rec"], state["chunking_rec"],
        state["tokenizer_rec"], state["indexing_rec"],
        approvals=state.get("approvals", {}),
        locks=locks,
    )
    df = pd.DataFrame(table)
    if not df.empty:
        df.index = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Step 1: Ingest + domain detect + run 4 agents in parallel
# ---------------------------------------------------------------------------

def run_analysis(files, domain_override):
    groq_api_key, model = ENV_GROQ_API_KEY, GROQ_MODEL
    if not groq_api_key:
        raise gr.Error("GROQ_API_KEY is not configured on the server. Set it in .env or the environment and restart.")
    if not files:
        raise gr.Error("Please upload at least one file (PDF, CSV, TXT, or image).")

    _maybe_enable_langsmith(LANGSMITH_ON, ENV_LANGCHAIN_API_KEY, ENV_LANGCHAIN_PROJECT)

    paths = [f.name if hasattr(f, "name") else f for f in files]
    ingested = ingest_files(paths)
    corpus_text = build_corpus_summary(ingested)

    override = domain_override if domain_override and domain_override != "Auto-detect" else None

    result = run_pipeline(corpus_text, groq_api_key, model=model, domain_override=override)

    # corpus_text is capped for the LLM prompts; the optional RAGAS check needs the whole document.
    full_text = "\n\n".join(d["text"] for d in ingested if d["modality"] != "error")[:MAX_EVAL_CHARS]

    state = {
        "corpus_text": corpus_text,
        "full_text": full_text,
        "api_key": groq_api_key,
        "model": model,
        "domain": result["domain"],
        "embedding_rec": result["embedding_rec"],
        "chunking_rec": result["chunking_rec"],
        "tokenizer_rec": result["tokenizer_rec"],
        "indexing_rec": result["indexing_rec"],
        "approvals": {},
    }

    override_note = " (manually overridden)" if override else " (auto-detected)"
    domain_md = (
        f"### Detected Domain: **{result['domain']}**{override_note}  \n"
        f"Confidence: {result.get('domain_confidence', 0):.2f}  \n"
        f"**Domain Explanation:** _{result.get('domain_reasoning', '')}_  \n"
        f"{_usage_text(result.get('usage', {}))}"
    )
    warning = _degraded_warning(result.get("degraded", []))
    if not override and result.get("domain_confidence", 0) == 0:
        warning += ("\n\n⚠️ **Domain detection failed**, so the domain below is a default guess, not a "
                    "detection. Use Domain Override to set it yourself.")
    if warning:
        domain_md += "\n\n" + warning

    ingestion_md = "**Files processed:** " + ", ".join(
        f"{d['filename']} ({d['modality']})" for d in ingested
    )

    comparison_df = _comparison_df(state, "All", "All", "All")

    return (
        json.dumps(state),
        domain_md,
        ingestion_md,
        _rec_to_df(state["embedding_rec"]),
        _rec_to_df(state["chunking_rec"]),
        _rec_to_df(state["tokenizer_rec"]),
        _rec_to_df(state["indexing_rec"]),
        gr.update(visible=True),   # HITL section
        gr.update(visible=False),  # final results section (hidden until approved)
        "",
        comparison_df,
        gr.update(value="All"),  # reset chunking lock
        gr.update(value="All"),  # reset tokenizer lock
        gr.update(value="All"),  # reset indexing lock
    )


# ---------------------------------------------------------------------------
# Step 2: HITL submission — approve/reject each agent
# ---------------------------------------------------------------------------

def submit_hitl(emb_decision, chk_decision, tok_decision, idx_decision,
                 emb_fb, chk_fb, tok_fb, idx_fb, state_json):
    if not state_json:
        raise gr.Error("Run analysis first.")
    state = json.loads(state_json)

    decisions = {
        "Embedding": emb_decision == "Approve",
        "Chunking": chk_decision == "Approve",
        "Tokenizer": tok_decision == "Approve",
        "Indexing": idx_decision == "Approve",
    }
    feedback = {
        "Embedding": emb_fb, "Chunking": chk_fb,
        "Tokenizer": tok_fb, "Indexing": idx_fb,
    }
    state["approvals"] = decisions
    state["feedback"] = feedback

    rejected = [k for k, v in decisions.items() if not v]

    if rejected:
        status_md = (
            f"**{len(rejected)} agent(s) rejected:** {', '.join(rejected)}. "
            f"Add optional feedback above and click **Re-run Rejected Agents**."
        )
        return (
            json.dumps(state),
            status_md,
            gr.update(visible=True),   # re-run button
            gr.update(visible=False),  # final results
            pd.DataFrame(),
            "",
            gr.update(),  # comparison table unchanged
        )

    # All approved -> run consensus engine
    result = build_recommendations(
        state["embedding_rec"], state["chunking_rec"],
        state["tokenizer_rec"], state["indexing_rec"],
        decisions,
    )
    df, final_md = _format_recommendations(result)
    comparison_df = _comparison_df(state, "All", "All", "All")

    status_md = "✅ All 4 agents approved by human reviewer. Consensus engine ran successfully."

    return (
        json.dumps(state),
        status_md,
        gr.update(visible=False),  # re-run button
        gr.update(visible=True),   # final results section
        df,
        final_md,
        comparison_df,
    )


# ---------------------------------------------------------------------------
# Step 3: Re-run only the rejected agents with feedback, then loop back to HITL
# ---------------------------------------------------------------------------

NODE_MAP = {
    "Embedding": node_embedding,
    "Chunking": node_chunking,
    "Tokenizer": node_tokenizer,
    "Indexing": node_indexing,
}
STATE_KEY_MAP = {
    "Embedding": "embedding_rec",
    "Chunking": "chunking_rec",
    "Tokenizer": "tokenizer_rec",
    "Indexing": "indexing_rec",
}


def rerun_rejected(state_json):
    if not state_json:
        raise gr.Error("Nothing to re-run yet.")
    state = json.loads(state_json)
    if "approvals" not in state:
        raise gr.Error("Nothing to re-run yet.")

    rejected = [k for k, v in state["approvals"].items() if not v]
    if not rejected:
        raise gr.Error("No rejected agents to re-run.")

    graph_state = {
        "corpus_text": state["corpus_text"],
        "api_key": state["api_key"],
        "model": state["model"],
        "domain": state["domain"],
        "feedback": state.get("feedback", {}),
        "bypass_cache": True,  # a re-run must produce a fresh answer, not replay the rejected one
    }

    for cat in rejected:
        updated = NODE_MAP[cat](dict(graph_state))
        state[STATE_KEY_MAP[cat]] = updated[STATE_KEY_MAP[cat]]

    status_md = f"🔁 Re-ran: {', '.join(rejected)}. Review the updated recommendations below and re-submit HITL decisions."
    still_degraded = [c for c in rejected if state[STATE_KEY_MAP[c]].get("degraded")]
    if still_degraded:
        status_md += "\n\n" + _degraded_warning(still_degraded)
    comparison_df = _comparison_df(state, "All", "All", "All")

    return (
        json.dumps(state),
        status_md,
        _rec_to_df(state["embedding_rec"]),
        _rec_to_df(state["chunking_rec"]),
        _rec_to_df(state["tokenizer_rec"]),
        _rec_to_df(state["indexing_rec"]),
        gr.update(visible=False),  # hide re-run button until next rejection
        comparison_df,
    )


# ---------------------------------------------------------------------------
# Step 4: Post-hoc filtering — lock Chunking/Tokenizer/Indexing and re-rank
# WITHOUT any new LLM calls (Embedding is intentionally not lockable here;
# see consensus.py docstring for why).
# ---------------------------------------------------------------------------

def filter_comparison_only(chk_lock, tok_lock, idx_lock, state_json):
    """
    Live filter for the Comparison tab only — no approval gate, since this
    is informational (browse the full search space), not the official
    Top-2 recommendation. Zero new LLM calls either way.
    """
    if not state_json:
        return pd.DataFrame()
    state = json.loads(state_json)
    if "embedding_rec" not in state:
        return pd.DataFrame()
    return _comparison_df(state, chk_lock, tok_lock, idx_lock)


def recompute_with_locks(chk_lock, tok_lock, idx_lock, state_json):
    if not state_json:
        raise gr.Error("Run analysis and get full HITL approval first.")
    state = json.loads(state_json)
    if not state.get("approvals") or not all(state["approvals"].values()):
        raise gr.Error("All 4 agents must be approved before filtering the recommendations.")

    locks = {}
    if chk_lock and chk_lock != "All":
        locks["Chunking"] = chk_lock
    if tok_lock and tok_lock != "All":
        locks["Tokenizer"] = tok_lock
    if idx_lock and idx_lock != "All":
        locks["Indexing"] = idx_lock

    result = build_recommendations(
        state["embedding_rec"], state["chunking_rec"],
        state["tokenizer_rec"], state["indexing_rec"],
        state["approvals"],
        locks=locks,
    )
    df, final_md = _format_recommendations(result)
    comparison_df = _comparison_df(state, chk_lock, tok_lock, idx_lock)

    lock_desc = ", ".join(f"{k}={v}" for k, v in locks.items()) or "none"
    status_md = f"🔓 Recomputed with locks: {lock_desc} — zero new LLM calls, pure re-ranking."

    return df, final_md, comparison_df, status_md


# ---------------------------------------------------------------------------
# Step 5 (optional): validate the Embedding / Chunking picks with RAGAS.
# Slow and uses LLM tokens, so it only runs when the button is clicked.
# ---------------------------------------------------------------------------

EVAL_COLUMNS = ["Option", "Chunks", "Context recall", "Context precision", "Measured score",
                "Measured rank", "AI retrieval score", "AI rank"]


def _eval_df(cmp_result: Dict[str, Any]) -> pd.DataFrame:
    if not cmp_result:
        return pd.DataFrame(columns=EVAL_COLUMNS)
    rows = [{
        "Option": r["option"],
        "Chunks": r.get("n_chunks"),
        "Context recall": r.get("context_recall"),
        "Context precision": r.get("context_precision"),
        "Measured score": r.get("measured"),
        "Measured rank": r.get("measured_rank"),
        "AI retrieval score": r.get("ai_retrieval_score"),
        "AI rank": r.get("ai_rank"),
    } for r in cmp_result["rows"]]
    return pd.DataFrame(rows, columns=EVAL_COLUMNS)


def _verdict(label: str, cmp_result: Dict[str, Any]) -> str:
    if not cmp_result:
        return f"**{label}:** could not be evaluated."
    if cmp_result["measured_winner"] is None:
        return f"**{label}:** no option could be scored."
    if cmp_result.get("tie"):
        mark = "✅ AI's pick is among them" if cmp_result["agrees"] else "⚠️ AI's pick is not among them"
        return (f"**{label}:** AI picked `{cmp_result['ai_top_pick']}`; measured scores tied "
                f"between `{cmp_result['measured_winner']}` — {mark}")
    mark = "✅ agrees" if cmp_result["agrees"] else "⚠️ differs"
    return (f"**{label}:** AI picked `{cmp_result['ai_top_pick']}`, "
            f"measured winner is `{cmp_result['measured_winner']}` — {mark}")


def run_ragas_validation(n_questions, state_json, progress=gr.Progress()):
    if not state_json:
        raise gr.Error("Run analysis first.")
    state = json.loads(state_json)
    if "embedding_rec" not in state:
        raise gr.Error("Run analysis first.")

    from evaluation import run_evaluation  # lazy: needs the optional RAGAS packages
    try:
        result = run_evaluation(
            state.get("full_text") or state["corpus_text"], state["api_key"], state["model"],
            state["embedding_rec"], state["chunking_rec"], n_questions=int(n_questions),
            progress=lambda frac, msg: progress(frac, desc=msg),
        )
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e))

    summary = [
        f"Tested with **{result['n_questions']}** generated questions on a {result['n_words']}-word document.",
        _verdict("Embedding", result["embedding"]),
        _verdict("Chunking", result["chunking"]),
        f"_Embedding was compared with `{result['chunking_held_fixed']}` chunking; "
        f"chunking was compared using `{result['embedding_used_for_chunking_test']}`._",
    ]
    if result["notes"]:
        summary.append("**Notes**\n" + "\n".join(f"- {n}" for n in result["notes"]))
    return "\n\n".join(summary), _eval_df(result["embedding"]), _eval_df(result["chunking"])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAG Architect") as demo:
    gr.Markdown(
        "# 🧭 RAG Architect\n"
        "### Multi-agent RAG architecture recommender with human-in-the-loop review and RAGAS validation.\n\n"
        "Upload enterprise "
        "documents, detect the domain, run 4 parallel specialist agents, approve "
        "via Human-in-the-Loop, and get the Top-2 recommended RAG architecture stacks."
    )

    app_state = gr.State("")

    with gr.Tabs():
        with gr.Tab("🚀 Recommender"):
            gr.Markdown("### 1️⃣ Upload documents (PDF, CSV, TXT, or images)")
            files_in = gr.File(file_count="multiple", label="Enterprise documents")
            domain_override_dd = gr.Dropdown(
                ["Auto-detect"] + DOMAINS, value="Auto-detect",
                label="Domain Override — forces a full pipeline re-run with this domain (new LLM calls)",
            )
            run_btn = gr.Button("🚀 Run Domain Detection + 4 Parallel Agents", variant="primary")

            domain_md = gr.Markdown()
            ingestion_md = gr.Markdown()

            with gr.Group(visible=False) as hitl_section:
                gr.Markdown("### 2️⃣ Human-in-the-Loop: Approve or Reject each agent's recommendation")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Embedding Agent**")
                        emb_df = gr.Dataframe(interactive=False, wrap=True)
                        emb_decision = gr.Radio(["Approve", "Reject"], value="Approve", label="Decision")
                        emb_fb = gr.Textbox(label="Feedback (if rejected)", lines=1)
                    with gr.Column():
                        gr.Markdown("**Chunking Agent**")
                        chk_df = gr.Dataframe(interactive=False, wrap=True)
                        chk_decision = gr.Radio(["Approve", "Reject"], value="Approve", label="Decision")
                        chk_fb = gr.Textbox(label="Feedback (if rejected)", lines=1)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Tokenizer Agent**")
                        tok_df = gr.Dataframe(interactive=False, wrap=True)
                        tok_decision = gr.Radio(["Approve", "Reject"], value="Approve", label="Decision")
                        tok_fb = gr.Textbox(label="Feedback (if rejected)", lines=1)
                    with gr.Column():
                        gr.Markdown("**Indexing Agent**")
                        idx_df = gr.Dataframe(interactive=False, wrap=True)
                        idx_decision = gr.Radio(["Approve", "Reject"], value="Approve", label="Decision")
                        idx_fb = gr.Textbox(label="Feedback (if rejected)", lines=1)

                hitl_submit_btn = gr.Button("✅ Submit HITL Decisions")
                hitl_status_md = gr.Markdown()
                rerun_btn = gr.Button("🔁 Re-run Rejected Agents", visible=False)

            with gr.Group(visible=False) as final_section:
                gr.Markdown("### 3️⃣ Top-2 Recommended RAG Architectures (Consensus Engine)")
                final_df = gr.Dataframe(interactive=False, wrap=True)
                final_md = gr.Markdown()

                gr.Markdown(
                    "#### 🔓 Post-hoc filters — free, zero new LLM calls\n"
                    "Lock a category to a specific candidate and recompute the Top-2 instantly "
                    "by re-ranking the already-scored candidates. Embedding isn't lockable here "
                    "since it drives the domain-specific retrieval estimate most directly — "
                    "changing it should go through a full re-run via Domain Override above instead."
                )
                with gr.Row():
                    chk_lock_dd = gr.Dropdown(["All"] + CHUNKING_OPTIONS, value="All", label="Lock Chunking")
                    tok_lock_dd = gr.Dropdown(["All"] + TOKENIZER_OPTIONS, value="All", label="Lock Tokenizer")
                    idx_lock_dd = gr.Dropdown(["All"] + INDEX_OPTIONS, value="All", label="Lock Indexing")
                recompute_btn = gr.Button("🔓 Recompute Top-2 (no new LLM calls)")
                recompute_status_md = gr.Markdown()

            with gr.Accordion("🧪 Validate with RAGAS (optional — slow, uses LLM tokens)", open=False):
                gr.Markdown(
                    "The agents' scores are an LLM's judgement. This **measures** the Embedding and "
                    "Chunking picks on your own document: it writes test questions, retrieves with each "
                    "option, and scores the results with RAGAS (context recall + precision). Takes a few "
                    "minutes, downloads embedding models on first use, and needs a document of 300+ words. "
                    "Requires `pip install -r requirements-eval.txt`."
                )
                eval_questions = gr.Slider(3, 10, value=5, step=1, label="Number of test questions")
                eval_btn = gr.Button("🧪 Validate with RAGAS")
                eval_summary_md = gr.Markdown()
                gr.Markdown("**Embedding options**")
                eval_emb_df = gr.Dataframe(headers=EVAL_COLUMNS, interactive=False, wrap=True)
                gr.Markdown("**Chunking options**")
                eval_chk_df = gr.Dataframe(headers=EVAL_COLUMNS, interactive=False, wrap=True)

        with gr.Tab("📊 Comparison — all considered combinations"):
            gr.Markdown(
                "Browse every combination the Consensus Engine considered (not just the "
                "Top-2), ranked by final score. Use the locks below to filter instantly — "
                "this reads cached agent scores only, no new LLM calls, and works even "
                "before all 4 agents are approved."
            )
            with gr.Row():
                cmp_chk_dd = gr.Dropdown(["All"] + CHUNKING_OPTIONS, value="All", label="Filter: Chunking")
                cmp_tok_dd = gr.Dropdown(["All"] + TOKENIZER_OPTIONS, value="All", label="Filter: Tokenizer")
                cmp_idx_dd = gr.Dropdown(["All"] + INDEX_OPTIONS, value="All", label="Filter: Indexing")
            comparison_df_out = gr.Dataframe(interactive=False, wrap=True)

    run_btn.click(
        run_analysis,
        inputs=[files_in, domain_override_dd],
        outputs=[app_state, domain_md, ingestion_md, emb_df, chk_df, tok_df, idx_df,
                 hitl_section, final_section, hitl_status_md,
                 comparison_df_out, cmp_chk_dd, cmp_tok_dd, cmp_idx_dd],
    )

    hitl_submit_btn.click(
        submit_hitl,
        inputs=[emb_decision, chk_decision, tok_decision, idx_decision,
                emb_fb, chk_fb, tok_fb, idx_fb, app_state],
        outputs=[app_state, hitl_status_md, rerun_btn, final_section, final_df, final_md, comparison_df_out],
    )

    rerun_btn.click(
        rerun_rejected,
        inputs=[app_state],
        outputs=[app_state, hitl_status_md, emb_df, chk_df, tok_df, idx_df, rerun_btn, comparison_df_out],
    )

    recompute_btn.click(
        recompute_with_locks,
        inputs=[chk_lock_dd, tok_lock_dd, idx_lock_dd, app_state],
        outputs=[final_df, final_md, comparison_df_out, recompute_status_md],
    )

    eval_btn.click(
        run_ragas_validation,
        inputs=[eval_questions, app_state],
        outputs=[eval_summary_md, eval_emb_df, eval_chk_df],
    )

    gr.on(
        triggers=[cmp_chk_dd.change, cmp_tok_dd.change, cmp_idx_dd.change],
        fn=filter_comparison_only,
        inputs=[cmp_chk_dd, cmp_tok_dd, cmp_idx_dd, app_state],
        outputs=[comparison_df_out],
    )


if __name__ == "__main__":
    demo.launch()
