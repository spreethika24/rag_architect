"""
evaluation.py
OPTIONAL measured check of the agents' Embedding and Chunking picks, using RAGAS.

The 4 agents score their options from an LLM's judgement. This module tests
that judgement on the user's own document:

  1. An LLM writes a few questions (with reference answers) from the document.
  2. For every Embedding option (then every Chunking option) we chunk the
     document, embed it, retrieve the top-k chunks per question.
  3. RAGAS scores the retrieved chunks: context recall + context precision.
  4. The measured winner is compared with the agent's top pick.

It is slow and uses LLM tokens, so app.py only runs it when the user clicks
"Validate with RAGAS". Heavy packages are imported lazily so the normal app
starts without them:  pip install -r requirements-eval.txt

Tokens are approximated by whitespace-separated words (no tokenizer needed).
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agents import (
    CHUNKING_OPTIONS, EMBEDDING_OPTIONS, _safe_json_parse, call_llm, get_llm,
)

TOP_K = 3
SEMANTIC_THRESHOLD = 0.80
SEMANTIC_MAX_WORDS = 512

EMBEDDING_MODEL_IDS = {
    "BGE Large": "BAAI/bge-large-en-v1.5",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
}
# Models that need extra text in front of queries / documents to work well.
QUERY_PREFIX = {
    "BGE Large": "Represent this sentence for searching relevant passages: ",
    "nomic-embed-text": "search_query: ",
}
DOC_PREFIX = {"nomic-embed-text": "search_document: "}

# Embedding is compared with this chunking held fixed; chunking is compared with
# the best measured embedding held fixed.
BASELINE_CHUNKING = "Fixed (512/100)"

EmbedFn = Callable[[str, List[str], str], np.ndarray]  # (option, texts, "query"|"document") -> matrix


# ---------------------------------------------------------------------------
# Chunking (plain Python, no framework needed)
# ---------------------------------------------------------------------------

def _words(text: str) -> List[str]:
    return text.split()


def fixed_chunks(text: str, size: int = 512, overlap: int = 100) -> List[str]:
    """Sliding window of `size` words, moving forward by size - overlap."""
    words = _words(text)
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def recursive_chunks(text: str, size: int = 1024, overlap: int = 200) -> List[str]:
    """
    Split on the biggest natural boundary that fits (paragraph -> line ->
    sentence -> word), then pack pieces up to `size` words, carrying `overlap`
    words from the end of one chunk into the next.
    """
    separators = ["\n\n", "\n", ". "]

    def split(piece: str, seps: List[str]) -> List[str]:
        if len(_words(piece)) <= size:
            return [piece]
        if not seps:  # nothing left to split on: fall back to a hard word window
            return fixed_chunks(piece, size, 0)
        out: List[str] = []
        for part in piece.split(seps[0]):
            if part.strip():
                out.extend(split(part, seps[1:]))
        return out

    pieces = split(text, separators)
    chunks: List[str] = []
    current: List[str] = []
    for piece in pieces:
        pw = _words(piece)
        if current and len(current) + len(pw) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(pw)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def semantic_chunks(text: str, embed_sentences: Callable[[List[str]], np.ndarray],
                    threshold: float = SEMANTIC_THRESHOLD,
                    max_words: int = SEMANTIC_MAX_WORDS) -> List[str]:
    """Start a new chunk when a sentence is less similar than `threshold` to the one before it."""
    sents = _sentences(text)
    if not sents:
        return []
    vecs = embed_sentences(sents)  # rows are L2-normalised, so dot product == cosine
    chunks: List[str] = []
    current = [sents[0]]
    size = len(_words(sents[0]))
    for i in range(1, len(sents)):
        sim = float(np.dot(vecs[i - 1], vecs[i]))
        n = len(_words(sents[i]))
        if sim < threshold or size + n > max_words:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(sents[i])
        size += n
    chunks.append(" ".join(current))
    return chunks


def chunk_text(text: str, chunking: str, embed_fn: Optional[EmbedFn] = None,
               embedding: str = "all-mpnet-base-v2") -> List[str]:
    if chunking == "Fixed (512/100)":
        return fixed_chunks(text, 512, 100)
    if chunking == "Recursive (1024/200)":
        return recursive_chunks(text, 1024, 200)
    if chunking == "Semantic (0.80)":
        if embed_fn is None:
            raise ValueError("Semantic chunking needs an embedding function")
        return semantic_chunks(text, lambda s: embed_fn(embedding, s, "document"))
    raise ValueError(f"Unknown chunking option: {chunking}")


# ---------------------------------------------------------------------------
# Embedding + retrieval
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """Lazy-loading wrapper so models are only downloaded when actually evaluated."""

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}

    def __call__(self, option: str, texts: List[str], kind: str) -> np.ndarray:
        if option not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                ) from e
            model_id = EMBEDDING_MODEL_IDS[option]
            trust = os.environ.get("EVAL_TRUST_REMOTE_CODE", "").lower() == "true"
            if option == "nomic-embed-text" and not trust:
                # This model runs code downloaded from the Hugging Face repo.
                raise RuntimeError(
                    "nomic-embed-text runs remote code from Hugging Face. Set "
                    "EVAL_TRUST_REMOTE_CODE=true in .env to allow it; skipped otherwise."
                )
            self._models[option] = SentenceTransformer(model_id, trust_remote_code=trust)
        prefix = (QUERY_PREFIX if kind == "query" else DOC_PREFIX).get(option, "")
        vecs = self._models[option].encode([prefix + t for t in texts], normalize_embeddings=True,
                                           show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def retrieve(question: str, chunks: List[str], chunk_vecs: np.ndarray, option: str,
             embed_fn: EmbedFn, top_k: int = TOP_K) -> List[str]:
    qv = embed_fn(option, [question], "query")[0]
    scores = chunk_vecs @ qv
    order = np.argsort(-scores)[:top_k]
    return [chunks[i] for i in order]


# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

def _passages(text: str, n: int, window: int = 250) -> List[str]:
    """n evenly spaced word windows, so questions cover the whole document."""
    words = _words(text)
    if len(words) <= window:
        return [" ".join(words)] if words else []
    n = min(n, max(1, len(words) // window))
    starts = [int(i * (len(words) - window) / max(1, n - 1)) for i in range(n)] if n > 1 else [0]
    return [" ".join(words[s:s + window]) for s in starts]


def generate_questions(text: str, api_key: str, model: str, n: int = 5) -> List[Dict[str, str]]:
    """Ask the LLM for one question + reference answer per passage. Bad replies are skipped."""
    out = []
    for passage in _passages(text, n):
        prompt = f"""Read the passage and write ONE question that can be answered only from it,
plus the correct answer taken from the passage. Do not refer to "the passage" in the question.

Passage:
\"\"\"{passage}\"\"\"

Respond ONLY with JSON, no markdown fences: {{"question": "...", "answer": "..."}}"""
        try:
            text, _ = call_llm(api_key, model, prompt, temperature=0.3)  # retries + model fallback
            data = _safe_json_parse(text)
            q, a = str(data["question"]).strip(), str(data["answer"]).strip()
            if q and a:
                out.append({"question": q, "reference": a})
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# RAGAS scoring
# ---------------------------------------------------------------------------

def _nanmean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 3) if vals else None


def ragas_scores(samples: List[Dict[str, Any]], api_key: str, model: str) -> Dict[str, Optional[float]]:
    """
    Score retrieved contexts with RAGAS. Each sample:
        {"question": str, "reference": str, "contexts": [str, ...]}
    Returns the mean context_recall and context_precision.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
        from ragas.run_config import RunConfig
    except ImportError as e:
        raise RuntimeError("RAGAS is not installed or failed to import. Run: pip install ragas "
                           "(on Python 3.14 see the notes in requirements-eval.txt)") from e

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(user_input=s["question"], retrieved_contexts=s["contexts"], reference=s["reference"])
        for s in samples
    ])
    result = evaluate(
        dataset,
        metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],
        llm=LangchainLLMWrapper(get_llm(api_key, model, temperature=0.0)),
        # Few workers + generous timeout: Groq rate-limits bursts of parallel calls.
        run_config=RunConfig(max_workers=2, timeout=180),
        show_progress=False,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    return {
        "context_recall": _nanmean(df["context_recall"].tolist()),
        "context_precision": _nanmean(df["llm_context_precision_with_reference"].tolist()),
    }


# ---------------------------------------------------------------------------
# Comparing measured results with the agents' guesses
# ---------------------------------------------------------------------------

def measured_score(scores: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in (scores.get("context_recall"), scores.get("context_precision")) if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def compare_with_ai(rows: List[Dict[str, Any]], ai_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    rows: [{"option", "context_recall", "context_precision", "measured"}]
    Adds the agent's retrieval_score and both rankings, and says whether the
    agent's top pick is also the measured winner.
    """
    ai_scores = {c["name"]: c["retrieval_score"] for c in ai_rec["candidates"]}
    scored = [r for r in rows if r.get("measured") is not None]
    def rank(values: Dict[str, float]) -> Dict[str, int]:
        # Competition ranking: equal scores share a rank (1, 1, 3), never an arbitrary order.
        return {k: 1 + sum(1 for o in values.values() if o > v) for k, v in values.items()}

    measured_rank = rank({r["option"]: r["measured"] for r in scored})
    ai_rank = rank(ai_scores)

    table = [{**r, "ai_retrieval_score": ai_scores.get(r["option"]),
              "measured_rank": measured_rank.get(r["option"]), "ai_rank": ai_rank.get(r["option"])}
             for r in rows]
    winners = [o for o, rk in measured_rank.items() if rk == 1]
    return {
        "rows": table,
        "ai_top_pick": ai_rec["top_choice"],
        "measured_winner": " / ".join(winners) if winners else None,
        "tie": len(winners) > 1,
        "agrees": ai_rec["top_choice"] in winners,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _evaluate_option(option: str, chunks: List[str], questions: List[Dict[str, str]],
                     embedding: str, embed_fn: EmbedFn, scorer: Callable) -> Dict[str, Any]:
    chunk_vecs = embed_fn(embedding, chunks, "document")
    samples = [{"question": q["question"], "reference": q["reference"],
                "contexts": retrieve(q["question"], chunks, chunk_vecs, embedding, embed_fn)}
               for q in questions]
    scores = scorer(samples)
    return {"option": option, "n_chunks": len(chunks), **scores, "measured": measured_score(scores)}


def run_evaluation(
    full_text: str,
    api_key: str,
    model: str,
    embedding_rec: Dict[str, Any],
    chunking_rec: Dict[str, Any],
    n_questions: int = 5,
    embed_fn: Optional[EmbedFn] = None,
    scorer: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Optional[float]]]] = None,
    question_fn: Optional[Callable[..., List[Dict[str, str]]]] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the Embedding options, then the Chunking options (using the best
    measured embedding). embed_fn / scorer / question_fn can be injected for
    tests; by default they use sentence-transformers, RAGAS and the Groq LLM.
    """
    embed_fn = embed_fn or SentenceTransformerEmbedder()
    scorer = scorer or (lambda samples: ragas_scores(samples, api_key, model))
    question_fn = question_fn or generate_questions
    tick = progress or (lambda frac, msg: None)
    notes: List[str] = []

    n_words = len(_words(full_text))
    if n_words < 300:
        raise ValueError(f"Document too short to evaluate ({n_words} words). Upload more text (300+ words).")

    tick(0.02, "Writing test questions from your document...")
    questions = question_fn(full_text, api_key, model, n_questions)
    if not questions:
        raise RuntimeError("Could not generate any test questions (the LLM reply was unusable). Try again.")

    # --- Embedding options, chunking held fixed ---
    emb_rows, skipped = [], []
    base_chunks = chunk_text(full_text, BASELINE_CHUNKING)
    if len(base_chunks) <= TOP_K:
        notes.append(f"Document makes only {len(base_chunks)} chunk(s) at {BASELINE_CHUNKING}, so top-{TOP_K} "
                     "retrieval returns almost everything and embedding options can't be told apart. "
                     "Use a longer document for a meaningful comparison.")
    for i, option in enumerate(EMBEDDING_OPTIONS):
        tick(0.1 + 0.4 * i / len(EMBEDDING_OPTIONS), f"Embedding: {option}...")
        try:
            emb_rows.append(_evaluate_option(option, base_chunks, questions, option, embed_fn, scorer))
        except Exception as e:
            skipped.append(f"{option}: {e}")
    notes += [f"Skipped {s}" for s in skipped]
    if not emb_rows:
        raise RuntimeError("No embedding option could be evaluated. " + " | ".join(skipped))
    embedding_cmp = compare_with_ai(emb_rows, embedding_rec)

    # --- Chunking options, best measured embedding held fixed ---
    top_embeddings = [r["option"] for r in embedding_cmp["rows"] if r.get("measured_rank") == 1]
    if embedding_rec["top_choice"] in top_embeddings:  # on a tie, keep the agent's pick
        best_emb = embedding_rec["top_choice"]
    else:
        best_emb = top_embeddings[0] if top_embeddings else emb_rows[0]["option"]
    chunk_rows = []
    for i, option in enumerate(CHUNKING_OPTIONS):
        tick(0.5 + 0.45 * i / len(CHUNKING_OPTIONS), f"Chunking: {option}...")
        try:
            chunks = chunk_text(full_text, option, embed_fn, best_emb)
            chunk_rows.append(_evaluate_option(option, chunks, questions, best_emb, embed_fn, scorer))
        except Exception as e:
            notes.append(f"Skipped {option}: {e}")
    chunking_cmp = compare_with_ai(chunk_rows, chunking_rec) if chunk_rows else None
    if chunk_rows and n_words < 1500:
        notes.append(f"Document is only {n_words} words; the 512- and 1024-word chunk sizes are close to "
                     "(or bigger than) the whole document, so chunking differences are small.")

    tick(1.0, "Done")
    return {
        "n_questions": len(questions),
        "n_words": n_words,
        "chunking_held_fixed": BASELINE_CHUNKING,
        "embedding_used_for_chunking_test": best_emb,
        "embedding": embedding_cmp,
        "chunking": chunking_cmp,
        "notes": notes,
    }
