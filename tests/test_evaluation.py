"""Tests for evaluation.py with fake embeddings / scorer / questions. No network, no model downloads."""

import zlib

import numpy as np
import pytest

import evaluation as ev
from evaluation import (
    fixed_chunks, recursive_chunks, semantic_chunks, retrieve, compare_with_ai,
    measured_score, run_evaluation, _passages, _nanmean,
)


def fake_embed(option, texts, kind):
    """Bag-of-words hashed into 64 dims, L2-normalised: similar words -> similar vectors."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % 64] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def _text(n_words):
    return " ".join(f"word{i}" for i in range(n_words))


def _ai_rec(top, scores):
    return {"category": "X", "top_choice": top, "candidates": [
        {"name": n, "retrieval_score": s, "latency_score": 0.5, "accuracy_score": 0.5, "rationale": ""}
        for n, s in scores.items()]}


# --- chunkers -----------------------------------------------------------------

def test_fixed_chunks_size_and_overlap():
    chunks = fixed_chunks(_text(1000), size=512, overlap=100)
    assert all(len(c.split()) <= 512 for c in chunks)
    # consecutive chunks share exactly `overlap` words
    assert chunks[0].split()[-100:] == chunks[1].split()[:100]


def test_fixed_chunks_cover_every_word():
    words = set(_text(1000).split())
    assert set(" ".join(fixed_chunks(_text(1000))).split()) == words


def test_fixed_chunks_short_text_is_one_chunk():
    assert len(fixed_chunks(_text(50))) == 1


def test_fixed_chunks_empty():
    assert fixed_chunks("") == []


def test_recursive_chunks_respect_size():
    text = "\n\n".join(_text(300) for _ in range(10))
    chunks = recursive_chunks(text, size=500, overlap=50)
    assert all(len(c.split()) <= 500 + 50 for c in chunks)
    assert len(chunks) > 1


def test_recursive_prefers_paragraph_boundaries():
    para_a, para_b = "alpha " * 40, "beta " * 40
    chunks = recursive_chunks(para_a.strip() + "\n\n" + para_b.strip(), size=50, overlap=0)
    assert chunks[0].split() == ["alpha"] * 40  # paragraph kept whole
    assert chunks[1].split()[0] == "beta"


def test_recursive_handles_text_with_no_separators():
    chunks = recursive_chunks(_text(3000), size=1024, overlap=200)
    assert len(chunks) >= 3


def test_semantic_splits_when_topic_changes():
    text = "Cats purr softly. Cats purr loudly. Stocks fell sharply today. Stocks fell again."
    # Topic-aware fake: sentences about cats vs stocks get orthogonal vectors.
    def embed(sents):
        return np.array([[1, 0] if "Cats" in s else [0, 1] for s in sents], dtype=np.float32)
    chunks = semantic_chunks(text, embed, threshold=0.8)
    assert len(chunks) == 2
    assert "Cats" in chunks[0] and "Stocks" in chunks[1]


def test_semantic_keeps_similar_sentences_together():
    text = "One sentence. Two sentence. Three sentence."
    chunks = semantic_chunks(text, lambda s: np.ones((len(s), 2), dtype=np.float32) / np.sqrt(2), 0.8)
    assert len(chunks) == 1


def test_chunk_text_rejects_unknown_option():
    with pytest.raises(ValueError):
        ev.chunk_text("text", "Magic")


def test_semantic_requires_embed_fn():
    with pytest.raises(ValueError):
        ev.chunk_text("a b c", "Semantic (0.80)")


# --- retrieval ----------------------------------------------------------------

def test_retrieve_returns_most_similar_chunk_first():
    chunks = ["the cat sat on the mat", "stock markets fell sharply", "bananas are yellow fruit"]
    vecs = fake_embed("m", chunks, "document")
    got = retrieve("why did stock markets fall", chunks, vecs, "m", fake_embed, top_k=1)
    assert got == ["stock markets fell sharply"]


# --- questions / helpers --------------------------------------------------------

def test_passages_are_spread_across_document():
    ps = _passages(_text(2000), n=4, window=250)
    assert len(ps) == 4
    assert ps[0].startswith("word0 ") and "word1999" in ps[-1]


def test_passages_short_text_gives_one():
    assert len(_passages(_text(100), n=5)) == 1


def test_nanmean_ignores_nan_and_none():
    assert _nanmean([1.0, float("nan"), None, 0.0]) == 0.5
    assert _nanmean([float("nan")]) is None


def test_measured_score_is_mean_of_available_metrics():
    assert measured_score({"context_recall": 0.8, "context_precision": 0.4}) == 0.6
    assert measured_score({"context_recall": 0.8, "context_precision": None}) == 0.8
    assert measured_score({"context_recall": None, "context_precision": None}) is None


# --- comparing with the AI --------------------------------------------------------

def test_compare_detects_agreement():
    rows = [{"option": "A", "measured": 0.9}, {"option": "B", "measured": 0.5}]
    out = compare_with_ai(rows, _ai_rec("A", {"A": 0.8, "B": 0.6}))
    assert out["agrees"] is True and out["measured_winner"] == "A"


def test_compare_detects_disagreement_and_ranks():
    rows = [{"option": "A", "measured": 0.3}, {"option": "B", "measured": 0.9}]
    out = compare_with_ai(rows, _ai_rec("A", {"A": 0.8, "B": 0.6}))
    assert out["agrees"] is False and out["measured_winner"] == "B"
    by = {r["option"]: r for r in out["rows"]}
    assert by["B"]["measured_rank"] == 1 and by["B"]["ai_rank"] == 2
    assert by["A"]["ai_retrieval_score"] == 0.8


def test_compare_reports_a_tie_instead_of_inventing_a_winner():
    rows = [{"option": "A", "measured": 0.6}, {"option": "B", "measured": 0.6},
            {"option": "C", "measured": 0.2}]
    out = compare_with_ai(rows, _ai_rec("B", {"A": 0.5, "B": 0.9, "C": 0.1}))
    ranks = {r["option"]: r["measured_rank"] for r in out["rows"]}
    assert ranks == {"A": 1, "B": 1, "C": 3}
    assert out["tie"] is True and out["agrees"] is True


def test_tie_without_the_ai_pick_is_a_disagreement():
    rows = [{"option": "A", "measured": 0.6}, {"option": "B", "measured": 0.6},
            {"option": "C", "measured": 0.2}]
    out = compare_with_ai(rows, _ai_rec("C", {"A": 0.5, "B": 0.4, "C": 0.9}))
    assert out["tie"] is True and out["agrees"] is False


def test_compare_with_no_measured_scores():
    out = compare_with_ai([{"option": "A", "measured": None}], _ai_rec("A", {"A": 0.8}))
    assert out["measured_winner"] is None and out["agrees"] is False


# --- full run with fakes --------------------------------------------------------------

def _questions(text, key, model, n):
    return [{"question": "what is word5", "reference": "word5"}] * 2


def _scorer_favouring(best_substring_len):
    """Fake RAGAS: rewards samples whose contexts are short (so smaller chunks win)."""
    def scorer(samples):
        avg = np.mean([len(" ".join(s["contexts"]).split()) for s in samples])
        score = round(float(1 / (1 + avg / best_substring_len)), 3)
        return {"context_recall": score, "context_precision": score}
    return scorer


def _emb_rec():
    return _ai_rec("BGE Large", {"BGE Large": 0.9, "all-mpnet-base-v2": 0.7, "nomic-embed-text": 0.6})


def _chk_rec():
    return _ai_rec("Fixed (512/100)", {"Fixed (512/100)": 0.7, "Semantic (0.80)": 0.8, "Recursive (1024/200)": 0.6})


def test_run_evaluation_end_to_end_with_fakes():
    progress = []
    out = run_evaluation(
        _text(3000) + ". " + _text(10), "key", "model", _emb_rec(), _chk_rec(), n_questions=2,
        embed_fn=fake_embed, scorer=_scorer_favouring(500), question_fn=_questions,
        progress=lambda f, m: progress.append(f),
    )
    assert out["n_questions"] == 2
    assert {r["option"] for r in out["embedding"]["rows"]} == set(ev.EMBEDDING_OPTIONS)
    assert {r["option"] for r in out["chunking"]["rows"]} == set(ev.CHUNKING_OPTIONS)
    assert out["embedding_used_for_chunking_test"] in ev.EMBEDDING_OPTIONS
    assert progress[-1] == 1.0 and progress == sorted(progress)


def test_run_evaluation_with_tied_scores_uses_a_single_valid_embedding():
    out = run_evaluation(_text(3000), "k", "m", _emb_rec(), _chk_rec(), embed_fn=fake_embed,
                         scorer=lambda s: {"context_recall": 0.5, "context_precision": 0.5},
                         question_fn=_questions)
    assert out["embedding"]["tie"] is True
    assert out["embedding_used_for_chunking_test"] == "BGE Large"  # the agent's pick, kept on a tie


def test_run_evaluation_rejects_tiny_document():
    with pytest.raises(ValueError, match="too short"):
        run_evaluation(_text(50), "k", "m", _emb_rec(), _chk_rec(),
                       embed_fn=fake_embed, scorer=lambda s: {}, question_fn=_questions)


def test_run_evaluation_fails_clearly_without_questions():
    with pytest.raises(RuntimeError, match="test questions"):
        run_evaluation(_text(500), "k", "m", _emb_rec(), _chk_rec(), embed_fn=fake_embed,
                       scorer=lambda s: {}, question_fn=lambda *a: [])


def test_run_evaluation_skips_an_option_that_fails_and_says_so():
    def flaky_embed(option, texts, kind):
        if option == "nomic-embed-text":
            raise RuntimeError("remote code not allowed")
        return fake_embed(option, texts, kind)
    out = run_evaluation(_text(2000), "k", "m", _emb_rec(), _chk_rec(), embed_fn=flaky_embed,
                         scorer=_scorer_favouring(100), question_fn=_questions)
    assert {r["option"] for r in out["embedding"]["rows"]} == {"BGE Large", "all-mpnet-base-v2"}
    assert any("nomic-embed-text" in n for n in out["notes"])


def test_run_evaluation_fails_when_every_embedding_fails():
    def broken(option, texts, kind):
        raise RuntimeError("no model")
    with pytest.raises(RuntimeError, match="No embedding option"):
        run_evaluation(_text(2000), "k", "m", _emb_rec(), _chk_rec(), embed_fn=broken,
                       scorer=_scorer_favouring(100), question_fn=_questions)


def test_run_evaluation_warns_when_document_is_short():
    out = run_evaluation(_text(400), "k", "m", _emb_rec(), _chk_rec(), embed_fn=fake_embed,
                         scorer=_scorer_favouring(100), question_fn=_questions)
    assert any("can't be told apart" in n for n in out["notes"])
