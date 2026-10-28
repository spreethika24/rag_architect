"""Tests for the ranking logic in consensus.py. Pure functions, no LLM calls."""

import pytest

from consensus import (
    build_recommendations, build_comparison_table, human_approval_score,
    _top_n_candidates, _sort_key,
)

ALL_APPROVED = {"Embedding": True, "Chunking": True, "Tokenizer": True, "Indexing": True}


def _cand(name, r=0.5, l=0.5, a=0.5):
    return {"name": name, "retrieval_score": r, "latency_score": l,
            "accuracy_score": a, "rationale": ""}


def _rec(category, cands):
    return {"category": category, "top_choice": cands[0]["name"], "candidates": cands}


def _flat(category, names):
    """Category where every candidate scores the same."""
    return _rec(category, [_cand(n) for n in names])


@pytest.fixture
def recs():
    return {
        "embedding_rec": _rec("Embedding", [
            _cand("BGE Large", r=0.9), _cand("mpnet", r=0.7), _cand("nomic", r=0.6)]),
        "chunking_rec": _flat("Chunking", ["Fixed", "Semantic", "Recursive"]),
        "tokenizer_rec": _flat("Tokenizer", ["tiktoken", "SentencePiece", "WordPiece"]),
        "indexing_rec": _flat("Indexing", ["FAISS", "HNSW", "Chroma"]),
    }


# --- human approval ----------------------------------------------------------

def test_human_score_all_approved():
    assert human_approval_score(ALL_APPROVED) == 1.0


def test_human_score_partial():
    assert human_approval_score({"a": True, "b": True, "c": False, "d": False}) == 0.5


def test_human_score_empty_is_zero():
    assert human_approval_score({}) == 0.0


# --- ranking -----------------------------------------------------------------

def test_best_embedding_ranks_first(recs):
    out = build_recommendations(**recs, approvals=ALL_APPROVED)
    assert out["recommendations"][0]["Embedding"] == "BGE Large"


def test_returns_top_k_distinct_stacks(recs):
    out = build_recommendations(**recs, approvals=ALL_APPROVED, top_k=2)
    stacks = [(r["Embedding"], r["Chunking"], r["Tokenizer"], r["Indexing"])
              for r in out["recommendations"]]
    assert len(stacks) == 2
    assert len(set(stacks)) == 2


def test_considers_every_combination(recs):
    out = build_recommendations(**recs, approvals=ALL_APPROVED)
    assert out["all_combos_considered"] == 3 ** 4


def test_final_score_matches_formula(recs):
    top = build_recommendations(**recs, approvals=ALL_APPROVED)["recommendations"][0]
    # Everything except embedding scores 0.5; embedding is BGE Large (0.9/0.5/0.5).
    avg_r = (0.9 + 0.5 * 3) / 4
    expected = 0.4 * avg_r + 0.2 * 0.5 + 0.2 * 1.0 + 0.2 * 0.5
    assert top["final_score"] == pytest.approx(expected, abs=1e-4)


def test_results_sorted_best_first(recs):
    table = build_comparison_table(**recs, approvals=ALL_APPROVED)
    scores = [c["final_score"] for c in table]
    assert scores == sorted(scores, reverse=True)


def test_tie_broken_deterministically(recs):
    # Same inputs must give identical output on every call.
    first = build_comparison_table(**recs, approvals=ALL_APPROVED)
    second = build_comparison_table(**recs, approvals=ALL_APPROVED)
    assert first == second


def test_tiebreak_prefers_higher_retrieval():
    a = {"final_score": 0.5, "retrieval_score": 0.9, "human_approval_score": 1.0,
         "latency_score": 0.1, "Embedding": "z"}
    b = {"final_score": 0.5, "retrieval_score": 0.4, "human_approval_score": 1.0,
         "latency_score": 0.9, "Embedding": "a"}
    assert sorted([b, a], key=_sort_key)[0] is a


def test_tiebreak_falls_back_to_embedding_name():
    base = {"final_score": 0.5, "retrieval_score": 0.5, "human_approval_score": 1.0,
            "latency_score": 0.5}
    a, b = {**base, "Embedding": "alpha"}, {**base, "Embedding": "beta"}
    assert sorted([b, a], key=_sort_key)[0] is a


# --- approval gate ------------------------------------------------------------

def test_gate_passes_only_when_all_approved(recs):
    assert build_recommendations(**recs, approvals=ALL_APPROVED)["gate_passed"] is True


def test_gate_fails_on_any_rejection(recs):
    approvals = {**ALL_APPROVED, "Chunking": False}
    out = build_recommendations(**recs, approvals=approvals)
    assert out["gate_passed"] is False
    assert out["human_approval_score"] == 0.75


def test_rejection_lowers_every_final_score(recs):
    approved = build_recommendations(**recs, approvals=ALL_APPROVED)["recommendations"][0]
    rejected = build_recommendations(
        **recs, approvals={**ALL_APPROVED, "Indexing": False})["recommendations"][0]
    assert rejected["final_score"] < approved["final_score"]


# --- locks --------------------------------------------------------------------

def test_lock_restricts_that_category(recs):
    out = build_recommendations(**recs, approvals=ALL_APPROVED, locks={"Indexing": "HNSW"})
    assert all(r["Indexing"] == "HNSW" for r in out["recommendations"])
    assert out["all_combos_considered"] == 3 * 3 * 3 * 1


def test_multiple_locks_combine(recs):
    out = build_recommendations(
        **recs, approvals=ALL_APPROVED,
        locks={"Chunking": "Semantic", "Tokenizer": "WordPiece", "Indexing": "Chroma"})
    top = out["recommendations"][0]
    assert (top["Chunking"], top["Tokenizer"], top["Indexing"]) == ("Semantic", "WordPiece", "Chroma")
    assert out["all_combos_considered"] == 3


def test_lock_all_means_no_lock(recs):
    out = build_recommendations(**recs, approvals=ALL_APPROVED, locks={"Chunking": "All"})
    assert out["all_combos_considered"] == 3 ** 4


def test_unknown_lock_name_does_not_empty_the_search(recs):
    out = build_recommendations(**recs, approvals=ALL_APPROVED, locks={"Chunking": "DoesNotExist"})
    assert out["all_combos_considered"] == 3 ** 4


def test_embedding_cannot_be_locked(recs):
    out = build_recommendations(**recs, approvals=ALL_APPROVED, locks={"Embedding": "nomic"})
    assert out["all_combos_considered"] == 3 ** 4


# --- candidate helper ---------------------------------------------------------

def test_top_n_orders_by_weighted_score():
    rec = _rec("X", [_cand("low", r=0.1, l=0.1, a=0.1), _cand("high", r=0.9, l=0.9, a=0.9)])
    assert [c["name"] for c in _top_n_candidates(rec, n=2)] == ["high", "low"]
