"""
consensus.py
Combines the 4 agents' scored candidates + human approval status into
ranked full-stack RAG architecture recommendations.

Final Score = 0.4 * Retrieval + 0.2 * Latency + 0.2 * Human Approval + 0.2 * Accuracy

Human approval is captured per-agent (Approve/Reject), per the case study's
HITL design ("each of the 4 agents requires Approve/Reject"). It is folded
into every combination's score as a shared multiplier/term: a rejection on
any agent pulls the human-approval component down for every combination
until that agent's recommendation is re-run and approved.

Post-hoc category locks (Chunking / Tokenizer / Indexing) let the caller
restrict the combination search to a single named candidate for that
category WITHOUT any new LLM calls — the agents already scored all 3
candidates per category, so locking is pure re-ranking over cached data.
Embedding is intentionally NOT lockable here: it drives the domain-specific
retrieval estimate most directly, so locking it post-hoc without re-scoring
would risk presenting a stale number. A genuine embedding change should go
through a full re-run instead (see agents.py domain_override for the same
pattern applied to domain).
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Any, List, Optional


# Candidates per agent used when building combinations. Widened from 2 to 3
# (i.e. ALL scored candidates) so that a post-hoc lock on a candidate that
# would previously have been pruned as "2nd/3rd best" still has full data
# to filter over instead of silently missing it.
DEFAULT_CANDIDATES_PER_AGENT = 3


def _top_n_candidates(agent_rec: Dict[str, Any], n: int = DEFAULT_CANDIDATES_PER_AGENT) -> List[Dict[str, Any]]:
    cands = sorted(
        agent_rec["candidates"],
        key=lambda c: (0.4 * c["retrieval_score"] + 0.2 * c["latency_score"] + 0.2 * c["accuracy_score"]),
        reverse=True,
    )
    return cands[:n]


def _filtered_candidates(
    agent_rec: Dict[str, Any],
    lock_name: Optional[str],
    n: int = DEFAULT_CANDIDATES_PER_AGENT,
) -> List[Dict[str, Any]]:
    """
    Return the candidate list for a category, restricted to a single named
    candidate if `lock_name` is set (and not "All"/falsy). This is the
    free, no-LLM-call filtering path.
    """
    if lock_name and lock_name != "All":
        matched = [c for c in agent_rec["candidates"] if c["name"] == lock_name]
        return matched if matched else _top_n_candidates(agent_rec, n)
    return _top_n_candidates(agent_rec, n)


def human_approval_score(approvals: Dict[str, bool]) -> float:
    """Fraction of the 4 agents approved by the human reviewer (0-1)."""
    if not approvals:
        return 0.0
    return sum(1 for v in approvals.values() if v) / len(approvals)


def _sort_key(c: Dict[str, Any]):
    """
    Explicit tiebreak chain for equal final_score, in priority order:
      1. Higher avg retrieval_score (carries the highest formula weight, 0.4)
      2. Higher human_approval_score
      3. Higher latency_score (1.0 = fastest -> prefer the faster stack)
      4. Lexicographic on Embedding name, for deterministic output on re-runs
    """
    return (
        -c["final_score"],
        -c["retrieval_score"],
        -c["human_approval_score"],
        -c["latency_score"],
        c["Embedding"],
    )


def _build_all_combos(
    embedding_rec: Dict[str, Any],
    chunking_rec: Dict[str, Any],
    tokenizer_rec: Dict[str, Any],
    indexing_rec: Dict[str, Any],
    approvals: Dict[str, bool],
    locks: Optional[Dict[str, str]] = None,
    candidates_per_agent: int = DEFAULT_CANDIDATES_PER_AGENT,
) -> List[Dict[str, Any]]:
    locks = locks or {}
    per_agent = {
        "Embedding": _top_n_candidates(embedding_rec, candidates_per_agent),
        "Chunking": _filtered_candidates(chunking_rec, locks.get("Chunking"), candidates_per_agent),
        "Tokenizer": _filtered_candidates(tokenizer_rec, locks.get("Tokenizer"), candidates_per_agent),
        "Indexing": _filtered_candidates(indexing_rec, locks.get("Indexing"), candidates_per_agent),
    }

    human_score = human_approval_score(approvals)

    combos = []
    for emb, chk, tok, idx in product(
        per_agent["Embedding"], per_agent["Chunking"], per_agent["Tokenizer"], per_agent["Indexing"]
    ):
        parts = [emb, chk, tok, idx]
        avg_retrieval = sum(p["retrieval_score"] for p in parts) / 4
        avg_latency = sum(p["latency_score"] for p in parts) / 4
        avg_accuracy = sum(p["accuracy_score"] for p in parts) / 4

        final_score = (
            0.4 * avg_retrieval
            + 0.2 * avg_latency
            + 0.2 * human_score
            + 0.2 * avg_accuracy
        )

        combos.append(
            {
                "Embedding": emb["name"],
                "Chunking": chk["name"],
                "Tokenizer": tok["name"],
                "Indexing": idx["name"],
                "retrieval_score": round(avg_retrieval, 3),
                "latency_score": round(avg_latency, 3),
                "human_approval_score": round(human_score, 3),
                "accuracy_score": round(avg_accuracy, 3),
                "final_score": round(final_score, 4),
            }
        )

    combos.sort(key=_sort_key)
    return combos


def build_recommendations(
    embedding_rec: Dict[str, Any],
    chunking_rec: Dict[str, Any],
    tokenizer_rec: Dict[str, Any],
    indexing_rec: Dict[str, Any],
    approvals: Dict[str, bool],
    top_k: int = 2,
    candidates_per_agent: int = DEFAULT_CANDIDATES_PER_AGENT,
    locks: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build up to `top_k` distinct full-stack recommendations by combining
    each agent's candidates (optionally restricted per-category via
    `locks`), scoring every combination, and returning the highest-ranked,
    de-duplicated set using the explicit tiebreak chain in `_sort_key`.
    """
    combos = _build_all_combos(
        embedding_rec, chunking_rec, tokenizer_rec, indexing_rec,
        approvals, locks=locks, candidates_per_agent=candidates_per_agent,
    )
    human_score = human_approval_score(approvals)
    all_approved = human_score == 1.0

    # De-duplicate by the (Embedding, Chunking, Tokenizer, Indexing) tuple
    # and keep the top_k distinct stacks. (Combos are already unique tuples
    # by construction, but de-dup is kept as a safety net.)
    seen = set()
    ranked: List[Dict[str, Any]] = []
    for c in combos:
        key = (c["Embedding"], c["Chunking"], c["Tokenizer"], c["Indexing"])
        if key in seen:
            continue
        seen.add(key)
        ranked.append(c)
        if len(ranked) >= top_k:
            break

    return {
        "gate_passed": all_approved,
        "human_approval_score": round(human_score, 3),
        "recommendations": ranked,
        "all_combos_considered": len(combos),
    }


def build_comparison_table(
    embedding_rec: Dict[str, Any],
    chunking_rec: Dict[str, Any],
    tokenizer_rec: Dict[str, Any],
    indexing_rec: Dict[str, Any],
    approvals: Optional[Dict[str, bool]] = None,
    locks: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Return ALL considered combinations (not just Top-2), fully ranked, for
    the Comparison tab. Uses all 3 candidates per category (minus any
    active lock) so the side-by-side view reflects the complete search
    space the Consensus Engine actually reasoned over.
    """
    approvals = approvals or {}
    return _build_all_combos(
        embedding_rec, chunking_rec, tokenizer_rec, indexing_rec,
        approvals, locks=locks, candidates_per_agent=DEFAULT_CANDIDATES_PER_AGENT,
    )
