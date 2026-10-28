"""
agents.py
Domain detection + 4 parallel specialist agents (Embedding, Chunking,
Tokenizer, Indexing), orchestrated with LangGraph and backed by Groq LLMs.

Each specialist agent scores its 3 fixed candidate options for the given
domain/corpus and returns structured (Pydantic) output:
    retrieval_score, latency_score, accuracy_score in [0, 1] + rationale.

The graph shape:

            domain_detect
                 |
     +-----------+-----------+-----------+
     |           |           |           |
 embedding    chunking    tokenizer   indexing     <- run in parallel
     |           |           |           |
     +-----------+-----------+-----------+
                 |
              join_node   (no-op merge; HITL + consensus happen outside
                            the graph in app.py so the UI can pause
                            between "agents ran" and "human approved")
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import threading
import time
from collections import OrderedDict
from functools import partial
from typing import Dict, Any, List, Optional, TypedDict

from groq import (
    APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError,
    InternalServerError, NotFoundError, RateLimitError,
)
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

DOMAINS = [
    "Healthcare", "Banking", "Insurance", "Manufacturing", "Retail",
    "Telecom", "Legal", "Education", "Pharma", "Aerospace",
]

EMBEDDING_OPTIONS = ["BGE Large", "all-mpnet-base-v2", "nomic-embed-text"]
CHUNKING_OPTIONS = ["Fixed (512/100)", "Semantic (0.80)", "Recursive (1024/200)"]
TOKENIZER_OPTIONS = ["tiktoken", "SentencePiece", "WordPiece"]
INDEX_OPTIONS = ["FAISS IVF", "HNSW", "Chroma"]

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


# ---------------------------------------------------------------------------
# LLM access: cost/latency tuning, retries, model fallback, caching, usage stats
# ---------------------------------------------------------------------------

MAX_OUTPUT_TOKENS = 2048   # hard cap so a runaway reasoning trace can't burn tokens
MAX_ATTEMPTS = 5           # tries per model for transient errors (429 / 5xx / timeout)
BASE_DELAY = 1.5           # seconds; doubles each attempt, plus jitter
MAX_DELAY = 20.0           # never sleep longer than this; a longer Retry-After means "move on"
PARSE_ATTEMPTS = 2         # re-ask once if the reply isn't valid JSON for our schema
CACHE_SIZE = 128

_NO_JSON_MODE: set = set()  # models that rejected response_format; learned at runtime
_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_LOCK = threading.Lock()
_USAGE = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0, "fallbacks": 0, "cache_hits": 0}


class LLMUnavailable(RuntimeError):
    """Every model in the fallback chain failed."""


def get_llm(api_key: str, model: str = "openai/gpt-oss-120b", temperature: float = 0.2,
            json_mode: bool = False) -> ChatGroq:
    kwargs: Dict[str, Any] = {}
    if "gpt-oss" in model:
        # These models "think" before answering and bill for it. Our task is a
        # short scoring job, so low effort cuts output tokens roughly in half.
        kwargs["reasoning_effort"] = "low"
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    # max_retries=0: retrying is handled in call_llm so it can also switch models.
    return ChatGroq(groq_api_key=api_key, model=model, temperature=temperature,
                    max_tokens=MAX_OUTPUT_TOKENS, max_retries=0, **kwargs)


def _bump(**counts: int) -> None:
    with _LOCK:
        for k, v in counts.items():
            _USAGE[k] += v


def get_usage() -> Dict[str, int]:
    with _LOCK:
        return dict(_USAGE)


def usage_since(before: Dict[str, int]) -> Dict[str, int]:
    now = get_usage()
    return {k: now[k] - before.get(k, 0) for k in now}


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _cache_get(key: str):
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            _USAGE["cache_hits"] += 1
            return copy.deepcopy(_CACHE[key])
    return None


def _cache_put(key: str, value: Any) -> None:
    with _LOCK:
        _CACHE[key] = copy.deepcopy(value)
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)


def _sleep(seconds: float) -> None:  # separate function so tests can skip real waiting
    time.sleep(seconds)


def _retry_after(err: Exception) -> Optional[float]:
    resp = getattr(err, "response", None)
    try:
        return float(resp.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def _invoke(api_key: str, model: str, prompt: str, temperature: float):
    """One LLM call, using JSON mode unless this model has already refused it."""
    json_mode = model not in _NO_JSON_MODE
    try:
        return get_llm(api_key, model, temperature, json_mode=json_mode).invoke(prompt)
    except BadRequestError as e:
        text = str(e)
        if json_mode and "response_format" in text and "json_validate_failed" not in text:
            _NO_JSON_MODE.add(model)  # this model can't do JSON mode: remember, retry without it
            return get_llm(api_key, model, temperature, json_mode=False).invoke(prompt)
        raise


def call_llm(api_key: str, model: str, prompt: str, temperature: float = 0.2,
             allow_fallback: bool = True) -> "tuple[str, str]":
    """
    Ask the LLM, surviving transient failures. Returns (text, model_that_answered).

    - 429 / 5xx / timeout / connection error: wait (honouring Retry-After) and retry.
    - Retry-After longer than MAX_DELAY, or retries used up: try the next model.
    - 404 (model not on this account): next model straight away.
    - 401 (bad key): fail immediately, since no other model would work.
    """
    chain = [model] + ([m for m in GROQ_MODELS if m != model] if allow_fallback else [])
    last_err: Optional[Exception] = None
    for position, m in enumerate(chain):
        if position:
            _bump(fallbacks=1)
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = _invoke(api_key, m, prompt, temperature)
                usage = getattr(resp, "usage_metadata", None) or {}
                _bump(calls=1, input_tokens=usage.get("input_tokens", 0),
                      output_tokens=usage.get("output_tokens", 0))
                return resp.content, m
            except AuthenticationError:
                raise
            except NotFoundError as e:
                last_err = e
                break
            except BadRequestError as e:
                last_err = e
                if "json_validate_failed" not in str(e):
                    break  # a malformed request won't improve on retry
                _bump(retries=1)  # the model produced invalid JSON; asking again usually fixes it
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
                last_err = e
                wait = _retry_after(e)
                if wait is not None and wait > MAX_DELAY:
                    break  # e.g. daily limit: waiting is pointless, switch model
                if attempt == MAX_ATTEMPTS - 1:
                    break
                delay = min(MAX_DELAY, BASE_DELAY * 2 ** attempt) + random.uniform(0, 0.5)
                _bump(retries=1)
                _sleep(max(delay, wait or 0))
    raise LLMUnavailable(f"All models failed. Last error: {type(last_err).__name__}: {last_err}")


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

class CandidateScore(BaseModel):
    name: str = Field(description="Candidate option name, must match one of the offered choices exactly")
    retrieval_score: float = Field(ge=0, le=1, description="Expected retrieval quality, 0-1")
    latency_score: float = Field(ge=0, le=1, description="1 = fastest / lowest latency, 0 = slowest")
    accuracy_score: float = Field(ge=0, le=1, description="Expected downstream answer accuracy, 0-1")
    rationale: str = Field(description="One or two sentence justification")


class AgentRecommendation(BaseModel):
    category: str
    top_choice: str
    candidates: List[CandidateScore]


class DomainDetection(BaseModel):
    domain: str = Field(description="Best matching domain from the provided list")
    confidence: float = Field(ge=0, le=1)
    reasoning: str


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    corpus_text: str
    api_key: str
    model: str
    domain: str
    domain_override: str
    domain_confidence: float
    domain_reasoning: str
    embedding_rec: Dict[str, Any]
    chunking_rec: Dict[str, Any]
    tokenizer_rec: Dict[str, Any]
    indexing_rec: Dict[str, Any]
    feedback: Dict[str, str]  # optional per-category rejection feedback for re-runs
    bypass_cache: bool        # True on human-requested re-runs, so they get a fresh answer
    allow_fallback: bool      # False to pin the run to `model` and never switch to another


# ---------------------------------------------------------------------------
# Prompting helpers
# ---------------------------------------------------------------------------

def _agent_prompt(category: str, options: List[str], domain: str, corpus_text: str, feedback: str = "") -> str:
    feedback_block = (
        f"\nA human reviewer REJECTED a previous recommendation for this category "
        f"with the note: \"{feedback}\". Take this into account and adjust your scoring "
        f"or choose a different top pick if justified.\n"
        if feedback else ""
    )
    return f"""You are the {category} specialist agent in a RAG architecture advisory system.

Domain detected for this content: {domain}

Content sample (may be truncated):
\"\"\"{corpus_text[:3000]}\"\"\"

Your fixed candidate options for {category} are exactly: {options}
{feedback_block}
Score EACH of the 3 candidates on:
- retrieval_score (0-1): how well it supports accurate retrieval for this domain/content
- latency_score (0-1): 1 = fastest, 0 = slowest, relative to the other two candidates
- accuracy_score (0-1): expected downstream answer accuracy contribution

Pick the top_choice (must be one of the 3 option strings, verbatim).

Respond ONLY with a JSON object of this exact shape, no markdown fences, no extra text:
{{
  "category": "{category}",
  "top_choice": "<one of the options>",
  "candidates": [
    {{"name": "<option 1>", "retrieval_score": 0.0, "latency_score": 0.0, "accuracy_score": 0.0, "rationale": "..."}},
    {{"name": "<option 2>", "retrieval_score": 0.0, "latency_score": 0.0, "accuracy_score": 0.0, "rationale": "..."}},
    {{"name": "<option 3>", "retrieval_score": 0.0, "latency_score": 0.0, "accuracy_score": 0.0, "rationale": "..."}}
  ]
}}"""


def _safe_json_parse(raw: str) -> Dict[str, Any]:
    # Some reasoning models prepend <think>...</think> to their answer.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1) if raw.startswith("json\n") else raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return json.loads(raw)


def _degraded_rec(category: str, options: List[str], reason: str) -> Dict[str, Any]:
    """Neutral placeholder scores, flagged so the UI can warn instead of passing them off as real."""
    return {
        "category": category,
        "top_choice": options[0],
        "candidates": [
            {"name": o, "retrieval_score": 0.5, "latency_score": 0.5,
             "accuracy_score": 0.5, "rationale": f"Fallback ({reason})"}
            for o in options
        ],
        "degraded": True,
        "error": reason,
    }


TOP_CHOICE_TOLERANCE = 0.1  # top pick may trail the best weighted score by at most this much


def _canonical(name: str, allowed: List[str]) -> Optional[str]:
    """Match a name to an allowed option ignoring case and surrounding spaces."""
    wanted = str(name).strip().lower()
    return next((a for a in allowed if a.lower() == wanted), None)


def _weighted(c: Dict[str, Any]) -> float:
    # Same weights the consensus engine uses, rescaled to 0-1 (retrieval 0.4, latency 0.2, accuracy 0.2).
    return (0.4 * c["retrieval_score"] + 0.2 * c["latency_score"] + 0.2 * c["accuracy_score"]) / 0.8


def _validate_rec(rec: Dict[str, Any], category: str, options: List[str]) -> Dict[str, Any]:
    """
    Catch answers that fit the JSON schema but make no sense. Raises ValueError with a
    message that is sent back to the model on the retry. Does NOT judge whether the
    scores are true - only that the answer is well-formed and consistent with itself.
    """
    names = []
    for c in rec["candidates"]:
        canon = _canonical(c["name"], options)
        if canon is None:
            raise ValueError(f"'{c['name']}' is not one of the allowed options {options}")
        c["name"] = canon
        names.append(canon)
    if sorted(names) != sorted(options):
        raise ValueError(f"score every option exactly once ({options}); you returned {names}")

    top = _canonical(rec["top_choice"], options)
    if top is None:
        raise ValueError(f"top_choice '{rec['top_choice']}' is not one of the allowed options {options}")
    rec["top_choice"] = top

    scores = {c["name"]: _weighted(c) for c in rec["candidates"]}
    if scores[top] < max(scores.values()) - TOP_CHOICE_TOLERANCE:
        best = max(scores, key=scores.get)
        raise ValueError(f"top_choice '{top}' scores {scores[top]:.2f} but '{best}' scores "
                         f"{scores[best]:.2f}; the top_choice must agree with your scores")
    rec["category"] = category  # the model sometimes echoes this wrongly; we know the right value
    return rec


def _retry_note(error: str) -> str:
    return (f"\n\nYour previous reply was rejected: {error}\n"
            "Answer again, following the instructions and the exact JSON shape.")


def _run_agent(call, category: str, options: List[str], domain: str,
               corpus_text: str, feedback: str = "") -> Dict[str, Any]:
    """
    `call(prompt) -> (text, model_used)` is the LLM (see call_llm). Never raises:
    if the LLM is unreachable or keeps returning unusable answers, a degraded
    placeholder is returned so the other agents still finish.
    """
    base_prompt = _agent_prompt(category, options, domain, corpus_text, feedback)
    prompt, last_err = base_prompt, "unknown error"
    for _ in range(PARSE_ATTEMPTS):
        try:
            text, used = call(prompt)
        except LLMUnavailable as e:
            return _degraded_rec(category, options, f"LLM unavailable: {e}")
        try:
            rec = AgentRecommendation(**_safe_json_parse(text)).model_dump()
            rec = _validate_rec(rec, category, options)
            rec["model_used"] = used
            rec["degraded"] = False
            return rec
        except Exception as e:
            last_err = f"invalid answer: {e}"
            prompt = base_prompt + _retry_note(str(e))  # tell the model what was wrong
            _bump(retries=1)
    return _degraded_rec(category, options, last_err)


def _cached_agent(state: "GraphState", category: str, options: List[str]) -> Dict[str, Any]:
    model = state.get("model", "openai/gpt-oss-120b")
    fb = (state.get("feedback") or {}).get(category, "")
    # Same model + domain + document + feedback gives the same question, so reuse the answer
    # instead of paying for it again. Degraded results are never cached.
    key = _cache_key("agent", model, category, state["domain"], state["corpus_text"], fb)
    # A re-run after the human rejected an answer must not just replay that same answer.
    hit = None if state.get("bypass_cache") else _cache_get(key)
    if hit is not None:
        return hit
    call = partial(call_llm, state["api_key"], model, allow_fallback=state.get("allow_fallback", True))
    rec = _run_agent(call, category, options, state["domain"], state["corpus_text"], fb)
    if not rec.get("degraded"):
        _cache_put(key, rec)
    return rec


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def node_domain_detect(state: GraphState) -> GraphState:
    override = state.get("domain_override")
    if override and override in DOMAINS:
        # Manual override: skip the LLM call entirely. Domain is an input to
        # every downstream agent's prompt, so this is a genuine re-evaluation
        # trigger, not a display filter -- but skipping the extra LLM call
        # here actually saves latency versus the auto-detect path.
        state["domain"] = override
        state["domain_confidence"] = 1.0
        state["domain_reasoning"] = "Manually overridden by user via the Domain Override control."
        return state

    model = state.get("model", "openai/gpt-oss-120b")
    key = _cache_key("domain", model, state["corpus_text"])
    hit = None if state.get("bypass_cache") else _cache_get(key)
    if hit is not None:
        state.update(hit)
        return state

    prompt = f"""Classify the following enterprise content sample into exactly one domain
from this list: {DOMAINS}

Content sample:
\"\"\"{state['corpus_text'][:3000]}\"\"\"

Respond ONLY with JSON, no markdown fences:
{{"domain": "<one of the list, verbatim>", "confidence": 0.0, "reasoning": "..."}}"""
    result, reason, current = None, "unknown error", prompt
    for _ in range(PARSE_ATTEMPTS):
        try:
            text, _ = call_llm(state["api_key"], model, current,
                               allow_fallback=state.get("allow_fallback", True))
        except LLMUnavailable as e:
            reason = f"LLM unavailable: {e}"
            break
        try:
            parsed = DomainDetection(**_safe_json_parse(text))
            domain = _canonical(parsed.domain, DOMAINS)
            if domain is None:
                raise ValueError(f"'{parsed.domain}' is not one of the allowed domains {DOMAINS}")
            result = {"domain": domain, "domain_confidence": parsed.confidence,
                      "domain_reasoning": parsed.reasoning}
            _cache_put(key, result)
            break
        except Exception as e:  # unreadable or not on the allowed list: ask again, saying why
            reason = f"invalid answer: {e}"
            current = prompt + _retry_note(str(e))
            _bump(retries=1)
    if result is None:  # keep going with a default, but say so (the UI warns on confidence 0)
        result = {"domain": "Manufacturing", "domain_confidence": 0.0,
                  "domain_reasoning": f"Fallback domain ({reason})"}
    state.update(result)
    return state


def node_embedding(state: GraphState) -> GraphState:
    return {"embedding_rec": _cached_agent(state, "Embedding", EMBEDDING_OPTIONS)}


def node_chunking(state: GraphState) -> GraphState:
    return {"chunking_rec": _cached_agent(state, "Chunking", CHUNKING_OPTIONS)}


def node_tokenizer(state: GraphState) -> GraphState:
    return {"tokenizer_rec": _cached_agent(state, "Tokenizer", TOKENIZER_OPTIONS)}


def node_indexing(state: GraphState) -> GraphState:
    return {"indexing_rec": _cached_agent(state, "Indexing", INDEX_OPTIONS)}


def node_join(state: GraphState) -> GraphState:
    # No-op merge point; LangGraph waits for all 4 parallel branches before this runs.
    return state


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("domain_detect", node_domain_detect)
    graph.add_node("embedding", node_embedding)
    graph.add_node("chunking", node_chunking)
    graph.add_node("tokenizer", node_tokenizer)
    graph.add_node("indexing", node_indexing)
    graph.add_node("join", node_join)

    graph.set_entry_point("domain_detect")
    for branch in ["embedding", "chunking", "tokenizer", "indexing"]:
        graph.add_edge("domain_detect", branch)
        graph.add_edge(branch, "join")
    graph.add_edge("join", END)

    return graph.compile()


def run_pipeline(corpus_text: str, api_key: str, model: str = "openai/gpt-oss-120b",
                  feedback: Dict[str, str] | None = None, domain_override: str | None = None,
                  allow_fallback: bool = True, use_cache: bool = True) -> Dict[str, Any]:
    """
    Synchronous convenience wrapper used by the Gradio app. Besides the graph
    state it returns:
      usage    - tokens / retries / fallbacks / cache hits used by this run
                 (approximate if several runs overlap in one process)
      degraded - names of agents that fell back to placeholder scores
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()  # compiling is not free, so do it once
    init_state: GraphState = {
        "corpus_text": corpus_text,
        "api_key": api_key,
        "model": model,
        "feedback": feedback or {},
        "domain_override": domain_override or "",
        "allow_fallback": allow_fallback,
        "bypass_cache": not use_cache,
    }
    before = get_usage()
    final_state = dict(_GRAPH.invoke(init_state))
    final_state["usage"] = usage_since(before)
    final_state["degraded"] = [
        c for c in ("Embedding", "Chunking", "Tokenizer", "Indexing")
        if final_state.get(f"{c.lower()}_rec", {}).get("degraded")
    ]
    return final_state


_GRAPH = None
