"""Tests for how agents.py reads (and survives) LLM replies. No network calls."""

import json

import pytest

import agents
from agents import (
    EMBEDDING_OPTIONS, LLMUnavailable, _safe_json_parse, _run_agent, node_domain_detect,
)


@pytest.fixture(autouse=True)
def fresh_cache():
    agents.clear_cache()
    yield
    agents.clear_cache()


def _valid_reply(options=EMBEDDING_OPTIONS, category="Embedding", top=None):
    return json.dumps({
        "category": category,
        "top_choice": top or options[0],
        "candidates": [
            {"name": o, "retrieval_score": 0.8, "latency_score": 0.5,
             "accuracy_score": 0.7, "rationale": "ok"}
            for o in options
        ],
    })


def fake_call(*replies):
    """An LLM stand-in: returns the given replies in order (the last one repeats)."""
    seen = []

    def call(prompt):
        seen.append(prompt)
        return replies[min(len(seen) - 1, len(replies) - 1)], "fake-model"

    call.seen = seen
    return call


# --- _safe_json_parse -------------------------------------------------------

def test_parses_plain_json():
    assert _safe_json_parse('{"a": 1}') == {"a": 1}


def test_strips_think_block():
    assert _safe_json_parse('<think>let me reason about this...</think>{"a": 1}') == {"a": 1}


def test_strips_multiline_think_block():
    assert _safe_json_parse('<think>line one\nline two {not json}\n</think>\n{"a": 2}') == {"a": 2}


def test_strips_markdown_fences():
    assert _safe_json_parse('```json\n{"a": 3}\n```') == {"a": 3}


def test_ignores_text_around_json():
    assert _safe_json_parse('Sure! Here is the answer: {"a": 4} Hope that helps.') == {"a": 4}


def test_think_and_fences_together():
    assert _safe_json_parse('<think>hmm</think>\n```json\n{"a": 5}\n```') == {"a": 5}


def test_garbage_raises():
    with pytest.raises(json.JSONDecodeError):
        _safe_json_parse("no json here at all")


# --- _run_agent: good and broken replies ------------------------------------

def test_run_agent_accepts_valid_reply():
    rec = _run_agent(fake_call(_valid_reply(top="nomic-embed-text")), "Embedding",
                     EMBEDDING_OPTIONS, "Healthcare", "some text")
    assert rec["top_choice"] == "nomic-embed-text"
    assert len(rec["candidates"]) == 3
    assert rec["degraded"] is False and rec["model_used"] == "fake-model"


def test_run_agent_handles_reply_with_think_block():
    rec = _run_agent(fake_call("<think>reasoning</think>" + _valid_reply()), "Embedding",
                     EMBEDDING_OPTIONS, "Healthcare", "text")
    assert rec["degraded"] is False


def test_run_agent_asks_again_after_an_unreadable_reply():
    call = fake_call("I cannot help with that.", _valid_reply())
    rec = _run_agent(call, "Embedding", EMBEDDING_OPTIONS, "Healthcare", "text")
    assert rec["degraded"] is False
    assert len(call.seen) == 2


def test_run_agent_degrades_when_every_reply_is_garbage():
    call = fake_call("I cannot help with that.")
    rec = _run_agent(call, "Embedding", EMBEDDING_OPTIONS, "Healthcare", "text")
    assert len(call.seen) == agents.PARSE_ATTEMPTS
    assert rec["degraded"] is True
    assert rec["top_choice"] == EMBEDDING_OPTIONS[0]
    assert all(c["retrieval_score"] == 0.5 for c in rec["candidates"])
    assert all(c["rationale"].startswith("Fallback") for c in rec["candidates"])


def test_run_agent_degrades_on_out_of_range_score():
    bad = json.loads(_valid_reply())
    bad["candidates"][0]["retrieval_score"] = 1.7  # must be within 0-1
    rec = _run_agent(fake_call(json.dumps(bad)), "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True


def test_run_agent_degrades_on_missing_field():
    bad = json.loads(_valid_reply())
    del bad["top_choice"]
    rec = _run_agent(fake_call(json.dumps(bad)), "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True


def test_degraded_rec_still_lists_every_option():
    rec = _run_agent(fake_call("nonsense"), "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert [c["name"] for c in rec["candidates"]] == EMBEDDING_OPTIONS


def test_run_agent_does_not_raise_when_llm_is_unavailable():
    def down(prompt):
        raise LLMUnavailable("all models failed")
    rec = _run_agent(down, "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True and "unavailable" in rec["error"].lower()


# --- answers that fit the schema but make no sense ----------------------------------

def _reply(candidates, top):
    """candidates: {name: (retrieval, latency, accuracy)} or {name: score}."""
    cands = []
    for name, s in candidates.items():
        r, l, a = s if isinstance(s, tuple) else (s, s, s)
        cands.append({"name": name, "retrieval_score": r, "latency_score": l,
                      "accuracy_score": a, "rationale": "x"})
    return json.dumps({"category": "Embedding", "top_choice": top, "candidates": cands})


A, B, C = EMBEDDING_OPTIONS


def test_invented_option_name_is_rejected_then_retried():
    bad = _reply({A: 0.8, B: 0.7, "OpenAI-ada-002": 0.9}, top=A)
    call = fake_call(bad, _reply({A: 0.8, B: 0.7, C: 0.6}, top=A))
    rec = _run_agent(call, "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is False
    assert len(call.seen) == 2
    # the retry tells the model what was wrong
    assert "OpenAI-ada-002" in call.seen[1] and "rejected" in call.seen[1]


def test_invented_name_every_time_ends_degraded():
    call = fake_call(_reply({A: 0.8, B: 0.7, "Made-Up": 0.9}, top=A))
    rec = _run_agent(call, "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True and "Made-Up" in rec["error"]


def test_missing_option_is_rejected():
    rec = _run_agent(fake_call(_reply({A: 0.8, B: 0.7}, top=A)), "Embedding",
                     EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True and "exactly once" in rec["error"]


def test_duplicated_option_is_rejected():
    rec = _run_agent(fake_call(_reply({A: 0.8, B: 0.7}, top=A).replace(f'"{B}"', f'"{A}"')),
                     "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True


def test_top_choice_that_is_not_an_option_is_rejected():
    rec = _run_agent(fake_call(_reply({A: 0.8, B: 0.7, C: 0.6}, top="Cohere")), "Embedding",
                     EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True and "Cohere" in rec["error"]


def test_top_choice_contradicting_the_scores_is_rejected():
    # says A is best but scores A at 0.5 and C at 0.9
    call = fake_call(_reply({A: 0.5, B: 0.6, C: 0.9}, top=A))
    rec = _run_agent(call, "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True and "must agree" in rec["error"]


def test_top_choice_close_to_the_best_is_accepted():
    # A trails C by 0.05: a judgement call, not a contradiction
    rec = _run_agent(fake_call(_reply({A: 0.80, B: 0.60, C: 0.85}, top=A)), "Embedding",
                     EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is False and rec["top_choice"] == A


def test_weighted_score_decides_not_retrieval_alone():
    # A has the best retrieval, but C is far better on latency and accuracy overall.
    scores = {A: (0.9, 0.1, 0.1), B: (0.5, 0.5, 0.5), C: (0.6, 1.0, 1.0)}
    rec = _run_agent(fake_call(_reply(scores, top=A)), "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is True


def test_option_names_are_matched_ignoring_case_and_spaces():
    reply = _reply({"bge large": 0.8, f" {B.upper()} ": 0.7, C: 0.6}, top="BGE LARGE")
    rec = _run_agent(fake_call(reply), "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["degraded"] is False
    assert [c["name"] for c in rec["candidates"]] == EMBEDDING_OPTIONS  # canonical spelling restored
    assert rec["top_choice"] == A


def test_wrong_category_echoed_by_the_model_is_corrected():
    reply = json.loads(_reply({A: 0.8, B: 0.7, C: 0.6}, top=A))
    reply["category"] = "Banana"
    rec = _run_agent(fake_call(json.dumps(reply)), "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert rec["category"] == "Embedding"


def test_a_consistent_answer_costs_only_one_call():
    call = fake_call(_reply({A: 0.8, B: 0.7, C: 0.6}, top=A))
    _run_agent(call, "Embedding", EMBEDDING_OPTIONS, "Healthcare", "t")
    assert len(call.seen) == 1


# --- domain must be on the allowed list ----------------------------------------------

def test_domain_not_on_the_list_is_rejected_then_retried(monkeypatch):
    prompts = []
    replies = ['{"domain": "Finance", "confidence": 0.9, "reasoning": "money"}',
               '{"domain": "Banking", "confidence": 0.8, "reasoning": "money"}']

    def fake(api_key, model, prompt, **kw):
        prompts.append(prompt)
        return replies[len(prompts) - 1], model

    monkeypatch.setattr(agents, "call_llm", fake)
    state = node_domain_detect({"corpus_text": "bank statement", "api_key": "k", "model": "m"})
    assert state["domain"] == "Banking" and state["domain_confidence"] == 0.8
    assert "Finance" in prompts[1] and "allowed domains" in prompts[1]


def test_domain_never_on_the_list_falls_back_with_zero_confidence(monkeypatch):
    monkeypatch.setattr(agents, "call_llm",
                        lambda *a, **k: ('{"domain": "Finance", "confidence": 0.9, "reasoning": "x"}', "m"))
    state = node_domain_detect({"corpus_text": "bank statement", "api_key": "k", "model": "m"})
    assert state["domain_confidence"] == 0.0 and "Finance" in state["domain_reasoning"]


def test_domain_is_matched_ignoring_case(monkeypatch):
    monkeypatch.setattr(agents, "call_llm",
                        lambda *a, **k: ('{"domain": " healthcare ", "confidence": 0.7, "reasoning": "x"}', "m"))
    state = node_domain_detect({"corpus_text": "claims", "api_key": "k", "model": "m"})
    assert state["domain"] == "Healthcare"


# --- domain detection --------------------------------------------------------

def _state(**kw):
    return {"corpus_text": "claims", "api_key": "k", "model": "m", **kw}


def test_domain_override_skips_the_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called when a domain override is set")
    monkeypatch.setattr(agents, "call_llm", boom)
    state = node_domain_detect(_state(domain_override="Legal"))
    assert state["domain"] == "Legal"
    assert state["domain_confidence"] == 1.0


def test_domain_detect_reads_valid_reply(monkeypatch):
    reply = '{"domain": "Healthcare", "confidence": 0.9, "reasoning": "claims data"}'
    monkeypatch.setattr(agents, "call_llm", lambda *a, **k: (reply, "m"))
    state = node_domain_detect(_state())
    assert state["domain"] == "Healthcare"
    assert state["domain_confidence"] == 0.9


def test_domain_detect_falls_back_on_garbage(monkeypatch):
    monkeypatch.setattr(agents, "call_llm", lambda *a, **k: ("???", "m"))
    state = node_domain_detect(_state())
    assert state["domain_confidence"] == 0.0
    assert state["domain_reasoning"].startswith("Fallback")


def test_domain_detect_survives_llm_outage(monkeypatch):
    def down(*a, **k):
        raise LLMUnavailable("down")
    monkeypatch.setattr(agents, "call_llm", down)
    state = node_domain_detect(_state())
    assert state["domain_confidence"] == 0.0
    assert "unavailable" in state["domain_reasoning"].lower()
