"""Retries, model fallback, caching and usage tracking in agents.py. No network, no real waiting."""

import json

import httpx
import pytest
from groq import (
    APIConnectionError, AuthenticationError, BadRequestError, InternalServerError,
    NotFoundError, RateLimitError,
)
from types import SimpleNamespace

import agents
from agents import call_llm, LLMUnavailable, GROQ_MODELS

PRIMARY, SECOND = GROQ_MODELS[0], GROQ_MODELS[1]


def _err(cls, status, message="boom", headers=None):
    req = httpx.Request("POST", "https://api.groq.com/x")
    return cls(message, response=httpx.Response(status, headers=headers or {}, request=req), body=None)


def _ok(text="hello", inp=10, out=5):
    return SimpleNamespace(content=text, usage_metadata={"input_tokens": inp, "output_tokens": out})


class Script:
    """Stands in for agents._invoke: each call pops the next outcome (an exception is raised)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.models = []

    def __call__(self, api_key, model, prompt, temperature):
        self.models.append(model)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    agents.clear_cache()
    agents._NO_JSON_MODE.clear()
    waits = []
    monkeypatch.setattr(agents, "_sleep", waits.append)
    yield waits
    agents.clear_cache()
    agents._NO_JSON_MODE.clear()


def use(monkeypatch, *outcomes):
    script = Script(*outcomes)
    monkeypatch.setattr(agents, "_invoke", script)
    return script


# --- call_llm ----------------------------------------------------------------

def test_success_returns_text_and_model_and_counts_tokens(monkeypatch):
    use(monkeypatch, _ok("hi", inp=100, out=40))
    before = agents.get_usage()
    assert call_llm("k", PRIMARY, "p") == ("hi", PRIMARY)
    used = agents.usage_since(before)
    assert (used["input_tokens"], used["output_tokens"], used["calls"]) == (100, 40, 1)


def test_rate_limit_is_retried_after_waiting(monkeypatch, isolate):
    script = use(monkeypatch, _err(RateLimitError, 429), _ok())
    before = agents.get_usage()
    assert call_llm("k", PRIMARY, "p")[1] == PRIMARY
    assert script.models == [PRIMARY, PRIMARY]
    assert len(isolate) == 1 and isolate[0] > 0
    assert agents.usage_since(before)["retries"] == 1


def test_backoff_grows_between_attempts(monkeypatch, isolate):
    use(monkeypatch, _err(RateLimitError, 429), _err(RateLimitError, 429), _err(RateLimitError, 429), _ok())
    call_llm("k", PRIMARY, "p")
    assert isolate[0] < isolate[1] < isolate[2]


def test_retry_after_header_is_honoured(monkeypatch, isolate):
    use(monkeypatch, _err(RateLimitError, 429, headers={"retry-after": "9"}), _ok())
    call_llm("k", PRIMARY, "p")
    assert isolate[0] >= 9


def test_very_long_retry_after_switches_model_instead_of_waiting(monkeypatch, isolate):
    script = use(monkeypatch, _err(RateLimitError, 429, headers={"retry-after": "3600"}), _ok("from second"))
    text, model = call_llm("k", PRIMARY, "p")
    assert (text, model) == ("from second", SECOND)
    assert isolate == []  # never slept
    assert script.models == [PRIMARY, SECOND]


def test_exhausted_retries_fall_back_to_next_model(monkeypatch):
    use(monkeypatch, *[_err(InternalServerError, 503)] * agents.MAX_ATTEMPTS, _ok("rescued"))
    before = agents.get_usage()
    assert call_llm("k", PRIMARY, "p") == ("rescued", SECOND)
    assert agents.usage_since(before)["fallbacks"] == 1


def test_model_not_found_skips_to_next_model_without_waiting(monkeypatch, isolate):
    script = use(monkeypatch, _err(NotFoundError, 404), _ok())
    assert call_llm("k", PRIMARY, "p")[1] == SECOND
    assert script.models == [PRIMARY, SECOND] and isolate == []


def test_connection_error_is_retried(monkeypatch):
    req = httpx.Request("POST", "https://api.groq.com/x")
    script = use(monkeypatch, APIConnectionError(request=req), _ok())
    assert call_llm("k", PRIMARY, "p")[1] == PRIMARY
    assert len(script.models) == 2


def test_bad_api_key_fails_immediately_without_trying_other_models(monkeypatch):
    script = use(monkeypatch, _err(AuthenticationError, 401))
    with pytest.raises(AuthenticationError):
        call_llm("k", PRIMARY, "p")
    assert script.models == [PRIMARY]


def test_every_model_failing_raises_llm_unavailable(monkeypatch):
    use(monkeypatch, *[_err(NotFoundError, 404)] * len(GROQ_MODELS))
    with pytest.raises(LLMUnavailable):
        call_llm("k", PRIMARY, "p")


def test_fallback_can_be_disabled(monkeypatch):
    script = use(monkeypatch, *[_err(NotFoundError, 404)])
    with pytest.raises(LLMUnavailable):
        call_llm("k", PRIMARY, "p", allow_fallback=False)
    assert script.models == [PRIMARY]


def test_json_validate_failed_is_retried(monkeypatch):
    err = _err(BadRequestError, 400, "json_validate_failed: model produced invalid JSON")
    script = use(monkeypatch, err, _ok())
    assert call_llm("k", PRIMARY, "p")[1] == PRIMARY
    assert len(script.models) == 2


def test_other_bad_requests_are_not_retried_on_the_same_model(monkeypatch):
    script = use(monkeypatch, _err(BadRequestError, 400, "invalid parameter"), _ok())
    assert call_llm("k", PRIMARY, "p")[1] == SECOND
    assert script.models == [PRIMARY, SECOND]


# --- JSON mode negotiation ----------------------------------------------------------

def test_model_without_json_mode_is_retried_plain_and_remembered(monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, json_mode):
            self.json_mode = json_mode

        def invoke(self, prompt):
            calls.append(self.json_mode)
            if self.json_mode:
                raise _err(BadRequestError, 400, "response_format is not supported with this model")
            return _ok("plain")

    monkeypatch.setattr(agents, "get_llm", lambda key, model, temp, json_mode=False: FakeLLM(json_mode))
    assert agents._invoke("k", "some-model", "p", 0.2).content == "plain"
    assert calls == [True, False]
    assert "some-model" in agents._NO_JSON_MODE
    agents._invoke("k", "some-model", "p", 0.2)
    assert calls[-1] is False and len(calls) == 3  # next time it goes straight to plain


# --- cost settings ----------------------------------------------------------------

def test_gpt_oss_uses_low_reasoning_effort_and_a_token_cap():
    llm = agents.get_llm("k", "openai/gpt-oss-120b")
    assert llm.reasoning_effort == "low"
    assert llm.max_tokens == agents.MAX_OUTPUT_TOKENS
    assert llm.max_retries == 0  # retrying is done in call_llm, not twice


def test_json_mode_is_requested_when_asked():
    llm = agents.get_llm("k", "openai/gpt-oss-20b", json_mode=True)
    assert llm.model_kwargs["response_format"] == {"type": "json_object"}


# --- caching ------------------------------------------------------------------------

def _agent_json(category, options):
    return json.dumps({"category": category, "top_choice": options[0], "candidates": [
        {"name": o, "retrieval_score": 0.8, "latency_score": 0.5, "accuracy_score": 0.7, "rationale": "ok"}
        for o in options]})


def _counting_llm(monkeypatch, fail=False):
    calls = []

    def fake(api_key, model, prompt, temperature=0.2, allow_fallback=True):
        calls.append(prompt)
        if fail:
            raise LLMUnavailable("down")
        return _agent_json("Embedding", agents.EMBEDDING_OPTIONS), model

    monkeypatch.setattr(agents, "call_llm", fake)
    return calls


def _state(**kw):
    return {"corpus_text": "doc", "api_key": "k", "model": "m", "domain": "Healthcare", **kw}


def test_same_question_is_answered_from_cache(monkeypatch):
    calls = _counting_llm(monkeypatch)
    first = agents.node_embedding(_state())["embedding_rec"]
    before = agents.get_usage()
    second = agents.node_embedding(_state())["embedding_rec"]
    assert len(calls) == 1
    assert second == first
    assert agents.usage_since(before)["cache_hits"] == 1


def test_cached_answer_cannot_be_mutated_by_callers(monkeypatch):
    _counting_llm(monkeypatch)
    agents.node_embedding(_state())["embedding_rec"]["top_choice"] = "TAMPERED"
    assert agents.node_embedding(_state())["embedding_rec"]["top_choice"] != "TAMPERED"


def test_different_document_model_domain_or_feedback_is_not_a_cache_hit(monkeypatch):
    calls = _counting_llm(monkeypatch)
    agents.node_embedding(_state())
    agents.node_embedding(_state(corpus_text="other doc"))
    agents.node_embedding(_state(model="other-model"))
    agents.node_embedding(_state(domain="Legal"))
    agents.node_embedding(_state(feedback={"Embedding": "try something else"}))
    assert len(calls) == 5


def test_human_rerun_bypasses_the_cache(monkeypatch):
    calls = _counting_llm(monkeypatch)
    agents.node_embedding(_state())
    agents.node_embedding(_state(bypass_cache=True))
    assert len(calls) == 2


def test_degraded_results_are_never_cached(monkeypatch):
    calls = _counting_llm(monkeypatch, fail=True)
    assert agents.node_embedding(_state())["embedding_rec"]["degraded"] is True
    agents.node_embedding(_state())
    assert len(calls) == 2  # asked again, because the failure wasn't remembered


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(agents, "CACHE_SIZE", 3)
    for i in range(6):
        agents._cache_put(f"k{i}", i)
    assert len(agents._CACHE) == 3 and agents._cache_get("k0") is None and agents._cache_get("k5") == 5


# --- whole pipeline with a fake LLM -----------------------------------------------------

OPTIONS = {"Embedding": agents.EMBEDDING_OPTIONS, "Chunking": agents.CHUNKING_OPTIONS,
           "Tokenizer": agents.TOKENIZER_OPTIONS, "Indexing": agents.INDEX_OPTIONS}


def _pipeline_llm(monkeypatch, broken=()):
    def fake(api_key, model, prompt, temperature=0.2, allow_fallback=True):
        if "Classify the following" in prompt:
            return '{"domain": "Healthcare", "confidence": 0.9, "reasoning": "claims"}', model
        for cat, opts in OPTIONS.items():
            if f"You are the {cat} specialist" in prompt:
                if cat in broken:
                    raise LLMUnavailable("down")
                return _agent_json(cat, opts), model
        raise AssertionError("unexpected prompt")

    monkeypatch.setattr(agents, "call_llm", fake)


def test_pipeline_runs_all_agents_and_reports_usage(monkeypatch):
    _pipeline_llm(monkeypatch)
    out = agents.run_pipeline("some document text", "k", model="m")
    assert out["domain"] == "Healthcare"
    assert all(out[f"{c.lower()}_rec"]["candidates"] for c in OPTIONS)
    assert out["degraded"] == []
    assert set(out["usage"]) >= {"input_tokens", "output_tokens", "retries", "cache_hits"}


def test_pipeline_survives_one_agent_failing(monkeypatch):
    _pipeline_llm(monkeypatch, broken=("Tokenizer",))
    out = agents.run_pipeline("some document text", "k", model="m")
    assert out["degraded"] == ["Tokenizer"]
    assert out["tokenizer_rec"]["degraded"] is True
    assert out["embedding_rec"]["degraded"] is False  # the other agents still finished


def test_pipeline_second_run_on_same_document_is_served_from_cache(monkeypatch):
    _pipeline_llm(monkeypatch)
    agents.run_pipeline("same text", "k", model="m")
    second = agents.run_pipeline("same text", "k", model="m")
    assert second["usage"]["cache_hits"] == 5  # domain + 4 agents


def test_pipeline_can_skip_the_cache(monkeypatch):
    _pipeline_llm(monkeypatch)
    agents.run_pipeline("same text", "k", model="m")
    second = agents.run_pipeline("same text", "k", model="m", use_cache=False)
    assert second["usage"]["cache_hits"] == 0
