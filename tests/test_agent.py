"""ProductionRAGAgent: primary path, fallback path, and what a total failure looks like."""

import pytest
from langchain_core.messages import AIMessage

from app import agent as agent_module
from app.config import get_settings


def make_agent(monkeypatch, primary, fallback):
    """
    Build the agent with fake models. The first model constructed is the primary and the second
    the fallback. Each behaviour is either an answer string or an exception to raise.
    """
    behaviours = iter([primary, fallback])

    class FakeLLM:
        def __init__(self, **kwargs):
            self.behaviour = next(behaviours)
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if isinstance(self.behaviour, Exception):
                raise self.behaviour
            return AIMessage(content=self.behaviour)

    monkeypatch.setattr(agent_module, "ChatOpenAI", FakeLLM)
    return agent_module.ProductionRAGAgent()


def test_the_primary_model_answers_when_it_works(monkeypatch):
    agent = make_agent(monkeypatch, "primary answer", "fallback answer")

    assert agent.invoke("What is a pod?") == {
        "response": "primary answer",
        "error": None,
        "model_used": "primary",
    }
    assert agent.fallback_llm.calls == 0


def test_the_fallback_model_answers_when_the_primary_fails(monkeypatch):
    agent = make_agent(monkeypatch, RuntimeError("primary is down"), "fallback answer")

    result = agent.invoke("What is a pod?")

    assert result["response"] == "fallback answer"
    assert result["error"] is None
    assert result["model_used"] == "secondary"


def test_when_every_model_fails_the_result_carries_an_error_flag(monkeypatch):
    agent = make_agent(monkeypatch, RuntimeError("401 bad key sk-secret"), RuntimeError("503 upstream down"))

    result = agent.invoke("What is a pod?")

    assert result["error"], "a total failure must be distinguishable from an answer"


def test_a_total_failure_does_not_put_the_exception_text_in_the_response(monkeypatch):
    agent = make_agent(monkeypatch, RuntimeError("401 bad key sk-secret"), RuntimeError("503 upstream down"))

    result = agent.invoke("What is a pod?")

    assert "sk-secret" not in result["response"]
    assert "503 upstream down" not in result["response"]


def test_the_agent_stops_trying_after_max_retries(monkeypatch):
    agent = make_agent(monkeypatch, RuntimeError("down"), RuntimeError("down"))

    agent.invoke("What is a pod?")

    assert agent.primary_llm.calls == 1
    assert 1 <= agent.fallback_llm.calls <= get_settings().max_retries
