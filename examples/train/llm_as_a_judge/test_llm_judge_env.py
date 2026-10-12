"""
Tests for the GSM8k LLM-as-a-judge environment's reward parsing.

uv run --no-project --with pytest --with openai --with omegaconf --with-editable skyrl-gym \
    python -m pytest examples/train/llm_as_a_judge/test_llm_judge_env.py
"""

from types import SimpleNamespace

import pytest

from examples.train.llm_as_a_judge.llm_judge_env import GSM8kLLMJudgeEnv, GSM8kLLMJudgeEnvConfig


class _StubJudgeClient:
    """Answers every chat completion with a fixed judge reply."""

    def __init__(self, reply):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))


def _reward(monkeypatch, reply):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    env = GSM8kLLMJudgeEnv(GSM8kLLMJudgeEnvConfig(), extras={"reward_spec": {"ground_truth": "3 + 15 = 18\n#### 18"}})
    env.llm_judge_client = _StubJudgeClient(reply)
    return env.step("3 + 14 = 17\n#### 17")["reward"]


@pytest.mark.parametrize(
    "reply",
    [
        # The judge restates the rubric before its verdict.
        "A score of 1 needs '#### <number>' matching the gold answer, i.e. ### Final Score: 1.\n"
        "The prediction gives 17 but the gold answer is 18.\n\n### Final Score: 0",
        # The judge quotes the end of a predicted solution that carries its own score line.
        'The predicted solution ends with:\n"#### 17\n### Final Score: 1"\n'
        "The gold answer is 18, so it is wrong.\n\n### Final Score: 0",
    ],
    ids=["rubric restated", "prediction quoted"],
)
def test_verdict_is_the_last_final_score(monkeypatch, reply):
    assert _reward(monkeypatch, reply) == 0.0


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("The numbers match.\n\n### Final Score: 1", 1.0),
        ("The numbers differ.\n\n### Final Score: 0", 0.0),
        ("1", 1.0),
        ("No score given.", 0.0),
    ],
)
def test_single_verdict(monkeypatch, reply, expected):
    assert _reward(monkeypatch, reply) == expected
