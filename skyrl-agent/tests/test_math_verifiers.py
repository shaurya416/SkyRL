"""
Tests for the PRIME math grader and the naive DAPO math verifier.

uv run --no-project --with sympy --with pylatexenc --with pytest python -m pytest skyrl-agent/tests/test_math_verifiers.py
"""

import importlib
import os
import sys
import types

import pytest

_PACKAGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skyrl_agent")


def _import_verifier(module):
    """Import a verifier without running the parent packages' ``__init__``.

    ``skyrl_agent/__init__.py`` imports the agent runners and their heavy dependencies; the
    math verifiers only need sympy and pylatexenc.
    """
    for name, subdir in [
        ("skyrl_agent", ""),
        ("skyrl_agent.tasks", "tasks"),
        ("skyrl_agent.tasks.verifiers", os.path.join("tasks", "verifiers")),
    ]:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [os.path.join(_PACKAGE_ROOT, subdir)]
            sys.modules[name] = package
    return importlib.import_module(f"skyrl_agent.tasks.verifiers.{module}")


grader = _import_verifier("prime_math.grader")
prime_math = _import_verifier("prime_math")
naive_dapo = _import_verifier("naive_dapo")

# A \pi answer is evaluated as arithmetic. These answers carry code instead; they avoid "_"
# because an answer containing it is first parsed as a base-n number.
_READS_REFERENCE = (
    "\\pi*0 or getattr(getattr(contextlib.sys, chr(95)+'getframe')(3), 'f'+chr(95)+'locals')['reference']"
)


def _writes_file(path):
    return "\\pi+len(open(bytes(%r).decode(), 'w').write('x') and '')" % list(str(path).encode())


class TestPrimeMathGrader:
    @pytest.mark.parametrize("reference", ["17", "\\frac{3}{8}", "(1, 2)"])
    def test_pi_answer_cannot_read_the_reference(self, reference):
        assert grader.math_equal(_READS_REFERENCE, reference) is False
        assert not prime_math.compute_score(f"\\boxed{{{_READS_REFERENCE}}}", reference)["score"]

    def test_pi_answer_does_not_run_code(self, tmp_path):
        marker = tmp_path / "ran"
        grader.math_equal(_writes_file(marker), "3")
        prime_math.compute_score(f"\\boxed{{{_writes_file(marker)}}}", "3")
        assert not marker.exists()

    @pytest.mark.parametrize(
        "prediction, reference",
        [
            ("6.2832", "2\\pi"),
            ("1.5708", "\\pi/2"),
            ("-3.1416", "-\\pi"),
            ("9.8696", "\\pi**2"),
            ("[3.1416, 6.2832]", "[\\pi, 2\\pi]"),
            ("(3.1416, 6.2832)", "(\\pi, 2\\pi)"),
            ("2\\pi", "6.2832"),
            ("{\\pi}", "\\pi"),
            ("{2\\pi}", "2\\pi"),
        ],
    )
    def test_pi_arithmetic_is_still_evaluated(self, prediction, reference):
        assert grader.math_equal(prediction, reference) is True

    def test_wrong_pi_value_is_rejected(self):
        assert grader.math_equal("6.5", "2\\pi") is False

    def test_eval_numeric_expression_rejects_code(self):
        assert grader.eval_numeric_expression("[1*3.5, -2 ** 2]") == [3.5, -4.0]
        for expression in [
            "len('x')",
            "x + 1",
            "(1).real",
            "'a' * 3",
            "[i for i in (1, 2)]",
            "1 << 1000",
            "[0] * 1000000000",
            "-(1, 2)",
        ]:
            with pytest.raises(ValueError):
                grader.eval_numeric_expression(expression)


class TestNaiveDapo:
    @pytest.mark.parametrize("answer", ["6.2832", "6.28"])
    def test_numeric_value_of_pi_answer_is_correct(self, answer):
        assert naive_dapo.compute_score(f"\\boxed{{{answer}}}", "2\\pi", extra_info={})["score"] == 1.0

    def test_wrong_value_of_pi_answer_is_incorrect(self):
        assert naive_dapo.compute_score("\\boxed{6.5}", "2\\pi", extra_info={})["score"] == 0.0

    @pytest.mark.parametrize("reference", ["17", "\\frac{3}{8}"])
    def test_pi_answer_cannot_read_the_reference(self, reference):
        response = f"\\boxed{{{_READS_REFERENCE}}}"
        assert naive_dapo.compute_score(response, reference, extra_info={})["score"] == 0.0
