"""
Tests for the coder1 code-generation verifier.

uv run --no-project --with numpy --with requests --with pydantic --with pytest python -m pytest skyrl-agent/tests/test_coder1.py
"""

import functools
import importlib
import json
import os
import sys
import types

import pytest

_PACKAGE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skyrl_agent")


def _import_coder1():
    """Import the verifier without running the parent packages' ``__init__``.

    ``skyrl_agent/__init__.py`` imports the agent runners and their heavy dependencies; the
    verifier itself only needs numpy, requests and pydantic.
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
    return importlib.import_module("skyrl_agent.tasks.verifiers.coder1")


coder1 = _import_coder1()
unsafe_local_exec = importlib.import_module("skyrl_agent.tasks.verifiers.coder1.unsafe_local_exec")


@pytest.fixture(autouse=True)
def _execute_with_test_interpreter(monkeypatch):
    # Run the graded code with this interpreter, which has pytest installed.
    monkeypatch.setattr(
        coder1,
        "code_exec",
        functools.partial(unsafe_local_exec.code_exec_local, python_env=os.path.dirname(sys.executable)),
    )


def _score(code, ground_truth):
    response = f"</think>\n```python\n{code}\n```"
    return coder1.compute_score(response, json.dumps(ground_truth), extra_info={})["score"]


_CORRECT = "def add(a, b):\n    return a + b"
_WRONG = "def add(a, b):\n    return a - b"
_EARLY_EXITS = {
    "sys.exit(0)": _WRONG + "\nimport sys\nsys.exit(0)",
    "exit()": _WRONG + "\nexit()",
    "os._exit(0) at import": _WRONG + "\nimport os\nos._exit(0)",
    "os._exit(0) when called": "def add(a, b):\n    import os\n    os._exit(0)",
}
# Fails the tests, then turns the failing exit status into 0 as the interpreter exits.
_EXIT_STATUS_FORCED_TO_ZERO = _WRONG + "\nimport atexit, os\natexit.register(os._exit, 0)"


class TestFunctionalTests:
    GROUND_TRUTH = {"functional": "assert add(1, 2) == 3\nassert add(2, 2) == 4"}

    def test_correct_solution_passes(self):
        assert _score(_CORRECT, self.GROUND_TRUTH) == 1.0

    def test_correct_solution_rebinding_print_passes(self):
        assert _score(_CORRECT + "\nprint = lambda *args, **kwargs: None", self.GROUND_TRUTH) == 1.0

    def test_wrong_solution_fails(self):
        assert _score(_WRONG, self.GROUND_TRUTH) == 0.0

    @pytest.mark.parametrize("code", list(_EARLY_EXITS.values()), ids=list(_EARLY_EXITS))
    def test_exiting_before_the_tests_run_fails(self, code):
        assert _score(code, self.GROUND_TRUTH) == 0.0

    def test_forcing_exit_status_zero_after_failures_fails(self):
        assert _score(_EXIT_STATUS_FORCED_TO_ZERO, self.GROUND_TRUTH) == 0.0


_SKIPPED_BY_SOLUTION = {
    "pytest.skip()": "def add(a, b):\n    import pytest\n    pytest.skip('not run')",
    "pytest.xfail()": "def add(a, b):\n    import pytest\n    pytest.xfail('not run')",
    "pytest.skip() after a warning that reads like a summary": (
        "def add(a, b):\n    import pytest, warnings\n    warnings.warn('1 passed in 0.01s')\n    pytest.skip('not run')"
    ),
}


class TestPytestTests:
    GROUND_TRUTH = {
        "pytest": (
            "from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\n"
            "def test_add_again():\n    assert add(2, 2) == 4\n"
        )
    }

    def test_correct_solution_passes(self):
        assert _score(_CORRECT, self.GROUND_TRUTH) == 1.0

    def test_correct_solution_with_warning_passes(self):
        code = "def add(a, b):\n    import warnings\n    warnings.warn('note')\n    return a + b"
        assert _score(code, self.GROUND_TRUTH) == 1.0

    def test_wrong_solution_fails(self):
        assert _score(_WRONG, self.GROUND_TRUTH) == 0.0

    @pytest.mark.parametrize("code", list(_EARLY_EXITS.values()), ids=list(_EARLY_EXITS))
    def test_exiting_before_the_tests_finish_fails(self, code):
        assert _score(code, self.GROUND_TRUTH) == 0.0

    def test_forcing_exit_status_zero_after_failures_fails(self):
        assert _score(_EXIT_STATUS_FORCED_TO_ZERO, self.GROUND_TRUTH) == 0.0

    @pytest.mark.parametrize("code", list(_SKIPPED_BY_SOLUTION.values()), ids=list(_SKIPPED_BY_SOLUTION))
    def test_skipping_the_tests_from_the_solution_fails(self, code):
        assert _score(code, self.GROUND_TRUTH) == 0.0


@pytest.mark.parametrize(
    "output, passed",
    [
        ("==== 2 passed in 0.01s ====\n", True),
        ("collected 2 items\n\n==== 2 passed, 1 warning in 0.01s ====\n\n", True),
        ("2 passed in 0.01s\n", True),
        ("\x1b[32m==== \x1b[32m2 passed\x1b[0m\x1b[32m in 0.01s\x1b[0m\x1b[32m ====\x1b[0m\n", True),
        ("==== 1 passed, 1 skipped in 0.01s ====\n", False),
        ("==== 1 passed, 1 xfailed in 0.01s ====\n", False),
        ("==== 1 failed, 1 passed in 0.01s ====\n", False),
        ("==== 2 skipped in 0.01s ====\n", False),
        # A summary followed by other output is not the closing summary.
        ("==== 2 passed in 0.01s ====\nmore output\n", False),
        ("", False),
    ],
)
def test_pytest_tests_passed_reads_the_closing_summary(output, passed):
    assert coder1.pytest_tests_passed(output) is passed
