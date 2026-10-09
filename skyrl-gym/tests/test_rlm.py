"""
Tests for RLM lm_query / llm_query_batched / rlm_query / rlm_query_batched.

All tests use mock callbacks — no real LLM or inference engine required.
"""

import skyrl_gym
from skyrl_gym.envs.registration import register, registry
from skyrl_gym.envs.rlm.env import BaseRLMEnv
from skyrl_gym.envs.rlm.repl import PersistentREPL


# ---------------------------------------------------------------------------
# Test subclass — BaseRLMEnv is abstract (not registered); these tests only
# need the inherited REPL / lm_callback / step-loop plumbing, so subclass with
# trivial defaults and register under a test-only id.
# ---------------------------------------------------------------------------


class _TestRLMEnv(BaseRLMEnv):
    pass


_TEST_ENV_ID = "_test_rlm"
if _TEST_ENV_ID not in registry:
    register(id=_TEST_ENV_ID, entry_point=_TestRLMEnv)


# ---------------------------------------------------------------------------
# Shared mock callbacks
# ---------------------------------------------------------------------------


def _mock_lm_callback(prompts):
    """Echo each prompt back with a prefix so tests can verify routing."""
    return [f"response:{p}" for p in prompts]


def _mock_subcall_fn(prompt, context=None):
    """Simulate a child RLM: returns a fixed string based on the prompt."""
    return f"child_result:{prompt}"


# ---------------------------------------------------------------------------
# 1. PersistentREPL — llm_query / llm_query_batched
# ---------------------------------------------------------------------------


class TestLLMQuery:
    def test_llm_query_registered_when_callback_provided(self):
        repl = PersistentREPL(lm_callback=_mock_lm_callback)
        assert "llm_query" in repl.globals
        assert "llm_query_batched" in repl.globals
        assert "rlm_query" in repl.globals
        assert "rlm_query_batched" in repl.globals

    def test_llm_query_not_registered_without_callback(self):
        repl = PersistentREPL()
        assert "llm_query" not in repl.globals
        assert "llm_query_batched" not in repl.globals
        assert "rlm_query" not in repl.globals
        assert "rlm_query_batched" not in repl.globals

    def test_llm_query_returns_response(self):
        repl = PersistentREPL(lm_callback=_mock_lm_callback)
        result = repl.execute("result = llm_query('hello')\nprint(result)")
        assert "response:hello" in result.stdout

    def test_llm_query_batched_returns_list(self):
        repl = PersistentREPL(lm_callback=_mock_lm_callback)
        result = repl.execute("results = llm_query_batched(['a', 'b', 'c'])\nprint(results)")
        assert "response:a" in result.stdout
        assert "response:b" in result.stdout
        assert "response:c" in result.stdout

    def test_llm_query_batched_preserves_order(self):
        repl = PersistentREPL(lm_callback=_mock_lm_callback)
        result = repl.execute("results = llm_query_batched(['x', 'y'])\n" "print(results[0])\n" "print(results[1])")
        lines = result.stdout.strip().splitlines()
        assert lines[0] == "response:x"
        assert lines[1] == "response:y"

    def test_llm_query_error_handling(self):
        def bad_callback(prompts):
            raise RuntimeError("inference error")

        repl = PersistentREPL(lm_callback=bad_callback)
        result = repl.execute("r = llm_query('test')\nprint(r)")
        assert "Error: LM query failed" in result.stdout

    def test_llm_query_batched_error_handling(self):
        def bad_callback(prompts):
            raise RuntimeError("batch error")

        repl = PersistentREPL(lm_callback=bad_callback)
        result = repl.execute("r = llm_query_batched(['a', 'b'])\nprint(r)")
        assert "Error: LM query failed" in result.stdout

    def test_scaffold_restored_after_overwrite_attempt(self):
        """Model code cannot permanently overwrite llm_query."""
        repl = PersistentREPL(lm_callback=_mock_lm_callback)
        repl.execute("llm_query = lambda p, m=None: 'hijacked'")
        # After execution scaffold is restored
        assert repl.globals["llm_query"] == repl._llm_query


# ---------------------------------------------------------------------------
# 2. PersistentREPL — rlm_query / rlm_query_batched
# ---------------------------------------------------------------------------


class TestRLMQuery:
    def test_rlm_query_uses_subcall_fn_when_provided(self):
        repl = PersistentREPL(lm_callback=_mock_lm_callback, subcall_fn=_mock_subcall_fn)
        result = repl.execute("r = rlm_query('do subtask')\nprint(r)")
        assert "child_result:do subtask" in result.stdout

    def test_rlm_query_falls_back_to_llm_query_without_subcall_fn(self):
        repl = PersistentREPL(lm_callback=_mock_lm_callback)
        result = repl.execute("r = rlm_query('fallback prompt')\nprint(r)")
        assert "response:fallback prompt" in result.stdout

    def test_rlm_query_batched_uses_subcall_fn(self):
        repl = PersistentREPL(lm_callback=_mock_lm_callback, subcall_fn=_mock_subcall_fn)
        result = repl.execute("results = rlm_query_batched(['p1', 'p2'])\n" "print(results[0])\n" "print(results[1])")
        assert "child_result:p1" in result.stdout
        assert "child_result:p2" in result.stdout

    def test_rlm_query_batched_falls_back_to_llm_query_batched(self):
        repl = PersistentREPL(lm_callback=_mock_lm_callback)
        result = repl.execute("results = rlm_query_batched(['p1', 'p2'])\n" "print(results[0])\n" "print(results[1])")
        assert "response:p1" in result.stdout
        assert "response:p2" in result.stdout

    def test_rlm_query_batched_preserves_order(self):
        """Results must come back in the same order as input prompts."""
        import time

        call_order = []

        def ordered_subcall(prompt, context=None):
            call_order.append(prompt)
            # Introduce artificial delay for first prompt to expose ordering bugs
            if prompt == "first":
                time.sleep(0.05)
            return f"result:{prompt}"

        repl = PersistentREPL(lm_callback=_mock_lm_callback, subcall_fn=ordered_subcall)
        result = repl.execute(
            "results = rlm_query_batched(['first', 'second', 'third'])\n" "for r in results:\n" "    print(r)"
        )
        lines = result.stdout.strip().splitlines()
        assert lines == ["result:first", "result:second", "result:third"]

    def test_rlm_query_single_prompt_no_threadpool(self):
        """Single-element list takes the sequential path (len <= 1)."""
        repl = PersistentREPL(lm_callback=_mock_lm_callback, subcall_fn=_mock_subcall_fn)
        result = repl.execute("r = rlm_query_batched(['solo'])\nprint(r[0])")
        assert "child_result:solo" in result.stdout

    def test_rlm_query_batched_partial_failure(self):
        """A failing subcall for one prompt should not crash the others."""
        call_count = {"n": 0}

        def flaky_subcall(prompt, context=None):
            call_count["n"] += 1
            if prompt == "bad":
                raise ValueError("intentional failure")
            return f"ok:{prompt}"

        repl = PersistentREPL(lm_callback=_mock_lm_callback, subcall_fn=flaky_subcall)
        result = repl.execute(
            "results = rlm_query_batched(['good', 'bad', 'also_good'])\n" "print(results[0])\n" "print(results[2])"
        )
        assert "ok:good" in result.stdout
        assert "ok:also_good" in result.stdout

    def test_rlm_query_error_handling(self):
        def bad_subcall(prompt, context=None):
            raise RuntimeError("subcall exploded")

        repl = PersistentREPL(lm_callback=_mock_lm_callback, subcall_fn=bad_subcall)
        result = repl.execute("r = rlm_query('oops')\nprint(r)")
        assert "Error: RLM query failed" in result.stdout


# ---------------------------------------------------------------------------
# 3. RLMEnv integration
# ---------------------------------------------------------------------------


class TestRLMEnvWithLMCallback:
    def _make_env(self, lm_callback=None, subcall_fn=None):
        extras = {
            "reward_spec": {"ground_truth": "Paris"},
            "max_turns": 5,
            "extra_info": {"context_text": "The capital of France is Paris."},
        }
        if lm_callback is not None:
            extras["lm_callback"] = lm_callback
        if subcall_fn is not None:
            extras["subcall_fn"] = subcall_fn
        return skyrl_gym.make(_TEST_ENV_ID, extras=extras)

    def test_env_init_without_callback(self):
        env = self._make_env()
        prompt = [{"role": "user", "content": "What is the capital of France?"}]
        messages, _ = env.init(prompt)
        assert any(m["role"] == "system" for m in messages)
        # llm_query should NOT appear in the system prompt
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert "llm_query" not in system_content

    def test_env_init_with_callback_updates_system_prompt(self):
        env = self._make_env(lm_callback=_mock_lm_callback)
        prompt = [{"role": "user", "content": "What is the capital of France?"}]
        messages, _ = env.init(prompt)
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert "llm_query" in system_content
        assert "llm_query_batched" in system_content
        assert "rlm_query" in system_content
        assert "rlm_query_batched" in system_content

    def test_step_with_llm_query_in_repl(self):
        env = self._make_env(lm_callback=_mock_lm_callback)
        prompt = [{"role": "user", "content": "What is the capital?"}]
        env.init(prompt)

        action = (
            "Let me ask the LM.\n\n"
            "```repl\n"
            "answer = llm_query('What is the capital of France?')\n"
            "print(answer)\n"
            "```"
        )
        step_out = env.step(action)
        # The REPL output should contain the mock callback's response
        obs_text = " ".join(o["content"] for o in step_out["observations"])
        assert "response:What is the capital of France?" in obs_text
        assert step_out["done"] is False

    def test_step_with_llm_query_batched_in_repl(self):
        env = self._make_env(lm_callback=_mock_lm_callback)
        prompt = [{"role": "user", "content": "Multi question"}]
        env.init(prompt)

        action = "```repl\n" "results = llm_query_batched(['q1', 'q2'])\n" "print(results)\n" "```"
        step_out = env.step(action)
        obs_text = " ".join(o["content"] for o in step_out["observations"])
        assert "response:q1" in obs_text
        assert "response:q2" in obs_text

    def test_step_with_rlm_query_uses_subcall_fn(self):
        env = self._make_env(lm_callback=_mock_lm_callback, subcall_fn=_mock_subcall_fn)
        prompt = [{"role": "user", "content": "Deep question"}]
        env.init(prompt)

        action = "```repl\n" "r = rlm_query('solve this subtask')\n" "print(r)\n" "```"
        step_out = env.step(action)
        obs_text = " ".join(o["content"] for o in step_out["observations"])
        assert "child_result:solve this subtask" in obs_text

    def test_step_with_rlm_query_falls_back_without_subcall_fn(self):
        env = self._make_env(lm_callback=_mock_lm_callback)
        prompt = [{"role": "user", "content": "question"}]
        env.init(prompt)

        action = "```repl\n" "r = rlm_query('fallback')\n" "print(r)\n" "```"
        step_out = env.step(action)
        obs_text = " ".join(o["content"] for o in step_out["observations"])
        assert "response:fallback" in obs_text

    def test_llm_query_not_available_without_callback(self):
        """Without lm_callback, calling llm_query in the REPL raises a NameError."""
        env = self._make_env()
        prompt = [{"role": "user", "content": "question"}]
        env.init(prompt)

        action = "```repl\n" "r = llm_query('test')\n" "```"
        step_out = env.step(action)
        obs_text = " ".join(o["content"] for o in step_out["observations"])
        assert "NameError" in obs_text or "name 'llm_query' is not defined" in obs_text

    def test_final_answer_without_repl_block_ends_episode(self):
        """FINAL(...) given as plain text, with no repl block, is the final answer."""
        env = self._make_env()
        env.init([{"role": "user", "content": "What is the capital of France?"}])

        step_out = env.step("The context says it plainly.\nFINAL(Paris)")
        assert step_out["done"] is True
        assert step_out["observations"] == []
        assert env.get_metrics()["final_answer"] == "Paris"

    def test_final_var_in_separate_response_ends_episode(self):
        """The system prompt's instructed flow: assign in a repl block, then FINAL_VAR in a separate response."""
        env = self._make_env()
        env.init([{"role": "user", "content": "What is the capital of France?"}])

        step_out = env.step("```repl\nanswer = 'Paris'\nprint(answer)\n```")
        assert step_out["done"] is False

        step_out = env.step("The variable holds the answer.\nFINAL_VAR(answer)")
        assert step_out["done"] is True
        assert step_out["reward"] == 1.0
        assert env.get_metrics()["final_answer"] == "Paris"

    def test_no_repl_block_and_no_final_answer_still_prompts_for_code(self):
        env = self._make_env()
        env.init([{"role": "user", "content": "question"}])

        step_out = env.step("I am thinking about it.")
        assert step_out["done"] is False
        assert "No ```repl``` code block found" in step_out["observations"][0]["content"]


# ---------------------------------------------------------------------------
# 4. Concurrent execution
# ---------------------------------------------------------------------------


class TestConcurrentREPLOutput:
    def test_concurrent_executions_capture_only_their_own_output(self):
        """Env steps run on a thread pool; one REPL's prints must not land in another's result."""
        import io
        import sys
        import threading

        a_started, b_started, a_finished = threading.Event(), threading.Event(), threading.Event()
        tools = {
            "a_started": a_started.set,
            "wait_for_b": lambda: b_started.wait(5),
            "a_finished": a_finished.set,
            "b_started": b_started.set,
            "wait_for_a": lambda: a_finished.wait(5),
        }
        repl_a = PersistentREPL(custom_tools=tools)
        repl_b = PersistentREPL(custom_tools=tools)
        results = {}

        process_stdout = io.StringIO()
        saved_stdout = sys.stdout
        sys.stdout = process_stdout
        try:
            # A is mid-execution when B starts, and A keeps printing while B is running.
            thread_a = threading.Thread(
                target=lambda: results.__setitem__(
                    "a", repl_a.execute("print('a1')\na_started()\nwait_for_b()\nprint('a2')\na_finished()")
                )
            )
            thread_a.start()
            assert a_started.wait(5)
            thread_b = threading.Thread(
                target=lambda: results.__setitem__("b", repl_b.execute("b_started()\nwait_for_a()\nprint('b1')"))
            )
            thread_b.start()
            thread_a.join(10)
            thread_b.join(10)
            print("outside")
        finally:
            sys.stdout = saved_stdout

        assert results["a"].stdout == "a1\na2\n"
        assert results["b"].stdout == "b1\n"
        # Output written outside any REPL execution still reaches the process stream.
        assert process_stdout.getvalue() == "outside\n"


# ---------------------------------------------------------------------------
# 5. Timed-out executions (thread fallback)
# ---------------------------------------------------------------------------


def _execute_off_main_thread(repl, code):
    """Run ``repl.execute`` on a worker thread, as env steps do, so the thread-timeout path is used."""
    import threading

    results = {}
    thread = threading.Thread(target=lambda: results.__setitem__("r", repl.execute(code)))
    thread.start()
    thread.join(10)
    return results["r"]


def _execute_timing_out(repl, code):
    """Execute code that times out; returns the result and the abandoned thread still running it."""
    import threading

    before = set(threading.enumerate())
    result = _execute_off_main_thread(repl, code)
    abandoned = [t for t in threading.enumerate() if t not in before and t.is_alive()]
    assert "Timeout" in result.stderr
    assert len(abandoned) == 1
    return result, abandoned[0]


class TestTimedOutExecution:
    """A timed-out execution keeps running on its abandoned thread; nothing it does may reach later turns."""

    def _repl(self, **extra_tools):
        import threading

        release = threading.Event()
        tools = {"block": lambda: release.wait(5), **extra_tools}
        return PersistentREPL(timeout=0.5, custom_tools=tools), release

    def test_thread_path_commits_variables_and_final_answer(self):
        repl, _ = self._repl()
        result = _execute_off_main_thread(repl, "x = 3\nFINAL_VAR('x')")
        assert result.stderr == ""
        assert result.final_answer == "3"
        assert repl.locals["x"] == 3

    def test_final_var_before_timeout_is_reported(self):
        repl, release = self._repl()
        result, abandoned = _execute_timing_out(repl, "x = 1\nFINAL_VAR('x')\nblock()")
        release.set()
        abandoned.join(5)
        assert result.final_answer == "1"

    def test_error_while_persisting_variables_is_not_reported_as_timeout(self):
        repl, _ = self._repl()
        # A non-string key in the namespace makes persisting the variables raise.
        result = _execute_off_main_thread(repl, "vars()[1] = 2")
        assert "Timeout" not in result.stderr
        assert "Traceback" in result.stderr

    def test_late_assignment_does_not_overwrite_newer_value(self):
        repl, release = self._repl()
        _, abandoned = _execute_timing_out(repl, "block()\nanswer = 'stale'")
        _execute_off_main_thread(repl, "answer = 'fresh'")
        release.set()
        abandoned.join(5)

        result = _execute_off_main_thread(repl, "FINAL_VAR('answer')")
        assert result.final_answer == "fresh"

    def test_late_final_var_is_not_reported_by_next_execution(self):
        repl, release = self._repl()
        timed_out, abandoned = _execute_timing_out(repl, "block()\nguess = 'late'\nFINAL_VAR('guess')")
        assert timed_out.final_answer is None
        release.set()
        abandoned.join(5)

        result = _execute_off_main_thread(repl, "x = 1")
        assert result.final_answer is None

    def test_late_finish_does_not_hide_variables_of_running_execution(self):
        abandoned_threads = []
        repl, release = self._repl(
            finish_timed_out_turn=lambda: (release.set(), abandoned_threads[0].join(5)),
        )
        _, abandoned = _execute_timing_out(repl, "block()")
        abandoned_threads.append(abandoned)

        # The timed-out turn completes while this one is between assigning y and reading it back.
        result = _execute_off_main_thread(repl, "y = 5\nfinish_timed_out_turn()\nFINAL_VAR('y')")
        assert not abandoned.is_alive()
        assert result.final_answer == "5"
