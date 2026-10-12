import re
from typing import Tuple, Optional
import time
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .utils import _ERROR_MSG_PREFIX

_MAX_CHAR_DISPLAY = 2048

# CODER1_EXEC = os.environ.get("CODER1_EXEC", "bwrap")
CODER1_EXEC = os.environ.get("CODER1_EXEC", "unsafe_local")

if CODER1_EXEC == "unsafe_local":
    from .unsafe_local_exec import code_exec_local

    code_exec = code_exec_local
elif CODER1_EXEC == "sandboxfusion":
    from .sandboxfusion_exec import code_exec_sandboxfusion

    code_exec = code_exec_sandboxfusion
else:
    raise ValueError(f"Unknown CODER1_EXEC: {CODER1_EXEC}")


CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)
ANSWER_PATTERN = re.compile(r"</think>(.*)", re.DOTALL)
# pytest's closing summary, e.g. "==== 3 passed, 1 warning in 0.12s ====". pytest prints it
# as the last line, and only once the session has finished, so a run that the solution ends
# early never shows it.
PYTEST_SUMMARY_LINE = re.compile(r"^=*\s*(?P<counts>\d+ [a-z ]+(?:, \d+ [a-z ]+)*) in [0-9.]+s\b")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
# Outcomes that are not a pass. The solution can produce "skipped" and "xfailed" itself by
# calling pytest.skip() / pytest.xfail() from the code under test, which leaves the exit
# status at 0 although the test did not pass.
PYTEST_NOT_PASSED_OUTCOMES = frozenset({"failed", "error", "errors", "skipped", "xfailed"})


def pytest_tests_passed(output: str) -> bool:
    """True if pytest's closing summary reports passed tests and none failed, errored, skipped or xfailed."""
    # Only the last line is the closing summary; earlier lines can carry text the solution
    # controls, such as a warning message in the warnings summary.
    lines = [line for line in ANSI_ESCAPE.sub("", output).splitlines() if line.strip()]
    match = PYTEST_SUMMARY_LINE.match(lines[-1].strip()) if lines else None
    if match is None:
        return False
    outcomes = set(re.findall(r"[a-z]+", match["counts"]))
    return "passed" in outcomes and not outcomes & PYTEST_NOT_PASSED_OUTCOMES


def append_completion_marker(test_code: str) -> Tuple[str, str]:
    """Append a statement that prints a fresh marker after ``test_code`` has run.

    A zero exit status alone does not show that the tests ran: the solution runs first
    and can end the process early with status 0 (``sys.exit(0)``, ``os._exit(0)``,
    ``exit()``). The marker is printed only if control reaches the end of the tests. It
    is written straight to file descriptor 1 so that a solution rebinding ``print`` or
    ``sys.stdout`` cannot hide it from a run that did complete.
    """
    marker = f"__coder1_tests_completed_{uuid.uuid4().hex}__"
    return f"{test_code}\n__import__('os').write(1, b'\\n{marker}\\n')\n", marker


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


# def validate_response_structure(processed_str: str) -> bool:
#     pattern = re.compile(r".*</think>.*", re.DOTALL)
#     return bool(pattern.search(processed_str.strip()))


def try_extract_solution(solution_str: str) -> str:
    match = re.search(ANSWER_PATTERN, solution_str)

    if match:
        return match.group(1).strip()

    return solution_str


def extract_code_from_string(solution_str):
    solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return "\n".join(code_blocks).strip()


def fuzzy_equal(actual: str, expected: str, tolerance: float = 1e-6, verbose=True) -> bool:
    """
    Compare two outputs line by line and element by element for approximate equality.
    Handles:
    1. Integer and floating-point number comparison with tolerance
    2. Case-insensitive comparison for yes/no

    Args:
        actual: The actual output from code execution
        expected: The expected output
        tolerance: Tolerance for floating point number comparison

    Returns:
        bool: True if outputs are approximately equal
    """
    # Save original values for debugging
    original_actual = actual
    original_expected = expected

    # Normalize line endings
    actual = actual.strip().replace("\r\n", "\n")
    expected = expected.strip().replace("\r\n", "\n")

    # If exact match after normalization, return early
    if actual == expected:
        return True

    # Split into lines
    actual_lines = actual.split("\n")
    expected_lines = expected.split("\n")

    # If different number of lines, they're definitely not equal
    if len(actual_lines) != len(expected_lines):
        return False

    # Track fuzzy matches for debugging
    fuzzy_match_reasons = []

    # Compare each line
    for i, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines)):
        # If lines match exactly, continue
        if actual_line == expected_line:
            continue

        # Split into tokens by whitespace
        actual_tokens = actual_line.split()
        expected_tokens = expected_line.split()

        # If different number of tokens, they're not equal
        if len(actual_tokens) != len(expected_tokens):
            return False

        # Compare each token
        for j, (actual_token, expected_token) in enumerate(zip(actual_tokens, expected_tokens)):
            # If tokens match exactly, continue
            if actual_token == expected_token:
                continue

            # For yes/no, use case-insensitive comparison
            if actual_token.lower() in ["yes", "no"] and expected_token.lower() in ["yes", "no"]:
                if actual_token.lower() == expected_token.lower():
                    fuzzy_match_reasons.append(
                        f"Line {i+1}, Token {j+1}: Case-insensitive yes/no match '{actual_token}' ≈ '{expected_token}'"
                    )
                    continue
                else:
                    return False

            # Try numeric comparison
            try:
                actual_num = float(actual_token)
                expected_num = float(expected_token)
                diff = abs(actual_num - expected_num)

                if diff <= tolerance:
                    fuzzy_match_reasons.append(
                        f"Line {i+1}, Token {j+1}: Numeric match '{actual_token}' ≈ '{expected_token}' (diff: {diff})"
                    )
                    continue
                else:
                    return False
            except ValueError:
                # Not numeric values
                return False

    # Output fuzzy match information if any occurred
    if fuzzy_match_reasons and verbose:
        print("😅 FUZZY MATCH - Outputs approximately equal:")
        print(f"  Expected: {repr(original_expected)}")
        print(f"  Actual:   {repr(original_actual)}")
        print("  Reasons for fuzzy matching:")
        for reason in fuzzy_match_reasons:
            print(f"    • {reason}")

    # If we made it here, all lines are approximately equal
    return True


def _compute_score(solution_str, ground_truth, extra_info, format_reward=0.0, answer_reward=1.0):
    reward_log = []

    # ground_truth is not code, but tests
    # pass_fmt = validate_response_structure(solution_str)
    solution_code = extract_code_from_string(solution_str)
    # print(f"Extracted code: {solution_code}")

    # NameError: name 'List' is not defined. Did you mean: 'list'?
    # NameError: name 'defaultdict' is not defined
    # reference solutions fail due to imports not being present

    if (
        # not pass_fmt or len(solution_code) == 0
        len(solution_code)
        == 0
    ):  # only print full output when there is an error
        reward_log.append("-" * 16 + "Bad format detected!" + "-" * 16)
        reward_log.append("-" * 16 + "Original Model Output" + "-" * 16)
        reward_log.append(solution_str)
        # return -answer_reward - format_reward, "\n".join(reward_log)
        return 0.0, "\n".join(reward_log)

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    ground_truth = json.loads(ground_truth)

    t_start = time.time()

    # log code
    if "functional" in ground_truth:
        if "prefix" in extra_info and extra_info["prefix"] != None:
            solution_code = extra_info["prefix"] + "\n" + solution_code
        reward_log.append(solution_code + "\n" + ground_truth["functional"])
    else:
        reward_log.append(solution_code)

    if "pytest" in ground_truth or "functional" in ground_truth or "solution_file" in ground_truth:
        if "functional" in ground_truth:
            if "prefix" in extra_info and extra_info["prefix"] != None:
                solution_code = extra_info["prefix"] + "\n" + solution_code
            test_code, marker = append_completion_marker(ground_truth["functional"])
            succ, output = code_exec(solution_code + "\n" + test_code)
            # The solution runs before the tests, so exiting early also yields status 0, as
            # does an exit handler that calls os._exit(0) after a failed assertion.
            succ = succ and marker in output
        elif "solution_file" in ground_truth:
            succ, output = code_exec(solution_code, solution=ground_truth["solution_file"])
        else:  # pytest
            succ, output = code_exec(solution_code, pytest=ground_truth["pytest"])
            # os._exit(0) or pytest.skip() while the tests import or call the solution also
            # yields status 0, as does an exit handler that calls os._exit(0) after failures.
            succ = succ and pytest_tests_passed(output)
        if not succ:
            # reward_log.append(
            #     "!" * 16
            #     + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
            #     + "!" * 16
            # )
            # reward_log.append(output[:_MAX_CHAR_DISPLAY])
            # reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            # reward_log.append(extra_info["original_prompt"].replace("\n\n", "\n"))
            return format_reward, "\n".join(reward_log)
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]

        # Add parallelism
        # ok i see why they had concurrency issues.
        # they call this one by one by one for each of the inputs
        # spawns a separate process for each input as opposed to just being a single file
        with ThreadPoolExecutor(max_workers=min(8, len(stdin_list))) as executor:
            futures = [
                executor.submit(remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list)
            ]
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                if not succ or not fuzzy_equal(output.strip(), stdout.strip()):
                    output = output[:_MAX_CHAR_DISPLAY]  # truncate output to print
                    # reward_log.append(
                    #     "!" * 16
                    #     + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                    #     + "!" * 16
                    # )
                    # reward_log.append(f"🔎Input: {repr(stdin)}")
                    # reward_log.append(f"✅Expected: {repr(stdout.strip())}")
                    # reward_log.append(
                    #     f"❌Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}"
                    # )
                    # reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                    # reward_log.append(extra_info["original_prompt"].replace("\n\n", "\n"))
                    return format_reward, "\n".join(reward_log)
    else:
        raise ValueError(
            "Current supports for ground-truth are ['pytest', 'functional', 'solution_file', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    reward_log.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    reward_log.append(output)
    return format_reward + answer_reward, "\n".join(reward_log)


def compute_score(solution_str, ground_truth, extra_info, format_reward=0.0, answer_reward=1.0):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    score, reward_log = _compute_score(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        format_reward=format_reward,
        answer_reward=answer_reward,
    )
    marker = "✅" if score == (format_reward + answer_reward) else "❌"
    reward_log = (
        marker * 16
        + "Reward Calculation"
        + marker * 16
        + "\n"
        + reward_log
        + "\n"
        + marker * 16
        + f"Final Rward = {score}"
        + marker * 16
    )
    return {"score": score, "acc": score}
