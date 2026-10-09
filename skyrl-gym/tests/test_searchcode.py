import skyrl_gym
import pytest
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "action, expected_tool_name, expected_tool_input",
    [
        (
            "<tool><search>how to reverse a string in Python</search></tool>",
            "search",
            ["how to reverse a string in Python"],
        ),
        ("<tool><python>print('hello')</python></tool>", "python", ["print('hello')"]),
        ("<tool><search>binary search in Java</search></tool>", "search", ["binary search in Java"]),
    ],
)
def test_tool_parsing(action, expected_tool_name, expected_tool_input):
    # Test the parsing logic and reward logic with dummy input
    env = skyrl_gym.make(
        "searchcode",
        env_config=DictConfig({"env_class": "searchcode"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": "random"}, "max_turns": 2},
    )
    # Skip init() since it's not used in this test

    _, tool_name, tool_input = env._parse_action(action)

    assert tool_name == expected_tool_name
    assert tool_input == expected_tool_input


@pytest.mark.parametrize(
    "action, expected_output",
    [
        (
            "<tool><python>print('hello world')</python></tool>",
            "hello world",
        ),
        (
            "<tool><python>print(1 + 1)</python></tool>",
            "2",
        ),
        (
            "<tool><python>raise ValueError('fail')</python></tool>",
            'Error executing Python code: Traceback (most recent call last):\n  File "<string>", line 1, in <module>\nValueError: fail',
        ),
        # A failing exit status is an error even when nothing was written to stderr.
        (
            "<tool><python>import sys; print('partial'); sys.exit(3)</python></tool>",
            "Error executing Python code: Process exited with status 3 and no error output.",
        ),
        # A successful run that also writes to stderr (e.g. a warning) keeps its stdout.
        (
            "<tool><python>import sys; print(42); print('note', file=sys.stderr)</python></tool>",
            "42\nnote",
        ),
    ],
)
def test_python_code_execution(action, expected_output):
    env = skyrl_gym.make(
        "searchcode",
        env_config=DictConfig({"env_class": "searchcode"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": "random"}, "max_turns": 2},
    )
    env.init([])  # Initialize the environment

    output = env.step(action)
    observation_content = output["observations"][0]["content"]

    assert expected_output == observation_content
