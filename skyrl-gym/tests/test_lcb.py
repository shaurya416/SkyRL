import pytest
import skyrl_gym
import json
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "model_response, tests, expected_reward",
    [
        # Correct code: second largest index
        (
            """```python
def main():
    N = int(input())
    A = list(map(int, input().split()))
    B = sorted(A, reverse=True)
    second = B[1]
    print(A.index(second) + 1)

if __name__ == "__main__":
    main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                    {"input": "3\n3 2 1\n", "output": "2\n", "testtype": "stdin"},
                ]
            ),
            1.0,
        ),
        # Wrong logic: returns index of largest
        (
            """```python
def main():
    N = int(input())
    A = list(map(int, input().split()))
    print(A.index(max(A)) + 1)

if __name__ == "__main__":
    main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                ]
            ),
            0.0,
        ),
        # Missing main() call — runtime error
        (
            """```python
def main():
    A = list(map(int, input().split()))
    B = sorted(A, reverse=True)
    second = B[1]
    print(A.index(second) + 1)

# forgot to call main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                ]
            ),
            0.0,
        ),
    ],
)
def test_compute_score(model_response, tests, expected_reward):
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": tests}},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(model_response)
    assert step_output["reward"] == expected_reward


# Call-based (LeetCode-style) tests: each input line is one JSON-encoded argument.
_FUNCTIONAL_TESTS = json.dumps(
    [
        {"input": "[1, 2, 3]\n2", "output": "[1, 3]", "testtype": "functional", "metadata": {"func_name": "dropValue"}},
        {"input": "[5, 5, 7]\n5", "output": "[7]", "testtype": "functional", "metadata": {"func_name": "dropValue"}},
    ]
)


@pytest.mark.parametrize(
    "solution, expected_reward",
    [
        # Correct solution
        (
            """class Solution:
    def dropValue(self, nums, value):
        return [n for n in nums if n != value]""",
            1.0,
        ),
        # Correct values returned as a tuple are still accepted
        (
            """class Solution:
    def dropValue(self, nums, value):
        return tuple(n for n in nums if n != value)""",
            1.0,
        ),
        # Correct values returned as numpy scalars inside a list are still accepted
        (
            """class Solution:
    def dropValue(self, nums, value):
        arr = np.array(nums)
        return list(arr[arr != value])""",
            1.0,
        ),
        # Wrong solution
        (
            """class Solution:
    def dropValue(self, nums, value):
        return nums""",
            0.0,
        ),
        # Wrong solution returning an object that claims to equal anything
        (
            """class AlwaysEqual:
    def __eq__(self, other):
        return True

class Solution:
    def dropValue(self, nums, value):
        return AlwaysEqual()""",
            0.0,
        ),
        # The same object nested inside a plain list of the expected length
        (
            """class AlwaysEqual:
    def __eq__(self, other):
        return True

class Solution:
    def dropValue(self, nums, value):
        return [AlwaysEqual() for n in nums if n != value]""",
            0.0,
        ),
        # A list subclass that overrides __eq__
        (
            """class AnyList(list):
    def __eq__(self, other):
        return True

class Solution:
    def dropValue(self, nums, value):
        return AnyList()""",
            0.0,
        ),
        # A metaclass whose __eq__ makes the class compare equal to the builtin types
        (
            """class Meta(type):
    def __eq__(cls, other):
        return True
    __hash__ = type.__hash__

class AlwaysEqual(metaclass=Meta):
    def __eq__(self, other):
        return True

class Solution:
    def dropValue(self, nums, value):
        return AlwaysEqual()""",
            0.0,
        ),
    ],
)
def test_compute_score_call_based(solution, expected_reward):
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": _FUNCTIONAL_TESTS}},
    )
    step_output = env.step(f"```python\n{solution}\n```")
    assert step_output["reward"] == expected_reward
