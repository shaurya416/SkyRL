from skyrl_gym.tools.core import tool, ToolGroup
import subprocess


class PythonCodeExecutorToolGroup(ToolGroup):
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        super().__init__(name="PythonCodeExecutorToolGroup")

    @tool
    def python(self, code: str) -> str:
        try:
            result = subprocess.run(
                ["python", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout, text=True
            )
            # The exit status decides success: a failed run can leave stderr empty
            # (sys.exit(1), a killed process) and a successful one can write warnings to it.
            if result.returncode != 0:
                detail = result.stderr.strip() or f"Process exited with status {result.returncode} and no error output."
                return f"Error executing Python code: {detail}"
            return "\n".join(stream.strip() for stream in (result.stdout, result.stderr) if stream.strip())
        except subprocess.TimeoutExpired:
            return f"Error executing Python code: Execution timed out after {self.timeout} seconds."
