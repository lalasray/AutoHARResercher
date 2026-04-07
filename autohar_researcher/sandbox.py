from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

SANDBOX_TIMEOUT_SECONDS = int(os.getenv("AUTOHAR_SANDBOX_TIMEOUT", "5"))
SANDBOX_MEMORY_MB = int(os.getenv("AUTOHAR_SANDBOX_MEMORY_MB", "128"))


@dataclass(frozen=True)
class SandboxResult:
    success: bool
    value: Any
    error: str | None
    timed_out: bool


def _sandbox_worker(
    script_content: str, func_name: str, args_json: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "value": None,
        "error": None,
        "timed_out": False,
    }

    try:
        namespace: dict[str, Any] = {"__builtins__": ALLOWED_BUILTINS}
        exec(script_content, namespace, namespace)

        func = namespace.get(func_name)
        if not callable(func):
            result["error"] = f"Function {func_name!r} not found in sandboxed code"
            return result

        import json

        args = json.loads(args_json)
        value = func(*args)

        try:
            import json

            json.dumps(value)
            result["value"] = value
            result["success"] = True
        except (TypeError, ValueError) as e:
            result["error"] = f"Return value is not JSON-serializable: {e}"

    except TimeoutError:
        result["timed_out"] = True
        result["error"] = f"Execution timed out after {SANDBOX_TIMEOUT_SECONDS}s"
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def run_sandboxed(
    script_content: str,
    func_name: str,
    args: list[Any],
) -> SandboxResult:
    import json

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _sandbox_worker,
                script_content,
                func_name,
                json.dumps(args),
            )
            try:
                raw_result = future.result(timeout=SANDBOX_TIMEOUT_SECONDS)
            except TimeoutError:
                return SandboxResult(
                    success=False,
                    value=None,
                    error=f"Execution timed out after {SANDBOX_TIMEOUT_SECONDS}s",
                    timed_out=True,
                )
    except Exception as e:  # noqa: BLE001
        return SandboxResult(success=False, value=None, error=str(e), timed_out=False)

    return SandboxResult(
        success=raw_result.get("success", False),
        value=raw_result.get("value"),
        error=raw_result.get("error"),
        timed_out=raw_result.get("timed_out", False),
    )


def run_sandboxed_subprocess(
    script_content: str,
    func_name: str,
    args: list[Any],
) -> SandboxResult:
    import json

    worker_script = f"""
import sys
import signal
import json

ALLOWED_BUILTINS = {ALLOWED_BUILTINS!r}

def _timeout_handler(signum, frame):
    sys.exit(124)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm({SANDBOX_TIMEOUT_SECONDS})

{script_content}

sys.stdout.write(json.dumps({{"success": True, "result": {func_name}(*json.loads(sys.stdin.read()))}}))
"""

    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)},
        )

        stdout, stderr = proc.communicate(
            input=json.dumps(args),
            timeout=SANDBOX_TIMEOUT_SECONDS + 2,
        )

        if proc.returncode == 124:
            proc.kill()
            return SandboxResult(
                success=False,
                value=None,
                error=f"Execution timed out after {SANDBOX_TIMEOUT_SECONDS}s",
                timed_out=True,
            )

        if proc.returncode != 0:
            return SandboxResult(
                success=False,
                value=None,
                error=stderr.strip() or f"Process exited with code {proc.returncode}",
                timed_out=False,
            )

        result = json.loads(stdout)
        return SandboxResult(
            success=True,
            value=result.get("result"),
            error=None,
            timed_out=False,
        )

    except subprocess.TimeoutExpired:
        proc.kill()
        return SandboxResult(
            success=False,
            value=None,
            error="Subprocess timeout",
            timed_out=True,
        )
    except Exception as e:  # noqa: BLE001
        return SandboxResult(
            success=False,
            value=None,
            error=str(e),
            timed_out=False,
        )


def sandboxed_namespace(script_content: str) -> dict[str, Any] | None:
    if not script_content or not script_content.strip():
        return None

    namespace: dict[str, Any] = {"__builtins__": ALLOWED_BUILTINS}
    try:
        exec(script_content, namespace, namespace)
        return namespace
    except Exception:  # noqa: BLE001
        return None


def safe_eval(
    expression: str,
    globals_dict: dict[str, Any] | None = None,
    locals_dict: dict[str, Any] | None = None,
) -> Any:
    if not expression or not expression.strip():
        return None

    safe_globals = {**ALLOWED_BUILTINS}
    safe_locals: dict[str, Any] = {}

    if globals_dict:
        safe_locals.update(globals_dict)
    if locals_dict:
        safe_locals.update(locals_dict)

    try:
        result = eval(expression, {"__builtins__": safe_globals}, safe_locals)
        if callable(result):
            return result(safe_locals.get("base_experiment"), safe_locals.get("config"))
        return result
    except Exception:  # noqa: BLE001
        return None


SANDBOX_MODE = os.getenv("AUTOHAR_SANDBOX_MODE", "thread").lower()

if SANDBOX_MODE not in {"thread", "subprocess"}:
    SANDBOX_MODE = "thread"


def exec_sandboxed(
    script_content: str,
    func_name: str,
    args: list[Any],
) -> SandboxResult:
    if SANDBOX_MODE == "subprocess":
        return run_sandboxed_subprocess(script_content, func_name, args)
    return run_sandboxed(script_content, func_name, args)
