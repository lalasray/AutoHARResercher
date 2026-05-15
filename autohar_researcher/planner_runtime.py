from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

ROLE_KEYS = {
    "orchestrator": "ORCHESTRATOR",
    "model_agent": "MODEL_AGENT",
    "augmentation_agent": "AUGMENTATION_AGENT",
    "data_agent": "DATA_AGENT",
    "judge": "JUDGE",
}

SUPPORTED_BACKENDS = {"openai", "opencode"}
DEFAULT_OPENCODE_MODELS = [
    "qwen/qwen3-30b-a3b-2507",
    "qwen/qwen3-coder-30b",
    "openai/gpt-oss-20b",
]
REMOTE_OPENCODE_MODELS = [
    "opencode/gpt-5-nano",
    "opencode/big-pickle",
    "opencode/nemotron-3-super-free",
]


def load_local_env(path: Path | None = None) -> None:
    env_path = path or (Path.cwd() / ".autohar.env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def role_key(role: str) -> str:
    return ROLE_KEYS.get(role, role.upper())


def validate_backend(backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise RuntimeError(
            f"Unsupported LLM backend '{backend}'. Supported backends: openai, opencode."
        )
    return normalized


def selected_backend(default: str = "openai") -> str:
    return validate_backend(os.getenv("AUTOHAR_LLM_BACKEND", default))


def role_backend(role: str) -> str:
    key = role_key(role)
    return validate_backend(
        os.getenv(
            f"AUTOHAR_{key}_BACKEND",
            os.getenv("AUTOHAR_LLM_BACKEND", "openai"),
        )
    )


def role_api_mode(role: str, default_api_mode: str) -> str:
    key = role_key(role)
    return (
        os.getenv(
            f"AUTOHAR_OPENAI_{key}_API_MODE",
            os.getenv(f"AUTOHAR_{key}_API_MODE", default_api_mode),
        )
        .strip()
        .lower()
    )


class OpencodeModelCatalog:
    def __init__(self) -> None:
        self._cache: list[str] | None = None

    def default_model(self, fallback_model: str = "") -> str:
        configured = os.getenv("AUTOHAR_OPENCODE_MODEL", "").strip()
        if configured:
            return configured

        models = self.models()
        for preferred in DEFAULT_OPENCODE_MODELS:
            if preferred in models:
                return preferred
        for preferred in REMOTE_OPENCODE_MODELS:
            if preferred in models:
                return preferred
        if DEFAULT_OPENCODE_MODELS:
            return DEFAULT_OPENCODE_MODELS[0]
        return fallback_model

    def models(self) -> list[str]:
        if self._cache is not None:
            return self._cache
        binary = os.getenv(
            "AUTOHAR_OPENCODE_BIN", os.path.expanduser("~/.opencode/bin/opencode")
        )
        try:
            result = subprocess.run(
                [binary, "models"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            self._cache = []
            return self._cache

        if result.returncode != 0:
            self._cache = []
            return self._cache
        self._cache = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        return self._cache


def role_runtime_config(
    role: str,
    *,
    planner_model: str,
    base_url: str | None,
    default_api_key: str | None,
    default_api_mode: str,
    default_opencode_model: Callable[[], str],
) -> dict[str, str]:
    key = role_key(role)
    backend = role_backend(role)
    if backend == "opencode":
        model = os.getenv(
            f"AUTOHAR_OPENCODE_{key}_MODEL",
            os.getenv(
                f"AUTOHAR_{key}_MODEL",
                os.getenv("AUTOHAR_OPENCODE_MODEL", ""),
            ),
        )
        return {
            "backend": backend,
            "model": model or default_opencode_model(),
            "base_url": os.getenv(
                f"AUTOHAR_OPENCODE_{key}_BIN",
                os.getenv(
                    "AUTOHAR_OPENCODE_BIN",
                    os.path.expanduser("~/.opencode/bin/opencode"),
                ),
            ),
            "api_key": "",
            "api_mode": "opencode",
        }

    return {
        "backend": backend,
        "model": os.getenv(
            f"AUTOHAR_OPENAI_{key}_MODEL",
            os.getenv(
                f"AUTOHAR_{key}_MODEL",
                os.getenv("AUTOHAR_OPENAI_MODEL", planner_model),
            ),
        ),
        "base_url": os.getenv(
            f"AUTOHAR_OPENAI_{key}_BASE_URL",
            os.getenv(f"AUTOHAR_{key}_BASE_URL", base_url or ""),
        ),
        "api_key": os.getenv(
            f"AUTOHAR_OPENAI_{key}_API_KEY",
            os.getenv(
                f"AUTOHAR_{key}_API_KEY",
                default_api_key or os.getenv("OPENAI_API_KEY", ""),
            ),
        ),
        "api_mode": role_api_mode(role, default_api_mode),
    }
