from __future__ import annotations

import os

import pytest

from autohar_researcher.planner_runtime import (
    OpencodeModelCatalog,
    load_local_env,
    role_runtime_config,
    selected_backend,
)


def test_load_local_env_preserves_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".autohar.env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=file-value",
                "AUTOHAR_OPENAI_MODEL='gpt-test'",
                "# ignored comment",
                "MALFORMED",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "existing-value")
    monkeypatch.delenv("AUTOHAR_OPENAI_MODEL", raising=False)

    load_local_env(env_file)

    assert selected_backend() == "openai"
    assert os.environ["OPENAI_API_KEY"] == "existing-value"
    assert os.environ["AUTOHAR_OPENAI_MODEL"] == "gpt-test"


def test_selected_backend_rejects_unsupported_backend(monkeypatch):
    monkeypatch.setenv("AUTOHAR_LLM_BACKEND", "unsupported")

    with pytest.raises(RuntimeError, match="Supported backends: openai, opencode"):
        selected_backend()


def test_opencode_catalog_uses_configured_model_without_shelling_out(monkeypatch):
    monkeypatch.setenv("AUTOHAR_OPENCODE_MODEL", "custom/model")

    assert OpencodeModelCatalog().default_model("fallback/model") == "custom/model"


def test_role_runtime_config_supports_openai_and_opencode(monkeypatch):
    monkeypatch.setenv("AUTOHAR_LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTOHAR_OPENAI_MODEL", "gpt-test")

    openai_runtime = role_runtime_config(
        "model_agent",
        planner_model="fallback-openai",
        base_url=None,
        default_api_key=None,
        default_api_mode="responses",
        default_opencode_model=lambda: "unused",
    )

    assert openai_runtime == {
        "backend": "openai",
        "model": "gpt-test",
        "base_url": "",
        "api_key": "test-key",
        "api_mode": "responses",
    }

    monkeypatch.setenv("AUTOHAR_LLM_BACKEND", "opencode")
    monkeypatch.delenv("AUTOHAR_OPENCODE_MODEL", raising=False)

    opencode_runtime = role_runtime_config(
        "model_agent",
        planner_model="fallback-openai",
        base_url=None,
        default_api_key=None,
        default_api_mode="responses",
        default_opencode_model=lambda: "qwen/test-model",
    )

    assert opencode_runtime["backend"] == "opencode"
    assert opencode_runtime["model"] == "qwen/test-model"
    assert opencode_runtime["api_key"] == ""
    assert opencode_runtime["api_mode"] == "opencode"
