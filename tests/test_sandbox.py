"""Tests for sandbox module."""

from __future__ import annotations

import pytest

from autohar_researcher.sandbox import (
    ALLOWED_BUILTINS,
    sandboxed_namespace,
    safe_eval,
    SandboxResult,
)


class TestAllowedBuiltins:
    def test_has_basic_math(self):
        assert "abs" in ALLOWED_BUILTINS
        assert "all" in ALLOWED_BUILTINS
        assert "any" in ALLOWED_BUILTINS
        assert "dict" in ALLOWED_BUILTINS
        assert "float" in ALLOWED_BUILTINS
        assert "int" in ALLOWED_BUILTINS
        assert "len" in ALLOWED_BUILTINS
        assert "list" in ALLOWED_BUILTINS
        assert "max" in ALLOWED_BUILTINS
        assert "min" in ALLOWED_BUILTINS
        assert "round" in ALLOWED_BUILTINS
        assert "set" in ALLOWED_BUILTINS
        assert "sorted" in ALLOWED_BUILTINS
        assert "str" in ALLOWED_BUILTINS
        assert "sum" in ALLOWED_BUILTINS
        assert "tuple" in ALLOWED_BUILTINS

    def test_missing_dangerous(self):
        assert "exec" not in ALLOWED_BUILTINS
        assert "eval" not in ALLOWED_BUILTINS
        assert "open" not in ALLOWED_BUILTINS
        assert "compile" not in ALLOWED_BUILTINS
        assert "import" not in ALLOWED_BUILTINS
        assert "__import__" not in ALLOWED_BUILTINS
        assert "breakpoint" not in ALLOWED_BUILTINS
        assert "reload" not in ALLOWED_BUILTINS
        assert "memoryview" not in ALLOWED_BUILTINS


class TestSandboxedNamespace:
    def test_empty_code(self):
        result = sandboxed_namespace("")
        assert result is None

    def test_whitespace_only(self):
        result = sandboxed_namespace("   \n\t  ")
        assert result is None

    def test_simple_function(self):
        code = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
        namespace = sandboxed_namespace(code)
        assert namespace is not None
        assert "add" in namespace
        assert "multiply" in namespace
        assert namespace["add"](2, 3) == 5
        assert namespace["multiply"](4, 5) == 20

    def test_with_builtins(self):
        code = """
def sum_list(items):
    return sum(items)

def max_val(a, b):
    return max(a, b)
"""
        namespace = sandboxed_namespace(code)
        assert namespace is not None
        assert namespace["sum_list"]([1, 2, 3]) == 6
        assert namespace["max_val"](10, 20) == 20

    def test_invalid_syntax(self):
        code = """
def broken(
    return 1
"""
        result = sandboxed_namespace(code)
        assert result is None

    def test_uses_whitelist_only(self):
        code = """
import os
def get_env():
    return os.environ
"""
        result = sandboxed_namespace(code)
        assert result is None

    def test_dict_return(self):
        code = """
def get_config():
    return {"model": "deepconvlstm", "lr": 0.001}
"""
        namespace = sandboxed_namespace(code)
        assert namespace is not None
        result = namespace["get_config"]()
        assert result == {"model": "deepconvlstm", "lr": 0.001}


class TestSafeEval:
    def test_empty_expression(self):
        result = safe_eval("")
        assert result is None

    def test_whitespace_only(self):
        result = safe_eval("   ")
        assert result is None

    def test_simple_dict(self):
        result = safe_eval('{"model": "cnn"}')
        assert result == {"model": "cnn"}

    def test_dict_with_variable(self):
        result = safe_eval(
            '{"model": model_name}',
            globals_dict={"model_name": "gru"},
        )
        assert result == {"model": "gru"}

    def test_dict_with_multiple_vars(self):
        result = safe_eval(
            '{"model": model_name, "lr": learning_rate}',
            globals_dict={"model_name": "lstm", "learning_rate": 0.001},
        )
        assert result == {"model": "lstm", "learning_rate": 0.001}

    def test_invalid_expression(self):
        result = safe_eval("this is not valid python")
        assert result is None

    def test_uses_only_safe_builtins(self):
        result = safe_eval(
            '{"result": abs(-5), "max_val": max(1, 2, 3)}',
        )
        assert result == {"result": 5, "max_val": 3}

    def test_complex_expression(self):
        result = safe_eval(
            '{"sum": sum([1, 2, 3]), "len": len("test"), "rounded": round(3.7)}',
        )
        assert result == {"sum": 6, "len": 4, "rounded": 4}


class TestSandboxResult:
    def test_success_result(self):
        result = SandboxResult(
            success=True,
            value={"model": "test"},
            error=None,
            timed_out=False,
        )
        assert result.success is True
        assert result.value == {"model": "test"}
        assert result.error is None
        assert result.timed_out is False

    def test_failure_result(self):
        result = SandboxResult(
            success=False,
            value=None,
            error="Execution failed",
            timed_out=False,
        )
        assert result.success is False
        assert result.error == "Execution failed"

    def test_timeout_result(self):
        result = SandboxResult(
            success=False,
            value=None,
            error="Timeout",
            timed_out=True,
        )
        assert result.timed_out is True
