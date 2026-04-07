"""Tests for config module."""

from __future__ import annotations

import os
import pytest

from autohar_researcher.config import (
    JudgeThresholds,
    load_judge_thresholds,
    TrainingConfig,
    load_training_config,
    SandboxConfig,
    load_sandbox_config,
    DEFAULT_JUDGE_THRESHOLDS,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_SANDBOX_CONFIG,
)


class TestJudgeThresholds:
    def test_default_values(self):
        assert DEFAULT_JUDGE_THRESHOLDS.prior_best_delta == 0.015
        assert DEFAULT_JUDGE_THRESHOLDS.recent_window_delta == 0.01
        assert DEFAULT_JUDGE_THRESHOLDS.relaxed_prior_best_delta == 0.03
        assert DEFAULT_JUDGE_THRESHOLDS.relaxed_recent_delta == 0.02

    def test_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_JUDGE_THRESHOLDS.prior_best_delta = 0.1

    def test_custom_values(self):
        thresholds = JudgeThresholds(
            prior_best_delta=0.02,
            recent_window_delta=0.015,
            relaxed_prior_best_delta=0.04,
            relaxed_recent_delta=0.03,
        )
        assert thresholds.prior_best_delta == 0.02

    def test_load_defaults_no_env(self):
        for key in [
            "AUTOHAR_JUDGE_PRIOR_BEST_DELTA",
            "AUTOHAR_JUDGE_RECENT_DELTA",
            "AUTOHAR_JUDGE_RELAXED_PRIOR_BEST_DELTA",
            "AUTOHAR_JUDGE_RELAXED_RECENT_DELTA",
        ]:
            if key in os.environ:
                del os.environ[key]

        thresholds = load_judge_thresholds()
        assert thresholds.prior_best_delta == 0.015
        assert thresholds.recent_window_delta == 0.01
        assert thresholds.relaxed_prior_best_delta == 0.03
        assert thresholds.relaxed_recent_delta == 0.02

    def test_load_from_env(self, monkeypatch):
        monkeypatch.setenv("AUTOHAR_JUDGE_PRIOR_BEST_DELTA", "0.025")
        monkeypatch.setenv("AUTOHAR_JUDGE_RECENT_DELTA", "0.02")
        monkeypatch.setenv("AUTOHAR_JUDGE_RELAXED_PRIOR_BEST_DELTA", "0.05")
        monkeypatch.setenv("AUTOHAR_JUDGE_RELAXED_RECENT_DELTA", "0.04")

        thresholds = load_judge_thresholds()
        assert thresholds.prior_best_delta == 0.025
        assert thresholds.recent_window_delta == 0.02
        assert thresholds.relaxed_prior_best_delta == 0.05
        assert thresholds.relaxed_recent_delta == 0.04


class TestTrainingConfig:
    def test_default_values(self):
        assert DEFAULT_TRAINING_CONFIG.base_lr_multiplier == 50.0
        assert DEFAULT_TRAINING_CONFIG.mlp_lr_multiplier == 50.0
        assert DEFAULT_TRAINING_CONFIG.grad_clip_mlp == 5.0
        assert DEFAULT_TRAINING_CONFIG.grad_clip_linear == 2.0
        assert DEFAULT_TRAINING_CONFIG.lr_decay_factor == 0.9
        assert DEFAULT_TRAINING_CONFIG.lr_decay_patience == 10

    def test_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_TRAINING_CONFIG.base_lr_multiplier = 100.0

    def test_custom_values(self):
        config = TrainingConfig(
            base_lr_multiplier=60.0,
            mlp_lr_multiplier=70.0,
            grad_clip_mlp=4.0,
            grad_clip_linear=1.5,
            lr_decay_factor=0.85,
            lr_decay_patience=15,
        )
        assert config.base_lr_multiplier == 60.0
        assert config.grad_clip_mlp == 4.0

    def test_load_from_env(self, monkeypatch):
        monkeypatch.setenv("AUTOHAR_BASE_LR_MULTIPLIER", "75.0")
        monkeypatch.setenv("AUTOHAR_MLP_LR_MULTIPLIER", "80.0")
        monkeypatch.setenv("AUTOHAR_GRAD_CLIP_MLP", "6.0")
        monkeypatch.setenv("AUTOHAR_GRAD_CLIP_LINEAR", "3.0")
        monkeypatch.setenv("AUTOHAR_LR_DECAY_FACTOR", "0.8")
        monkeypatch.setenv("AUTOHAR_LR_DECAY_PATIENCE", "20")

        config = load_training_config()
        assert config.base_lr_multiplier == 75.0
        assert config.mlp_lr_multiplier == 80.0
        assert config.grad_clip_mlp == 6.0
        assert config.grad_clip_linear == 3.0
        assert config.lr_decay_factor == 0.8
        assert config.lr_decay_patience == 20


class TestSandboxConfig:
    def test_default_values(self):
        assert DEFAULT_SANDBOX_CONFIG.timeout_seconds == 5
        assert DEFAULT_SANDBOX_CONFIG.memory_mb == 128
        assert DEFAULT_SANDBOX_CONFIG.mode == "thread"

    def test_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_SANDBOX_CONFIG.timeout_seconds = 10

    def test_custom_values(self):
        config = SandboxConfig(
            timeout_seconds=10,
            memory_mb=256,
            mode="subprocess",
        )
        assert config.timeout_seconds == 10
        assert config.memory_mb == 256
        assert config.mode == "subprocess"

    def test_load_from_env(self, monkeypatch):
        monkeypatch.setenv("AUTOHAR_SANDBOX_TIMEOUT", "15")
        monkeypatch.setenv("AUTOHAR_SANDBOX_MEMORY_MB", "512")
        monkeypatch.setenv("AUTOHAR_SANDBOX_MODE", "subprocess")

        config = load_sandbox_config()
        assert config.timeout_seconds == 15
        assert config.memory_mb == 512
        assert config.mode == "subprocess"
