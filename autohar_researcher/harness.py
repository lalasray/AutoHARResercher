from __future__ import annotations

from autohar_researcher.types import CandidateConfig, CandidateResult


class DummyHARHarness:
    """Deterministic stand-in for a real IMU HAR training/eval pipeline."""

    MODEL_SCORES = {
        "cnn": 0.61,
        "gru": 0.63,
        "cnn_gru": 0.67,
        "tiny_transformer": 0.65,
    }

    PREPROCESSING_SCORES = {
        "raw": 0.00,
        "normalized": 0.03,
        "normalized+gravity_filter": 0.05,
        "normalized+magnitude_channels": 0.06,
    }

    AUGMENTATION_SCORES = {
        "none": 0.00,
        "jitter": 0.01,
        "rotation": 0.02,
        "time_warp": 0.015,
        "mixup": 0.01,
    }

    WINDOW_SCORES = {
        64: 0.01,
        128: 0.025,
        256: 0.015,
    }

    LR_SCORES = {
        1e-4: 0.010,
        3e-4: 0.025,
        1e-3: 0.015,
    }

    def evaluate(
        self,
        round_id: int,
        agent_name: str,
        branch_name: str,
        worktree_path: str,
        config: CandidateConfig,
    ) -> CandidateResult:
        base = 0.05
        model = self.MODEL_SCORES[config.model_name]
        preprocessing = self.PREPROCESSING_SCORES[config.preprocessing]
        augmentation = self.AUGMENTATION_SCORES[config.augmentation]
        window = self.WINDOW_SCORES[config.window_size]
        lr = self.LR_SCORES[config.learning_rate]

        synergy = 0.0
        if config.model_name == "cnn_gru" and config.preprocessing == "normalized+gravity_filter":
            synergy += 0.03
        if config.model_name == "tiny_transformer" and config.augmentation == "rotation":
            synergy += 0.02
        if config.preprocessing == "normalized+magnitude_channels" and config.window_size == 128:
            synergy += 0.01

        val_score = round(min(base + model + preprocessing + augmentation + window + lr + synergy, 0.98), 4)
        summary = (
            f"{agent_name} proposed {config.model_name} with {config.preprocessing}, "
            f"{config.augmentation}, lr={config.learning_rate}, window={config.window_size}"
        )
        return CandidateResult(
            round_id=round_id,
            agent_name=agent_name,
            branch_name=branch_name,
            worktree_path=worktree_path,
            config=config,
            val_score=val_score,
            summary=summary,
            score_breakdown={
                "base": base,
                "model": model,
                "preprocessing": preprocessing,
                "augmentation": augmentation,
                "window": window,
                "lr": lr,
                "synergy": synergy,
            },
        )
