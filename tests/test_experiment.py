"""Tests for experiment module."""

from __future__ import annotations

import json
import importlib
import tempfile
from pathlib import Path

import pytest

from autohar_researcher.experiment import (
    MODEL_HYPERPARAMETERS,
    available_augmentations,
    available_model_architectures,
    base_experiment_config,
    default_model_params,
    default_augmentation_params,
    default_data_params,
    model_architecture_catalog,
    module_owned_fields,
    experiment_to_candidate_config,
    clone_experiment,
    write_module_python,
    load_module_python,
    apply_model_proposal,
    apply_augmentation_proposal,
    apply_data_proposal,
    MODULE_FILENAMES,
)
from autohar_researcher.types import ModelProposal, AugmentationProposal, DataProposal


class TestDefaultParams:
    def test_available_model_architectures_include_custom(self):
        architectures = available_model_architectures()
        assert architectures[0] == "deepconvlstm"
        assert "deepconvlstmattention" in architectures
        assert "bilstm" in architectures
        assert "bilstmattention" in architectures
        assert "custom" in architectures

    def test_model_architecture_catalog_has_defaults(self):
        catalog = model_architecture_catalog()
        names = [item["name"] for item in catalog]
        assert names == available_model_architectures()
        assert all("default_params" in item for item in catalog)
        assert "learning_rate" in MODEL_HYPERPARAMETERS

    def test_default_model_params_deepconvlstm(self):
        params = default_model_params("deepconvlstm")
        assert params["hidden_dim"] == 64
        assert params["depth"] == 2
        assert params["temporal_pool"] == "avg"
        assert params["spectral_bins"] == 6
        assert params["use_interactions"] is True
        assert params["use_multiscale"] is True

    def test_default_model_params_cnn(self):
        params = default_model_params("cnn")
        assert params["hidden_dim"] == 32
        assert params["depth"] == 1
        assert params["use_interactions"] is False

    def test_default_model_params_gru(self):
        params = default_model_params("gru")
        assert params["hidden_dim"] == 40
        assert params["temporal_pool"] == "last"

    def test_default_model_params_bilstm(self):
        params = default_model_params("bilstm")
        assert params["bidirectional"] is True
        assert params["optimizer"] == "adam"
        assert params["loss"] == "cross_entropy"
        assert "parent_architectures" in params
        assert "merge_strategy" in params

    def test_default_model_params_bilstmattention(self):
        params = default_model_params("bilstmattention")
        assert params["bidirectional"] is True
        assert params["temporal_pool"] == "attention"
        assert params["attention_mix"] == 0.7

    def test_default_model_params_tiny_transformer(self):
        params = default_model_params("tiny_transformer")
        assert params["temporal_pool"] == "concat"
        assert params["hidden_dim"] == 48

    def test_default_augmentation_params_rotation(self):
        params = default_augmentation_params("rotation", "medium")
        assert params["angle_deg"] == 10.0

    def test_default_augmentation_params_rotation_strong(self):
        params = default_augmentation_params("rotation", "strong")
        assert params["angle_deg"] == 15.0

    def test_default_augmentation_params_rotation_weak(self):
        params = default_augmentation_params("rotation", "weak")
        assert params["angle_deg"] == 5.0

    def test_default_augmentation_params_jitter(self):
        params = default_augmentation_params("jitter", "medium")
        assert "jitter_std" in params

    def test_available_augmentations_include_generative_placeholders(self):
        augmentations = available_augmentations()
        assert "imu_to_imu" in augmentations
        assert "text_to_imu" in augmentations
        assert "video_to_imu" in augmentations

    def test_default_augmentation_params_text_to_imu(self):
        params = default_augmentation_params("text_to_imu", "medium")
        assert params["generator"] == "t2mgpt_placeholder"
        assert "sentence_input" in params
        assert "variation_amount" in params
        assert params["synthetic_source"] == "text_to_imu"
        assert "synthetic_class" in params
        assert "synthetic_ratio" in params

    def test_augmentation_modules_exist(self):
        augmentation_dir = Path("research/modules/augmentations")
        for augmentation in available_augmentations():
            assert (augmentation_dir / f"{augmentation}.py").exists()

    def test_augmentation_files_are_callable_modules(self):
        for augmentation in available_augmentations():
            module = importlib.import_module(
                f"research.modules.augmentations.{augmentation}"
            )
            assert callable(getattr(module, "augment_window", None))
            assert getattr(module, "NAME") == augmentation

    def test_default_augmentation_params_none(self):
        params = default_augmentation_params("none", "medium")
        assert params == {}

    def test_default_data_params_raw(self):
        params = default_data_params("raw", 128)
        assert params["channel_layout"] == "total_acc+gyro"
        assert params["normalize"] is False
        assert params["resample_mode"] == "nearest"

    def test_default_data_params_normalized(self):
        params = default_data_params("normalized", 128)
        assert params["normalize"] is True

    def test_default_data_params_gravity_filter(self):
        params = default_data_params("normalized+gravity_filter", 128)
        assert params["channel_layout"] == "body_acc+gyro"
        assert params["normalize"] is True

    def test_default_data_params_large_window(self):
        params = default_data_params("raw", 256)
        assert params["resample_mode"] == "linear"

    def test_default_data_params_magnitude_channels(self):
        params = default_data_params("normalized+magnitude_channels", 128)
        assert params["append_magnitude"] is True


class TestBaseExperimentConfig:
    def test_returns_valid_config(self):
        config = base_experiment_config()
        assert "dataset" in config
        assert "model" in config
        assert "data" in config
        assert "augmentation" in config
        assert "training" in config

    def test_model_defaults(self):
        config = base_experiment_config()
        assert config["model"]["name"] == "deepconvlstm"
        assert config["model"]["learning_rate"] == 3e-4

    def test_data_defaults(self):
        config = base_experiment_config()
        assert config["data"]["preprocessing"] == "raw"
        assert config["data"]["window_size"] == 128

    def test_augmentation_defaults(self):
        config = base_experiment_config()
        assert config["augmentation"]["name"] == "none"

    def test_training_defaults(self):
        config = base_experiment_config()
        assert config["training"]["epochs"] == 30
        assert config["training"]["min_epochs"] == 4
        assert config["training"]["early_stopping_patience"] == 5


class TestModuleOwnedFields:
    def test_model_agent_fields(self):
        fields = module_owned_fields("model_agent")
        assert "model.name" in fields
        assert "model.learning_rate" in fields
        assert "model.params" in fields

    def test_augmentation_agent_fields(self):
        fields = module_owned_fields("augmentation_agent")
        assert "augmentation.name" in fields
        assert "augmentation.strength" in fields
        assert "augmentation.params" in fields

    def test_data_agent_fields(self):
        fields = module_owned_fields("data_agent")
        assert "data.preprocessing" in fields
        assert "data.window_size" in fields
        assert "data.params" in fields


class TestCloneExperiment:
    def test_deep_copy(self):
        original = base_experiment_config()
        cloned = clone_experiment(original)
        cloned["model"]["name"] = "cnn"
        assert original["model"]["name"] == "deepconvlstm"

    def test_nested_dicts_are_copies(self):
        original = base_experiment_config()
        cloned = clone_experiment(original)
        cloned["model"]["params"]["hidden_dim"] = 128
        assert original["model"]["params"]["hidden_dim"] == 64


class TestExperimentToCandidateConfig:
    def test_conversion(self):
        experiment = base_experiment_config()
        config = experiment_to_candidate_config(experiment, "test notes")
        assert config.model_name == "deepconvlstm"
        assert config.preprocessing == "raw"
        assert config.augmentation == "none"
        assert config.learning_rate == 3e-4
        assert config.window_size == 128
        assert config.notes == "test notes"

    def test_with_custom_values(self):
        experiment = base_experiment_config()
        experiment["model"]["name"] = "cnn_gru"
        experiment["model"]["learning_rate"] = 0.001
        experiment["data"]["preprocessing"] = "normalized"
        experiment["data"]["window_size"] = 256
        experiment["augmentation"]["name"] = "rotation"
        experiment["augmentation"]["strength"] = "strong"
        config = experiment_to_candidate_config(experiment, "")
        assert config.model_name == "cnn_gru"
        assert config.learning_rate == 0.001
        assert config.preprocessing == "normalized"
        assert config.window_size == 256
        assert config.augmentation == "rotation"
        assert config.augmentation_strength == "strong"


class TestApplyProposals:
    def test_apply_model_proposal(self):
        experiment = base_experiment_config()
        proposal = ModelProposal(
            model_name="cnn",
            learning_rate=0.0005,
            model_params={"hidden_dim": 32},
        )
        updated = apply_model_proposal(experiment, proposal)
        assert updated["model"]["name"] == "cnn"
        assert updated["model"]["learning_rate"] == 0.0005
        assert updated["model"]["params"]["hidden_dim"] == 32

    def test_apply_augmentation_proposal(self):
        experiment = base_experiment_config()
        proposal = AugmentationProposal(
            augmentation="rotation",
            strength="medium",
        )
        updated = apply_augmentation_proposal(experiment, proposal)
        assert updated["augmentation"]["name"] == "rotation"
        assert updated["augmentation"]["strength"] == "medium"
        assert "angle_deg" in updated["augmentation"]["params"]

    def test_apply_data_proposal(self):
        experiment = base_experiment_config()
        proposal = DataProposal(
            preprocessing="normalized",
            window_size=192,
        )
        updated = apply_data_proposal(experiment, proposal)
        assert updated["data"]["preprocessing"] == "normalized"
        assert updated["data"]["window_size"] == 192


class TestModuleFilenames:
    def test_filenames(self):
        assert MODULE_FILENAMES["model_agent"] == "model_module.py"
        assert MODULE_FILENAMES["augmentation_agent"] == "augmentation_module.py"
        assert MODULE_FILENAMES["data_agent"] == "data_module.py"

    def test_architecture_specs_exist(self):
        architecture_dir = Path("research/modules/architectures")
        for architecture in available_model_architectures():
            assert (architecture_dir / f"{architecture}.py").exists()

    def test_architecture_files_are_buildable_modules(self):
        for architecture in available_model_architectures():
            module = importlib.import_module(
                f"research.modules.architectures.{architecture}"
            )
            assert callable(getattr(module, "build_model", None))
            assert getattr(module, "ARCHITECTURE_NAME") == architecture

    def test_architecture_files_accept_raw_sequences(self):
        torch = pytest.importorskip("torch")
        for architecture in available_model_architectures():
            module = importlib.import_module(
                f"research.modules.architectures.{architecture}"
            )
            params = default_model_params(architecture)
            params["input_channels"] = 6
            model = module.build_model(input_dim=6, num_classes=6, params=params)
            output = model(torch.randn(2, 32, 6))
            assert tuple(output.shape) == (2, 6)


class TestWriteAndLoadModule:
    def test_write_and_load_model_module(self, tmp_path):
        payload = {
            "agent_name": "model_agent",
            "branch_name": "agent-model-main",
            "round_id": 1,
            "sync_from_branch": "main",
            "owned_fields": ["model.name", "model.learning_rate"],
            "config": {
                "name": "deepconvlstm",
                "learning_rate": 0.0003,
                "params": {"hidden_dim": 64},
                "notes": "test",
            },
            "logic_expression": "",
            "helper_code": "",
            "function_rewrite": "",
        }

        module_dir = tmp_path / "research" / "modules"
        module_path = module_dir / "model_module.py"

        write_module_python(module_path, payload)
        assert module_path.exists()

        loaded = load_module_python(module_path)
        assert loaded["agent_name"] == "model_agent"
        assert loaded["config"]["name"] == "deepconvlstm"
        namespace: dict[str, object] = {}
        exec(module_path.read_text(encoding="utf-8"), namespace)
        assert namespace["stable_model_families"]() == available_model_architectures()

    def test_write_and_load_augmentation_module(self, tmp_path):
        payload = {
            "agent_name": "augmentation_agent",
            "branch_name": "agent-augmentation-main",
            "round_id": 1,
            "sync_from_branch": "main",
            "owned_fields": ["augmentation.name", "augmentation.strength"],
            "config": {
                "name": "rotation",
                "strength": "medium",
                "params": {"angle_deg": 10.0},
                "notes": "test",
            },
            "logic_expression": "",
            "helper_code": "",
            "function_rewrite": "",
        }

        module_dir = tmp_path / "research" / "modules"
        module_path = module_dir / "augmentation_module.py"

        write_module_python(module_path, payload)
        assert module_path.exists()

        loaded = load_module_python(module_path)
        assert loaded["agent_name"] == "augmentation_agent"
        assert loaded["config"]["name"] == "rotation"

    def test_write_and_load_data_module(self, tmp_path):
        payload = {
            "agent_name": "data_agent",
            "branch_name": "agent-data-main",
            "round_id": 1,
            "sync_from_branch": "main",
            "owned_fields": ["data.preprocessing", "data.window_size"],
            "config": {
                "preprocessing": "normalized",
                "window_size": 128,
                "params": {"normalize": True},
                "notes": "test",
            },
            "logic_expression": "",
            "helper_code": "",
            "function_rewrite": "",
        }

        module_dir = tmp_path / "research" / "modules"
        module_path = module_dir / "data_module.py"

        write_module_python(module_path, payload)
        assert module_path.exists()

        loaded = load_module_python(module_path)
        assert loaded["agent_name"] == "data_agent"
        assert loaded["config"]["preprocessing"] == "normalized"
