from __future__ import annotations

from autohar_researcher.planner_prompts import data_agent_prompt
from autohar_researcher.planner_schemas import (
    data_agent_plan_schema,
    model_agent_plan_schema,
    proposal_judgments_schema,
)


def test_data_agent_schema_is_strict_for_openai_responses():
    schema = data_agent_plan_schema()
    data_params = schema["properties"]["data_params"]

    assert schema["additionalProperties"] is False
    assert data_params["additionalProperties"] is False
    assert set(data_params["required"]) == set(data_params["properties"])


def test_model_schema_requires_all_model_params():
    schema = model_agent_plan_schema()
    model_params = schema["properties"]["model_params"]

    assert model_params["additionalProperties"] is False
    assert set(model_params["required"]) == set(model_params["properties"])


def test_judge_schema_has_three_specialists():
    schema = proposal_judgments_schema()

    assert schema["required"] == ["model_agent", "augmentation_agent", "data_agent"]


def test_data_prompt_mentions_whar_and_class_targeted_synthetic_sources():
    guides = {
        "data_agent": "Use good data judgment.",
    }
    context = {
        "round_id": 1,
        "agent_memories": {"data_agent": {}},
    }
    prompt = data_agent_prompt(
        guides=guides,
        context=context,
        orchestrator_plan={"summary": "test"},
        rewrite_preference="",
    )

    assert "teco-kit/whar-datasets" in prompt
    assert "evaluation_dataset defines the target activity universe" in prompt
    assert "synthetic_source_targets" in prompt
