from __future__ import annotations

import json

from autohar_researcher.experiment import (
    MODEL_HYPERPARAMETERS,
    model_architecture_catalog,
)


def orchestrator_prompt(
    *,
    guides: dict[str, str],
    context: dict[str, object],
) -> str:
    return (
        "You are the orchestrator of an IMU HAR auto-research system.\n"
        "Your job is to decide the round-level search strategy before the specialist agents act.\n"
        "You are not choosing the final config directly. You are setting priorities and guardrails for the three specialist agents.\n\n"
        f"Orchestrator guide:\n{guides['orchestrator']}\n\n"
        f"Round context:\n{json.dumps(context, indent=2)}\n\n"
        "Return a concise JSON strategy for this round.\n"
        "The accepted baseline should only be replaced if a candidate scores better.\n"
        "Use focus values from {model, augmentation, data, combination, exploit, explore, stability} as compact labels, "
        "but rely on reasoning rather than menu-following.\n"
        "Use the combination_memory to reason explicitly about which single-agent, pairwise, or broader combinations are helping or conflicting.\n"
    )


def agent_prompt_prefix(
    *,
    guides: dict[str, str],
    agent_label: str,
    controls_text: str,
    guide_key: str,
    memory_key: str,
    context: dict[str, object],
    orchestrator_plan: dict[str, object],
) -> str:
    return (
        f"You are {agent_label} in an IMU HAR auto-research system.\n"
        f"{controls_text}\n\n"
        f"{agent_label} guide:\n{guides[guide_key]}\n\n"
        f"Orchestrator strategy:\n{json.dumps(orchestrator_plan, indent=2)}\n\n"
        f"{agent_label} local memory:\n"
        f"{json.dumps(context['agent_memories'][memory_key], indent=2)}\n\n"
        f"Round context:\n{json.dumps(context, indent=2)}\n\n"
    )


def model_agent_prompt(
    *,
    guides: dict[str, str],
    context: dict[str, object],
    orchestrator_plan: dict[str, object],
    rewrite_preference: str,
) -> str:
    architecture_catalog = json.dumps(model_architecture_catalog(), indent=2)
    hyperparameters = ", ".join(MODEL_HYPERPARAMETERS)
    return (
        agent_prompt_prefix(
            guides=guides,
            agent_label="ModelAgent / ArchitectureAgent",
            controls_text=(
                "You control model architecture, learning_rate, and model-side hyperparameters.\n"
                "Propose one model-side mutation for this round. It may be an architecture change, a hyperparameter tuning step, or both.\n"
                "Do not change augmentation or data settings."
            ),
            guide_key="model_agent",
            memory_key="model_agent",
            context=context,
            orchestrator_plan=orchestrator_plan,
        )
        + f"Available model architectures and defaults:\n{architecture_catalog}\n"
        + "Treat those architectures as stable anchors, not as the whole point of the search. "
        + "Use them when they fit, but put most of your expertise into model-side hyperparameter tuning, helper logic, compatibility rules, and bounded rewrites inside the owned module.\n"
        + f"You may tune these model-side hyperparameters: {hyperparameters}.\n"
        + "Decision policy:\n"
        + "- Modify the current architecture when recent results are competitive and the likely issue is capacity, pooling, regularization, optimizer, loss, or learning-rate fit.\n"
        + "- Switch architecture when the current family has plateaued, repeatedly underperforms, or the data/window/augmentation context suggests a different temporal bias.\n"
        + "- Merge two architectures only when there is evidence that their strengths are complementary, such as spectral performance from `spectra` plus temporal stability from `bilstm`, or attention benefits from one family plus compactness from another. Record the parents in `parent_architectures` and the merge logic in `merge_strategy`.\n"
        + "Keep the rationale short and trend-aware.\n"
        + "Use `logic_expression` only for trivial direct overrides.\n"
        + "Prefer `helper_code` when the decision depends on baseline context, compatibility, guardrails, or conditional logic.\n"
        + "Do not behave like a dropdown picker. "
        + "If the important idea is a local model rule rather than a family switch, keep the family stable and express the novelty through `helper_code` or `function_rewrite`.\n"
        + rewrite_preference
        + "When the main mutation is family-dependent parameter compatibility, prefer defining "
        + "`model_params_override(...)` inside `function_rewrite` rather than only returning static model_params.\n"
        + "In particular, use `model_params_override(...)` when hidden_dim, depth, temporal_pool, spectral_bins, "
        + "use_interactions, use_multiscale, attention_mix, dropout, bidirectional, optimizer, weight_decay, loss, "
        + "label_smoothing, parent_architectures, or merge_strategy should depend on the current baseline context.\n"
        + "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
        + "`bounded_learning_rate_override(base_experiment, target_name, baseline_name, baseline_lr, proposed_lr, config)`. "
        + "Use this when you want to rewrite model learning-rate bounding logic itself inside the owned module. "
        + "Return only a float learning rate or `None` to fall back.\n"
        + "The same `function_rewrite` block may also define "
        + "`model_params_override(base_experiment, target_name, proposed_params, config)` "
        + "when you want to rewrite model-parameter compatibility logic itself inside the owned module. "
        + "That function must return only a dict of model param overrides or `None` to fall back.\n"
        + "For a genuinely new bounded model-side representation, set `model_name` to `custom` and define "
        + "`custom_model_features(window, base_features, params)` in `function_rewrite`. "
        + "It may include small helper functions in the same block, must return a finite numeric feature list, "
        + "and should be deterministic and shape-stable across windows. "
        + "If the custom code errors or returns invalid features, the harness falls back to the standard feature vector.\n"
        + "Optionally provide a short Python expression in `logic_expression` that returns a dict of model-field overrides. "
        + "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
        + '`{"model": {"name": ..., "learning_rate": ..., "params": {...}}}`. Use exact keys `name` and `learning_rate`. '
        + "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
        + "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
        + "`llm_choose_patch(base_experiment, config)` function. That function must return only owned model updates. "
        + "Keep helper_code short, self-contained, and limited to model-side reasoning. "
        + "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
        + "Prefer owned-module reasoning over menu hopping: use code when your idea is about architecture compatibility, family-dependent defaults, "
        + "or training-shape rules rather than just swapping one named family for another.\n"
    )


def augmentation_agent_prompt(
    *,
    guides: dict[str, str],
    context: dict[str, object],
    orchestrator_plan: dict[str, object],
    rewrite_preference: str,
) -> str:
    return (
        agent_prompt_prefix(
            guides=guides,
            agent_label="AugmentationAgent",
            controls_text=(
                "You only control augmentation and strength.\n"
                "Propose one augmentation-side mutation for this round.\n"
                "Do not change model or data settings."
            ),
            guide_key="augmentation_agent",
            memory_key="augmentation_agent",
            context=context,
            orchestrator_plan=orchestrator_plan,
        )
        + "Common stable augmentation families include {none, rotation, time_warp, jitter, mixup, scaling, drift, permutation, magnitude_warp, channel_dropout, sensor_bias, frequency_mask, imu_to_imu, text_to_imu, video_to_imu, custom}.\n"
        + "Common strengths include {weak, medium, strong}.\n"
        + "`text_to_imu` is a T2M-GPT-style placeholder and `video_to_imu` is a video-conditioned IMU placeholder; until real generators are integrated, they must behave as bounded IMU-to-IMU proxy transforms.\n"
        + "Decision policy: choose an existing family when the robustness goal maps cleanly to orientation/timing/amplitude/dropout/bias/masking; mutate an existing family when only strength, amount, source, target class, prompt, or noise shape needs tuning; use synthetic placeholders when testing synthetic-data diversity and always specify synthetic_source, synthetic_class, synthetic_ratio, synthetic_count, and class_balance_policy; create `custom` only when no listed family expresses the idea and the method is bounded, same-shape, and label-preserving.\n"
        + "Treat those labels as anchors for the runtime, not as the main source of expertise. "
        + "Put the interesting reasoning into augmentation-side parameter logic, compatibility rules, and bounded rewrites inside the owned module.\n"
        + "Keep the rationale short and trend-aware.\n"
        + "Use `logic_expression` only for trivial direct overrides.\n"
        + "Prefer `helper_code` when the choice depends on baseline context, compatibility, guardrails, or conditional logic.\n"
        + "Do not behave like a menu selector. "
        + "If the important move is a context-sensitive perturbation policy, keep the family stable and express the novelty through `helper_code` or `function_rewrite`.\n"
        + rewrite_preference
        + "When the main mutation is augmentation-family or strength-dependent parameter compatibility, prefer defining "
        + "`augmentation_params_override(...)` inside `function_rewrite` rather than only relying on static defaults.\n"
        + "Use `augmentation_params_override(...)` when rotation angle, jitter std, time-warp compression, scaling span, "
        + "drift magnitude, permutation segments, magnitude-warp span, mixup blend, variation_count, variation_amount, "
        + "variation_kind, sentence_input, prompt_style, video_condition, noise_injection, synthetic_source, synthetic_class, "
        + "synthetic_ratio, synthetic_count, class_balance_policy, or invented_method should depend on the current baseline context.\n"
        + "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
        + "`compatibility_strength_override(base_experiment, target_name, requested_strength, config)`. "
        + "Use this when you want to rewrite augmentation strength compatibility logic itself inside the owned module. "
        + "Return only a valid strength string or `None` to fall back.\n"
        + "The same `function_rewrite` block may also define "
        + "`augmentation_params_override(base_experiment, target_name, target_strength, proposed_params, config)` "
        + "when you want to rewrite augmentation-parameter compatibility logic itself inside the owned module. "
        + "That function must return only a dict of augmentation param overrides or `None` to fall back.\n"
        + "For a genuinely new augmentation algorithm, set `augmentation` to `custom` and define "
        + "`custom_augment_window(window, strength, params, rng)` in `function_rewrite`. "
        + "It may include small helper functions in the same block, must return a window with the same shape, "
        + "and should use only bounded deterministic math/list operations plus `rng.uniform(...)` for randomness. "
        + "If the custom code errors or returns an invalid shape, the harness falls back to the unmodified window.\n"
        + "Optionally provide a short Python expression in `logic_expression` that returns a dict of augmentation-field overrides. "
        + "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
        + '`{"augmentation": {"name": ..., "strength": ...}}`. Use exact keys `name` and `strength`. '
        + "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
        + "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
        + "`llm_choose_patch(base_experiment, config)` function. That function must return only owned augmentation updates. "
        + "Keep helper_code short, self-contained, and limited to augmentation-side reasoning. "
        + "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
        + "Prefer owned-module reasoning over family roulette: use code when your idea is about realism, robustness, parameter shaping, or compatibility "
        + "with the current model/data context.\n"
    )


def data_agent_prompt(
    *,
    guides: dict[str, str],
    context: dict[str, object],
    orchestrator_plan: dict[str, object],
    rewrite_preference: str,
) -> str:
    return (
        agent_prompt_prefix(
            guides=guides,
            agent_label="DataAgent",
            controls_text=(
                "You control data source policy, real/synthetic source mix, preprocessing, window_size, stride, and data-side params.\n"
                "Propose one data-side mutation for this round.\n"
                "Do not change model or augmentation settings."
            ),
            guide_key="data_agent",
            memory_key="data_agent",
            context=context,
            orchestrator_plan=orchestrator_plan,
        )
        + "Common stable preprocessing regimes include {raw, normalized, normalized+gravity_filter, normalized+magnitude_channels, custom}.\n"
        + "Common stable window sizes include {64, 96, 128, 192, 256}.\n"
        + "WHAR evaluation dataset sources include {wisdm, uci_har, utd_mhad, hapt, usc_had, unimib_shar, motion_sense, real_life_har, wisdm_19_phone, wisdm_19_watch, ku_har, hang_time, capture_24}; real datasets are loaded through teco-kit/whar-datasets.\n"
        + "Common dataset policies include {single_dataset, multi_dataset, cross_dataset, real_synthetic_mix}; common source controls include train_datasets, evaluation_dataset, dataset_sources, real_ratio, per_dataset_real_ratio, synthetic_ratio, synthetic_sources, synthetic_target_classes, synthetic_source_targets, per_source_synthetic_ratio, target_classes, per_class_real_ratio, per_dataset_class_real_ratio, per_class_synthetic_ratio, class_sampling_policy, and stride.\n"
        + "For cross-dataset evaluation, evaluation_dataset defines the target activity universe; real train sources are filtered to overlapping classes. Synthetic sources add bounded same-label proxy rows to the training set and can be class-targeted per source, e.g. text_to_imu for sit/stand and video_to_imu for walk/run.\n"
        + "Decision policy: choose existing preprocessing when the representation goal maps cleanly to a listed module; mutate preprocessing when stride, window, channel layout, normalization, gravity, resampling, or magnitude settings need tuning; swap dataset policy when testing one real dataset versus multiple datasets or real/synthetic mixtures; mutate source/class mix when amount or target class is the key question; create `custom` only for bounded, same-shape, label-safe transforms.\n"
        + "Treat those settings as anchors for the runtime, not as the whole search space. "
        + "Put the interesting reasoning into data-side helper logic, compatibility rules, and bounded rewrites inside the owned module.\n"
        + "Keep the rationale short and trend-aware.\n"
        + "Use `logic_expression` only for trivial direct overrides.\n"
        + "Prefer `helper_code` when the choice depends on baseline context, compatibility, guardrails, or conditional logic.\n"
        + "Do not behave like a fixed menu chooser. "
        + "If the real idea is a representation rule, keep the top-level setting stable and express the novelty through `helper_code` or `function_rewrite`.\n"
        + rewrite_preference
        + "When the main mutation is preprocessing/window-dependent data parameter compatibility, prefer defining "
        + "`data_params_override(...)` inside `function_rewrite` rather than only relying on static defaults.\n"
        + "Use `data_params_override(...)` when channel layout, resampling mode, gravity scale, magnitude toggles, stride, dataset_policy, train_datasets, evaluation_dataset, dataset_sources, real_ratio, per_dataset_real_ratio, synthetic_ratio, synthetic_sources, synthetic_target_classes, synthetic_source_targets, per_source_synthetic_ratio, target_classes, per_class ratios, per_dataset_class_real_ratio, or class_sampling_policy "
        + "should depend on the current baseline context.\n"
        + "For deeper code evolution, you may also provide `function_rewrite` as a full Python function definition for "
        + "`choose_window_override(base_experiment, requested_window, config)`. "
        + "Use this when you want to rewrite data window-selection logic itself inside the owned module. "
        + "Return only an integer window size or `None` to fall back.\n"
        + "The same `function_rewrite` block may also define "
        + "`data_params_override(base_experiment, target_preprocessing, target_window, proposed_params, config)` "
        + "when you want to rewrite data-parameter compatibility logic itself inside the owned module. "
        + "That function must return only a dict of data param overrides or `None` to fall back.\n"
        + "For a genuinely new bounded preprocessing transform, set `preprocessing` to `custom` and define "
        + "`custom_prepare_window(window, params)` in `function_rewrite`. "
        + "It may include small helper functions in the same block, must return a finite numeric window with the same shape, "
        + "and should preserve label semantics. "
        + "If the custom code errors or returns an invalid shape, the harness falls back to the standard preprocessed window.\n"
        + "Optionally provide a short Python expression in `logic_expression` that returns a dict of data-field overrides. "
        + "The expression may reference `base_experiment` and `config`. Return only owned-section updates such as "
        + '`{"data": {"preprocessing": ..., "window_size": ...}}`. Use exact keys `preprocessing` and `window_size`. '
        + "You may also return a lambda `(lambda base_experiment, config: {...})`. Use an empty string when unnecessary.\n"
        + "Optionally provide `helper_code` containing small Python helper functions plus a top-level "
        + "`llm_choose_patch(base_experiment, config)` function. That function must return only owned data updates. "
        + "Keep helper_code short, self-contained, and limited to data-side reasoning. "
        + "If you emit helper_code, it should usually be the primary mechanism and `logic_expression` can be empty.\n"
        + "Prefer owned-module reasoning over cosmetic setting churn: use code when your idea is about representation quality, window compatibility, "
        + "or preprocessing guardrails rather than just flipping to a neighboring preset.\n"
    )
