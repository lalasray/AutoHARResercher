"""Explicit JSON LLM adapter; malformed responses never become approvals."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from .patches import OWNED_FILES, DATA_FIELDS, MODEL_FIELDS, AUGMENTATION_FIELDS
from .types import Judgment, Proposal

RULES = '''You are implementing AutoResearcHAR (ISWC 2026), Section 3.
Only source-training and fixed source-validation evidence may guide search.
Never request target signals, modify splits, labels, channel order, units,
the task-defined sampling rate, evaluator code, or the Macro-F1 objective.
Use the mutation limits supplied by the experiment context; defaults are
3 files, 3 functions, 8 configuration fields, and 120 changed code lines.
Return one JSON object only. No Markdown, shell commands, network or file I/O.
The harness alone promotes strictly improving source-validation Macro-F1.
'''

INTERFACES = '''Optional executable files use these interfaces:
research/components/data.py: preprocess(x) -> x, numpy [batch,time,channels],
shape preserving; optionally define fit_preprocessor(x_train, params) -> state
and transform(x, state) -> x. These two hooks must be supplied together.
Order: optional stateless preprocess, then fitted transform, then the built-in
normalizer. State is fitted ONLY on retained real source-training windows,
before synthetic samples; the same state transforms validation and synthetic
signals. Final refits learn fresh state on the non-test training partition.
State may contain finite NumPy arrays, numeric scalars, strings, lists, tuples
and string-keyed dictionaries, not estimators/callables. Transform receives a
copy and must preserve shape/channels and finite float values. For robust scaling,
fit training median and IQR (with a positive scale floor), transform using them,
and set normalization=none to avoid unintentionally standardizing twice.
research/components/model.py: build_model(input_dim, num_classes, params) ->
torch.nn.Module accepting [batch,time,channels] and returning [batch,classes].
Model code may call the injected baseline_model(input_dim, num_classes, params)
to construct the registered architecture and modify or wrap it. This is the
registered factory, not recursive execution of candidate code. It preserves the
author architecture unless your proposal explicitly changes it.
research/components/augmentation.py: augment(x, rng) -> x, training only; x is a device-resident torch.Tensor and rng a torch.Generator. Preserve device.
The augmentation hook runs only when augmentation.kind=custom; it replaces the
built-in conventional policy rather than automatically composing with it.
When switching to custom, set changes={"kind":"custom","params":{},
"transforms":[],"per_class":{}} to clear inherited built-in settings. The harness
applies the global probability mask after the custom hook; do not apply it twice.
Label mixing remains a separate harness step.
Data preprocessing runs before harness-fitted normalization. balanced_sampling and
class_ratios affect training sample selection only. Conventional augmentation is
training only (apart from runtime preflight checks on disposable probe tensors).
Each component can read its custom_params dictionary through component_params.
Model source can additionally define build_optimizer(model, params) returning a
PyTorch optimizer and build_loss(params, class_weights) returning a scalar loss
callable. Loss must accept logits and either integer or mixed soft-label targets
when mixing is enabled. These hooks alter training, not the evaluation metric.
Augmentation source may instead define augment_labeled(x, labels, rng), where
labels are the original training-class indices and activity_labels names them.
Both augmentation interfaces return only the transformed signals, never labels.
Allowed imports: numpy, torch, scipy, math, random. No I/O or introspection.
A component can be supplied as full replacement source in files; it executes
from the candidate snapshot, never by overwriting protected repository files.
'''


FIELDS = {'data': DATA_FIELDS, 'model': MODEL_FIELDS, 'augmentation': AUGMENTATION_FIELDS}


def patch_contract(role: str) -> str:
    return (
        f'Authoritative {role} patch contract: changes is a partial, role-local object. '
        'Use bare field names, without a role wrapper or dotted prefix. '
        f'Editable fields: {", ".join(sorted(FIELDS[role]))}. '
        'For example a model response is {"changes":{"learning_rate":0.0005},'
        '"files":{},"rationale":"Test a smaller learning rate",'
        '"dependencies":[],"validation":[]}. Other specialists use their own fields. '
        'Omitted fields retain incumbent values. Nested policy objects remain nested. '
        'Make an actual change relative to the incumbent; unchanged values and empty '
        'files are a duplicate. A rejected, untrained hypothesis is not empirical '
        'evidence against that hypothesis. Schema errors concern representation, '
        'not whether a field in this authoritative list exists.\n'
    )


def normalize_changes(role: str, changes: dict) -> dict:
    """Accept unambiguous role-local aliases without weakening field ownership."""
    if not isinstance(changes, dict):
        raise ValueError('changes must be a JSON object')
    if set(changes) == {role} and isinstance(changes[role], dict):
        changes = changes[role]
    result = {}
    for name, value in changes.items():
        field = name.removeprefix(role + '.')
        if field not in FIELDS[role]:
            raise ValueError(f'Unsupported/protected {role} configuration field: {name}')
        if field in result:
            raise ValueError(f'Ambiguous duplicate field: {field}')
        result[field] = value
    return result


class LLMAgents:
    def __init__(self, backend, code_edits=False):
        self.backend = backend
        self.code_edits = code_edits

    def generalist(self, context: dict) -> list[Proposal]:
        result = self.backend('generalist', 'You are one general-purpose HAR research agent. '
            'Propose one bounded pipeline change across any components. There are no '
            'specialists, Orchestrator, or independent Judge. Return changes containing '
            'partial data/model/augmentation dictionaries, files (path: source), rationale. '
            'Combined edits must meet the same global mutation bounds.\n' + INTERFACES +
            '\n' + json.dumps(context))
        files = result.get('files', {})
        if set(files) - set(OWNED_FILES.values()):
            raise ValueError('Generalist attempted a protected edit')
        changes = result.get('changes', {})
        if set(changes) - set(OWNED_FILES):
            raise ValueError('Unknown generalist component')
        return [Proposal(role, changes.get(role, {}),
                         {path: files[path]} if path in files else {}, result['rationale'])
                for role, path in OWNED_FILES.items() if changes.get(role) or path in files]

    def review_repeat(self, proposal, context):
        from dataclasses import asdict
        failed = [row for row in context['exploration']['failed_ideas'] if row['role'] == proposal.role]
        if not failed:
            return {'repeat': False, 'matched_round': None, 'reason': 'No failed idea for this role', 'material_difference': 'Untested role'}
        prompt = (
            'Check whether this proposal repeats an empirically unsuccessful mechanism. '
            'Return JSON with repeat (boolean), matched_round (integer or null), reason (string), '
            'material_difference (string). Compare actual configuration changes and executable '
            'semantics, not wording, variable names, formatting or claimed novelty. '
            'Set repeat=true when the same failed intervention is proposed in materially the '
            'same baseline context, including equivalent rewrites or cosmetic changes. '
            'Set repeat=false for a genuinely untested mechanism or a concrete materially '
            'different parameter, interaction, or baseline context; explain exactly what differs. '
            'A new rationale alone is not a difference. Do not ban an entire family because '
            'one setting failed. Do not predict improvement or reject due to uncertain benefit. '
            'For repeats matched_round must identify a supplied unsuccessful trial.\n'
            + json.dumps({'proposal': asdict(proposal), 'baseline': context['baseline'],
                          'failed_trials': failed}))
        result = self.backend('repeat_review', prompt)
        if type(result.get('repeat')) is not bool or not isinstance(result.get('reason'), str) or not result['reason'].strip():
            raise ValueError('Invalid failed-idea review')
        if result['repeat'] and result.get('matched_round') not in {row['round'] for row in failed}:
            raise ValueError('Repeated-idea review did not cite a recorded failed trial')
        if not result['repeat'] and not str(result.get('material_difference', '')).strip():
            raise ValueError('Failed-idea review requires a concrete material difference')
        return result

    def plan(self, context: dict) -> dict:
        if context.get('exploration'):
            prompt = (
                'Plan directed exploration. Return priority (permutation of data/model/augmentation), '
                'search_mode, risk, composition_bias, rationale, and hypotheses mapping each role to '
                '{family, hypothesis, evidence, expected_effect, falsification}. Choose family from '
                'exploration.allowed_focus for that role. Cite concrete source-validation outcomes '
                'or identify missing evidence. Explain a concise mechanism, what observation would '
                'refute it, and why this is the next useful experiment. Do not assume changes improve '
                'scores. Treat failed_ideas as persistent negative evidence: do not repackage a failed mechanism under new wording or variable names. A retry needs a concrete material parameter, interaction, or baseline-context change; cite the earlier result and explain why it changes the prediction. Prefer untested mechanisms over cosmetic retries. Avoid repeating configurations or code edits in experiment_ledger. Both '
                'configuration-only and executable changes are valid. Architecture defaults are '
                'starting points, not mandatory to preserve. Choose coherent staged tests, not '
                'arbitrary simultaneous changes. Do not restrict all exploration to tiny steps.\n'
                + json.dumps(context))
            from .exploration import validate_plan
            for attempt in range(2):
                result = self.backend('orchestrator', prompt)
                try:
                    validate_plan(result, context['exploration'])
                    return result
                except ValueError as exc:
                    if attempt:
                        raise
                    prompt += '\nCorrect this plan schema error: ' + str(exc) + '\nPrevious response: ' + json.dumps(result)
        return self.backend('orchestrator', 'Plan this round. Return search_mode, priority '
            '(permutation of data/model/augmentation), risk, composition_bias, and rationale.\n'
            + json.dumps(context))

    def propose(self, role: str, context: dict, plan: dict) -> Proposal:
        prompt = (f'You are the {role} specialist. Own only the {role} configuration section '
            f'and {OWNED_FILES[role]}.\n' + patch_contract(role) + INTERFACES + '\n'
            'Return changes (partial configuration object), files (path: full Python source), '
            'rationale, dependencies (other specialist roles required together), validation '
            '(list of assertions the change requires). Empty files is valid. '
            'Use bounded executable changes when existing configuration is insufficient.\n'
            'Data: eligible source subset, 0.5–10s windows, overlap 0–0.95, normalization '
            '(standard/minmax/none), balanced_sampling, class_ratios (label: retention in (0,1]). Model: architecture, dimensions, optimizer '
            '(adam/adamw/nadam/sgd/rmsprop), loss (cross_entropy/weighted_cross_entropy), learning_rate, '
            'weight_decay, label_smoothing, dropout, batch_size, max_epochs, patience. '
            'Architecture choices: tinyhar, tinierhar, deepconvlstm, bilstmattention, spectra, '
            'hart, mobilehart, deepconvlstmattention; cnn/gru/cnn_gru/bilstm/tiny_transformer are generic experiments. '
            'Changing architecture resets its model configuration to that architecture preset before applying explicit changes. '
            'All models use PyTorch. BiLSTM-Attention fixes hidden_dim=64 and '
            'requires complete sensor triplets; HART requires six channels; MobileHART requires six channels and 128 samples. '
            'SPECTRA is a paper reconstruction with n_fft and stft_hop settings. '
            'Augmentation: kind (none/custom/compose or augmentation_catalog method), params, probability, transforms (1–4 signal steps), mixing (mixup/temporal_cutmix with params and probability), strength shorthand, synthetic '
            '(requests with generator text/video, label, count, ratio in [0,1], and prompt '
            'or videos/selection). Synthetic requests require configured MCP generators. '
            'per_class maps fixed activity names to complete conventional policies (kind required): '
            'choose methods, params/strength, probabilities, compositions and mixing independently per class. '
            'Each entry replaces the global conventional policy for that class; unlisted classes use the global policy. '
            'kind=none disables transforms and mixing for that class. No nested per_class, custom, or synthetic entries. '
            'Mixup/CutMix policy follows the recipient original class; donors may come from any training class and targets are mixed.\n'
            + json.dumps({'context': context, 'plan': plan}))
        if context.get('exploration'):
            prompt += ('\nBROAD DIRECTED SEARCH: Follow your assigned hypothesis and family. '
                'Configuration-only, code-only, and combined changes are all allowed. Use '
                'exploration.mutation_limits rather than paper-sized edit limits. Activations, '
                'architecture, widths/depths, pooling, attention, losses, optimizers, training '
                'duration, windows, overlap-derived stride, normalization, sampling, filtering, '
                'augmentation families, per-class policies and mixing are editable. Implement '
                'unsupported settings in executable hooks and custom_params; do not invent '
                'top-level configuration fields. Default architecture constraints apply to '
                'the registered factory, but you may build or modify a compatible model in '
                'code. Files entries may be null to delete an incumbent custom override. '
                'Architecture switching resets model defaults and its prior custom override. '
                'Avoid redundant reimplementation of a built-in method just to create code. '
                'Explain your mechanism and expected failure mode, using the complete experiment '
                'ledger. Preserve unaffected mechanisms to make the result interpretable.')
        if self.code_edits:
            prompt += ('\nCODE-EDIT SEARCH: Return a substantive Python edit in your owned files entry. '
                'Configuration-only proposals are not eligible in this mode. Use configuration '
                'changes only to support executable changes. Start from current candidate code '
                'when present; model edits can use baseline_model to preserve the author model. '
                'Inspect model_source_context for the registered architecture. Do not replace '
                'the model with a simplified stand-in just to satisfy the file requirement. '
                'Data: implement preprocess. Model: implement build_model. Augmentation: '
                'implement augment and set changes.kind=custom; explicitly preserve any '
                'incumbent transform you intend to retain because the hook replaces the '
                'built-in augmentation. No placeholder, comments-only, or identity edits. '
                'Keep the same paper mutation limits. Full file source belongs in files, '
                'not in changes. Dependencies and validation describe executable assumptions.')
        result = self.backend(role, prompt)
        if self.code_edits:
            path = OWNED_FILES[role]
            source = result.get('files', {}).get(path)
            incumbent = context.get('baseline', {}).get('code', {}).get(path, '')
            import ast
            if not source or ast.dump(ast.parse(source)) == ast.dump(ast.parse(incumbent)):
                raise ValueError('Code-edit search requires a changed executable owned component')
            entry = {'data': 'preprocess', 'model': 'build_model', 'augmentation': 'augment'}[role]
            names = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
            if entry not in names and not (role == 'data' and {'fit_preprocessor', 'transform'} <= names):
                raise ValueError(f'Code-edit search requires the {entry} entry point')
        return Proposal(role, normalize_changes(role, result['changes']), result.get('files', {}), result['rationale'],
                        tuple(result.get('dependencies', [])), tuple(result.get('validation', [])))

    def judge(self, proposal: Proposal, context: dict, plan: dict, size: dict) -> Judgment:
        prompt = ('''Independently judge this proposal before training. Return integer scores
1–5 for novelty, plausibility, compatibility, plus rationale as a string.
Novelty: 1 duplicate, 2 minor new variation, 3 distinct change, 4 substantial
new mechanism, 5 substantially new justified direction.
Plausibility: 1 invalid or no coherent mechanism, 2 exploratory but coherent hypothesis, 3 reasonable HAR basis,
4 strong mechanism/evidence, 5 strongly justified for this source task.
Compatibility: 1 conflicting, 2 major interface problems, 3 unresolved
assumptions, 4 compatible with explicit dependencies, 5 directly composable.
Assess overlapping edits, dependencies, preprocessing/model assumptions,
tensor shapes, fixed labels, duplicated functionality, and mutation size.
Thresholds: novelty >=2, plausibility >=2, compatibility >=4. Do not predict
or fabricate measured scores. Use this fixed exploratory rubric throughout the run.
Missing dataset-specific evidence, uncertain benefit, or possible performance
regression is not alone a reason to reject a coherent, bounded experiment.
For example training-only sample retention can test sensitivity to redundancy
and coverage; score its coherent but unverified rationale at least 2.
Reserve plausibility 1 for a concrete contradiction or absent coherent rationale.
Keep compatibility strict for leakage, invalid shapes, unsupported operations,
and unmet dependencies; the harness determines actual improvement by validation.
''' + patch_contract(proposal.role) + INTERFACES + '''
The harness has already passed deterministic configuration validation, role/file
ownership, code restrictions, mutation bounds, and the nonduplicate-baseline check
for this proposal. Role-local fields in the contract are supported and editable;
do not reject them for lacking a prefix or for earlier malformed representations.
The documented hooks are provided by the harness. Runtime preflight follows this
review; static validation does not prove runtime correctness or empirical benefit.
Still assess concrete scientific and interface conflicts independently. A minor
untried hyperparameter variation can earn novelty 2. Missing measured improvement
is not an incompatibility: candidate training measures that after this review.
''' + json.dumps({'proposal': asdict(proposal), 'size': size, 'context': context, 'plan': plan}))
        result = self.backend('judge', prompt)
        return Judgment(result['novelty'], result['plausibility'], result['compatibility'], result.get('rationale', ''))

    def compose(self, proposals: dict, context: dict, plan: dict) -> list[list[str]]:
        result = self.backend('orchestrator', 'Choose compatible subsets of the eligible '
            'specialist roles. Singletons are automatically evaluated. Return {"subsets": '
            '[["data", "model"], ...]}. Reject conflicting interfaces, dependencies, '
            'shapes, label assumptions, or duplicate functionality. Choose before empirical '
            'evaluation; do not assume that individually improving edits compose well.\n'
            + json.dumps({'proposals': {r: asdict(p) for r, p in proposals.items()},
                          'context': context, 'plan': plan}))
        return result['subsets']


class OpenCodeBackend:
    """OpenCode JSON transport; tools are owned by the harness, not the LLM."""
    def __init__(self, model: str, binary: str = 'opencode', timeout: float = 180):
        self.model, self.binary, self.timeout = model, binary, timeout

    def __call__(self, role: str, prompt: str) -> dict:
        import subprocess
        import tempfile
        config = {'agent': {'autohar': {
            'description': 'Bounded AutoResearcHAR JSON proposals', 'mode': 'primary',
            'model': self.model, 'temperature': .2, 'top_p': .9, 'maxTokens': 4096,
            'steps': 1, 'permission': {'*': 'deny'}, 'prompt': RULES,
        }}, 'permission': {'*': 'deny'}, 'share': 'disabled'}
        with tempfile.TemporaryDirectory(prefix='autohar-llm-') as directory:
            env = dict(os.environ, OPENCODE_CONFIG_CONTENT=json.dumps(config))
            result = subprocess.run([self.binary, 'run', '--pure', '--format', 'json',
                '--dir', directory, '--model', self.model, '--agent', 'autohar', prompt],
                capture_output=True, text=True, timeout=self.timeout, env=env, check=True)
        texts = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('type') == 'error':
                raise ValueError('OpenCode returned an error event')
            if event.get('type') == 'text':
                texts.append(event['part']['text'])
        payload = json.loads('\n'.join(texts))
        if not isinstance(payload, dict):
            raise ValueError('OpenCode must return one JSON object')
        return payload


class CodexBackend:
    """Remote-model transport through authenticated, noninteractive Codex CLI."""
    def __init__(self, model: str, binary: str = 'codex', timeout: float = 180):
        self.model, self.binary, self.timeout = model, binary, timeout

    def __call__(self, role: str, prompt: str) -> dict:
        import subprocess
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(prefix='autohar-codex-') as directory:
            output = Path(directory) / 'response.json'
            command = [self.binary, 'exec', '--model', self.model,
                       '--ignore-user-config', '--ignore-rules', '--ephemeral',
                       '--skip-git-repo-check', '--sandbox', 'read-only',
                       '--cd', directory, '--output-last-message', str(output),
                       '-c', 'approval_policy="never"', '-c', 'web_search="disabled"',
                       '-c', 'model_provider="openai"']
            for feature in ('shell_tool', 'unified_exec', 'apps', 'plugins', 'hooks',
                            'multi_agent', 'multi_agent_v2', 'browser_use', 'computer_use',
                            'image_generation', 'view_image', 'code_mode', 'code_mode_host',
                            'skill_search', 'memories'):
                command.extend(['--disable', feature])
            command.append('-')
            subprocess.run(command, input=RULES + '\nRole: ' + role + '\n' + prompt,
                           capture_output=True, text=True, timeout=self.timeout, check=True)
            payload = json.loads(output.read_text())
        if not isinstance(payload, dict):
            raise ValueError('Codex must return one JSON object')
        return payload
