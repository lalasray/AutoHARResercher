"""Explicit JSON LLM adapter; malformed responses never become approvals."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from .patches import OWNED_FILES
from .types import Judgment, Proposal

RULES = '''You are implementing AutoResearcHAR (ISWC 2026), Section 3.
Only source-training and fixed source-validation evidence may guide search.
Never request target signals, modify splits, labels, channel order, units,
50 Hz resampling, evaluator code, or the Macro-F1 objective. Changes are bounded
by 3 files, 3 functions, 8 configuration fields, and 120 changed code lines.
Return one JSON object only. No Markdown, shell commands, network or file I/O.
The harness alone promotes strictly improving source-validation Macro-F1.
'''

INTERFACES = '''Optional executable files use these interfaces:
research/components/data.py: preprocess(x) -> x, numpy [batch,time,channels],
shape preserving, stateless; fitted standardization is owned by the harness.
research/components/model.py: build_model(input_dim, num_classes, params) ->
torch.nn.Module accepting [batch,time,channels] and returning [batch,classes].
research/components/augmentation.py: augment(x, rng) -> x, training only; x is a device-resident torch.Tensor and rng a torch.Generator. Preserve device.
Allowed imports: numpy, torch, scipy, math, random. No I/O or introspection.
A component can be supplied as full replacement source in files; it executes
from the candidate snapshot, never by overwriting protected repository files.
'''


class LLMAgents:
    def __init__(self, backend):
        self.backend = backend

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

    def plan(self, context: dict) -> dict:
        return self.backend('orchestrator', 'Plan this round. Return search_mode, priority '
            '(permutation of data/model/augmentation), risk, composition_bias, and rationale.\n'
            + json.dumps(context))

    def propose(self, role: str, context: dict, plan: dict) -> Proposal:
        prompt = (f'You are the {role} specialist. Own only the {role} configuration section '
            f'and {OWNED_FILES[role]}.\n' + INTERFACES + '\n'
            'Return changes (partial configuration object), files (path: full Python source), '
            'rationale, dependencies (other specialist roles required together), validation '
            '(list of assertions the change requires). Empty files is valid. '
            'Use bounded executable changes when existing configuration is insufficient.\n'
            'Data: eligible source subset, 1–5s windows, overlap 0–0.75, normalization '
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
        result = self.backend(role, prompt)
        return Proposal(role, result['changes'], result.get('files', {}), result['rationale'],
                        tuple(result.get('dependencies', [])), tuple(result.get('validation', [])))

    def judge(self, proposal: Proposal, context: dict, plan: dict, size: dict) -> Judgment:
        prompt = ('''Independently judge this proposal before training. Return integer scores
1–5 for novelty, plausibility, compatibility, plus rationale.
Novelty: 1 duplicate, 2 minor new variation, 3 distinct change, 4 substantial
new mechanism, 5 substantially new justified direction.
Plausibility: 1 unsupported/invalid, 2 weak rationale, 3 reasonable HAR basis,
4 strong mechanism/evidence, 5 strongly justified for this source task.
Compatibility: 1 conflicting, 2 major interface problems, 3 unresolved
assumptions, 4 compatible with explicit dependencies, 5 directly composable.
Assess overlapping edits, dependencies, preprocessing/model assumptions,
tensor shapes, fixed labels, duplicated functionality, and mutation size.
Thresholds: novelty >=2, plausibility >=3, compatibility >=4. Do not predict
or fabricate measured scores. No relaxation for stagnation.
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
