import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from autohar_researcher.planner import CodexBackend, OpenCodeBackend
from autohar_researcher.synthetic import MCPGenerator


@pytest.mark.parametrize('response', ['{"ok": true}', '[]', 'not json'])
def test_codex_transport_and_fail_closed(monkeypatch, response):
    from pathlib import Path
    def run(command, **kwargs):
        assert command[:2] == ['codex', 'exec']
        assert '--ignore-user-config' in command and '--ephemeral' in command
        assert command[command.index('--sandbox') + 1] == 'read-only'
        assert command[-1] == '-'
        assert 'Role: judge' in kwargs['input']
        assert 'features' not in kwargs
        Path(command[command.index('--output-last-message') + 1]).write_text(response)
    monkeypatch.setattr('subprocess.run', run)
    if response.startswith('{'):
        assert CodexBackend('explicit-model')('judge', 'test')['ok']
    else:
        with pytest.raises(ValueError):
            CodexBackend('explicit-model')('judge', 'test')


def test_opencode_uses_isolated_denied_tool_agent(monkeypatch):
    def run(command, **kwargs):
        config = json.loads(kwargs['env']['OPENCODE_CONFIG_CONTENT'])
        agent = config['agent']['autohar']
        assert agent['permission'] == {'*': 'deny'}
        assert agent['temperature'] == .2 and agent['top_p'] == .9
        assert agent['maxTokens'] == 4096
        assert '--pure' in command
        assert 'autohar-llm-' in command[command.index('--dir') + 1]
        return SimpleNamespace(stdout=json.dumps({'type': 'text', 'part': {'text': '{"ok": true}'}}))
    monkeypatch.setattr('subprocess.run', run)
    assert OpenCodeBackend('local/model')('judge', 'test')['ok']


def test_real_stdio_mcp_transport(tmp_path):
    pytest.importorskip('mcp')
    server = tmp_path / 'server.py'
    server.write_text('''try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP
mcp = FastMCP("test")
@mcp.tool()
def generate(label: str, count: int, sampling_hz: int, window_steps: int, channels: int, seed: int, prompt: str) -> dict:
    return {"samples": [[[0.0] * channels for _ in range(window_steps)] for _ in range(count)]}
mcp.run(transport="stdio")
''')
    generator = MCPGenerator({'text': {'command': sys.executable, 'args': [str(server)], 'tool': 'generate'}})
    samples = generator.generate({'generator': 'text', 'label': 'walking', 'count': 2, 'prompt': 'walk'}, 50, 6)
    assert samples.shape == (2, 50, 6)


@pytest.mark.parametrize('role,field,value', [
    ('data', 'balanced_sampling', True),
    ('model', 'learning_rate', .0005),
    ('augmentation', 'probability', .6),
])
@pytest.mark.parametrize('style', ['bare', 'wrapped', 'dotted'])
def test_specialist_patch_contract_and_aliases(role, field, value, style):
    from autohar_researcher.planner import LLMAgents
    from autohar_researcher.patches import check_proposal
    from autohar_researcher.types import Pipeline
    changes = {field: value}
    if style == 'wrapped':
        changes = {role: changes}
    elif style == 'dotted':
        changes = {role + '.' + field: value}
    def backend(called_role, prompt):
        assert called_role == role
        assert 'Authoritative ' + role + ' patch contract' in prompt
        return {'changes': changes, 'rationale': 'Test variation'}
    proposal = LLMAgents(backend).propose(role, {}, {})
    assert proposal.changes == {field: value}
    assert check_proposal(Pipeline(), proposal, {'uci_har'})['fields'] == 1


@pytest.mark.parametrize('changes', [
    {'data': {'balanced_sampling': True}},
    {'data.balanced_sampling': True},
    {'learning_rate': .001, 'model.learning_rate': .002},
    {'model': {'learning_rate': .001}, 'dropout': .2},
    {'unknown': 1}, [],
])
def test_aliases_do_not_bypass_ownership_or_ambiguity(changes):
    from autohar_researcher.planner import normalize_changes
    with pytest.raises(ValueError):
        normalize_changes('model', changes)


def test_nested_class_policy_is_preserved():
    from autohar_researcher.planner import normalize_changes
    changes = {'per_class': {'walking': {'kind': 'jitter', 'probability': .3}}}
    assert normalize_changes('augmentation', changes) == changes


def test_judge_receives_interfaces_and_static_validation_without_autoapproval():
    from autohar_researcher.planner import LLMAgents
    from autohar_researcher.types import Proposal
    def backend(role, prompt):
        assert role == 'judge'
        assert 'learning_rate' in prompt and 'Authoritative model patch contract' in prompt
        assert 'build_model(input_dim, num_classes, params)' in prompt
        assert 'already passed deterministic configuration validation' in prompt
        assert 'static validation does not prove runtime correctness' in prompt
        return {'novelty': 2, 'plausibility': 1, 'compatibility': 5, 'rationale': 'No coherent mechanism'}
    judgment = LLMAgents(backend).judge(Proposal('model', {'learning_rate': .0005}, {}, 'test'), {}, {}, {})
    assert not judgment.eligible


@pytest.mark.parametrize('source', [None, '', '# only a comment'])
def test_code_edit_mode_rejects_configuration_only_and_empty_code(source):
    from autohar_researcher.planner import LLMAgents
    def backend(role, prompt):
        assert 'CODE-EDIT SEARCH' in prompt
        return {'changes': {'learning_rate': .0005}, 'files': {'research/components/model.py': source} if source is not None else {}, 'rationale': 'test'}
    with pytest.raises(ValueError, match='changed executable'):
        LLMAgents(backend, code_edits=True).propose('model', {}, {})


def test_code_edit_mode_accepts_changed_owned_source():
    from autohar_researcher.planner import LLMAgents
    source = 'def preprocess(x):\n    return x.clip(-10, 10)\n'
    def backend(role, prompt):
        return {'changes': {}, 'files': {'research/components/data.py': source}, 'rationale': 'Test clipping'}
    assert LLMAgents(backend, code_edits=True).propose('data', {}, {}).files['research/components/data.py'] == source
