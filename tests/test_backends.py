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
