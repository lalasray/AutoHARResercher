import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from autohar_researcher.types import Pipeline, Proposal
from autohar_researcher.patches import compose
from research.modules.architectures.presets import model_defaults

ROOT = Path(__file__).resolve().parents[1]


def test_colocated_author_source_checksums_match():
    root = ROOT / 'research/modules/architectures'
    for metadata in json.loads((root / 'ARCHITECTURES.json').read_text()).values():
        entry = metadata.get('source', {})
        if 'sha256' not in entry:
            continue
        assert hashlib.sha256((root / entry['file']).read_bytes()).hexdigest() == entry['sha256']


@pytest.mark.parametrize('name', ['tinyhar', 'tinierhar', 'deepconvlstmattention'])
def test_torch_adapter_matches_author_logits_and_gradients(name):
    torch.set_num_threads(1)
    params = dict(model_defaults(name), input_channels=6)
    adapter = importlib.import_module(f'research.modules.architectures.{name}').build_model(128 * 6, 5, params)
    assert adapter.original.__class__.__module__ == f'research.modules.architectures.author_{name}'
    import copy
    original = copy.deepcopy(adapter.original)
    adapter.eval()
    original.eval()
    x = torch.randn(2, 128, 6, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    actual = adapter(x)
    expected = original(reference_x.unsqueeze(1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    assert sum(p.numel() for p in adapter.parameters()) == sum(p.numel() for p in original.parameters())


def test_architecture_switch_uses_its_own_defaults():
    initial = Pipeline()
    switched = compose(initial, [Proposal('model', {'architecture': 'tinierhar'}, {}, 'Use original TinierHAR')])
    assert switched.model == model_defaults('tinierhar')
    assert 'head_dropout' not in switched.model
    assert switched.model['hidden_dim'] == 16 and switched.model['conv_filters'] == 4
    assert initial.model['architecture'] == 'cnn'


@pytest.mark.parametrize('layers', [1, 2])
def test_fused_deepconvlstm_matches_standard_lstm_cells(layers):
    from research.modules.architectures.deepconvlstm import DeepConvLSTM
    torch.set_num_threads(1)
    model = DeepConvLSTM(9, {'input_channels': 3, 'conv_filters': 4,
                            'hidden_dim': 8, 'lstm_layers': layers, 'dropout': 0}).double()
    reference = torch.nn.ModuleList([torch.nn.LSTMCell(12 if i == 0 else 8, 8).double()
                                    for i in range(layers)])
    with torch.no_grad():
        for index, layer in enumerate(reference):
            for parameter in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
                getattr(layer, parameter).copy_(getattr(model.lstm, f'{parameter}_l{index}'))
    x = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    actual, _ = model.lstm(x)
    expected = reference_x
    for layer in reference:
        hidden = expected.new_zeros(2, 8)
        cell = torch.zeros_like(hidden)
        outputs = []
        for value in expected.unbind(1):
            hidden, cell = layer(value, (hidden, cell))
            outputs.append(hidden)
        expected = torch.stack(outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-10, atol=1e-12)
    assert not any('peephole' in name for name, _ in model.named_parameters())
    logits = model(torch.randn(4, 50, 3, dtype=torch.float64))
    assert logits.shape == (4, 9)
    torch.nn.functional.cross_entropy(logits, torch.arange(4)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_lasagne_rmsprop_epsilon_placement():
    from autohar_researcher.optimizers import LasagneRMSprop
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    optimizer = LasagneRMSprop([parameter], lr=.01, alpha=.9, eps=1e-6)
    expected = parameter.detach().numpy().copy()
    average = np.zeros(2)
    for gradient in (np.array([1e-5, .5]), np.array([.2, -.1])):
        parameter.grad = torch.from_numpy(gradient)
        average = .9 * average + .1 * gradient**2
        expected -= .01 * gradient / np.sqrt(average + 1e-6)
        optimizer.step()
        np.testing.assert_allclose(parameter.detach(), expected, rtol=1e-14)


def test_spectra_stft_matches_explicit_windowed_dft_and_backpropagates():
    from research.modules.architectures.spectra import build_model
    model = build_model(100 * 6, 5, dict(model_defaults('spectra'), input_channels=6))
    x = torch.randn(2, 100, 6, requires_grad=True)
    actual = model.stft(x)
    frames = x.transpose(1, 2).unfold(-1, model.n_fft, model.hop)
    expected = torch.fft.rfft(frames * model.hann, dim=-1).abs()
    torch.testing.assert_close(actual, expected)
    output = model(x)
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert model.recurrent.bidirectional
    assert model.channel_attention.gamma.grad is not None


@pytest.mark.parametrize('name', ['bilstmattention', 'hart', 'mobilehart'])
def test_pytorch_ports_reload_and_backpropagate_without_reference_framework(name, monkeypatch):
    import builtins
    import copy
    from research.modules.architectures.provenance import model_provenance
    original_import = builtins.__import__

    def checked_import(module, *args, **kwargs):
        if module.split('.')[0] == 'tensorflow':
            raise AssertionError('PyTorch model must not import the reference framework')
        return original_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', checked_import)
    torch.set_num_threads(1)
    params = dict(model_defaults(name), input_channels=6)
    assert model_provenance(params)['status'] == 'author_architecture_port'
    factory = importlib.import_module(f'research.modules.architectures.{name}').build_model
    model = factory(128 * 6, 5, params).eval()
    x = torch.randn(2, 128, 6, requires_grad=True)
    logits = model(x)
    logits.square().sum().backward()
    assert logits.shape == (2, 5)
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    restored = factory(128 * 6, 5, params).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    torch.testing.assert_close(restored(x.detach()), logits.detach(), rtol=0, atol=0)


def test_lr_schedule_is_replayed_without_validation(manifest, tmp_path):
    from autohar_researcher.datasets import make_task
    from autohar_researcher.harness import HARHarness
    config = Pipeline.from_dict({'data': {'window_seconds': 1}, 'model': {
        'architecture': 'cnn', 'hidden_dim': 4, 'depth': 1, 'max_epochs': 3,
        'patience': 3, 'scheduler': 'plateau_validation_loss', 'lr_patience': 1, 'lr_factor': .1}})
    harness = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path, device='cpu')
    harness.evaluate(config)
    schedule = list(harness._lr_schedules[harness._key(config)])
    assert schedule[0] == .001 and schedule[1] == pytest.approx(.0001)
    x, y, _ = harness._training_arrays(config, fit=True)
    harness._fit(config, x, y, epochs=len(schedule))
    assert harness._last_lr_schedule == schedule


def test_torch_port_final_refit_saves_reloadable_original_weights(manifest, tmp_path):
    from autohar_researcher.datasets import make_task
    from autohar_researcher.harness import HARHarness
    config = Pipeline.from_dict({'data': {'window_seconds': 1}, 'model': {'architecture': 'bilstmattention', 'max_epochs': 1}})
    harness = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path, device='cpu')
    harness.evaluate(config)
    result = harness.finalize(config)
    assert 0 <= result['test_macro_f1'] <= 1
    saved = torch.load(tmp_path / 'final-model.pt', weights_only=False)
    assert saved['framework'] == 'pytorch'
    assert saved['architecture_provenance']['status'] == 'author_architecture_port'
    model = harness._build_model(config)
    model.load_state_dict(saved['state_dict'])
    output = model(torch.zeros(2, 50, 6))
    assert torch.isfinite(output).all()
    assert result['refit_learning_rates'] == saved['learning_rates']


def test_bilstm_head_dropout_is_tunable_with_author_default():
    from research.modules.architectures.bilstmattention import build_model
    params = dict(model_defaults('bilstmattention'), input_channels=6)
    assert build_model(128 * 6, 5, params).dropout.p == .5
    model = build_model(128 * 6, 5, dict(params, head_dropout=.3))
    assert model.dropout.p == .3
    x = torch.randn(2, 128, 6, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 5)
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
