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


def test_pinned_sources_are_byte_identical():
    root = ROOT / 'research/vendor'
    for entry in json.loads((root / 'SOURCES.json').read_text()).values():
        assert hashlib.sha256((root / entry['file']).read_bytes()).hexdigest() == entry['sha256']


@pytest.mark.parametrize('name', ['tinyhar', 'tinierhar', 'deepconvlstmattention'])
def test_torch_adapter_matches_author_logits_and_gradients(name):
    torch.set_num_threads(1)
    params = dict(model_defaults(name), input_channels=6)
    adapter = importlib.import_module(f'research.modules.architectures.{name}').build_model(128 * 6, 5, params)
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


def test_deepconvlstm_peepholes_match_independent_numpy_recurrence():
    from research.modules.architectures.deepconvlstm import PeepholeLSTM
    layer = PeepholeLSTM(3, 2).double()
    x = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    actual = layer(x)
    wi, wh, bias, peepholes = [p.detach().numpy() for p in (layer.weight_ih, layer.weight_hh, layer.bias, layer.peepholes)]
    hidden = np.zeros((2, 2))
    cell = np.zeros_like(hidden)
    output = []
    sigmoid = lambda a: 1 / (1 + np.exp(-a))
    for step in x.detach().numpy().transpose(1, 0, 2):
        i, f, candidate, o = np.split(step @ wi + hidden @ wh + bias, 4, axis=1)
        cell = sigmoid(f + cell * peepholes[1]) * cell + sigmoid(i + cell * peepholes[0]) * np.tanh(candidate)
        hidden = sigmoid(o + cell * peepholes[2]) * np.tanh(cell)
        output.append(hidden)
    np.testing.assert_allclose(actual.detach(), np.stack(output, axis=1), atol=1e-14)
    actual.sum().backward()
    assert torch.isfinite(layer.peepholes.grad).all()
    assert layer.peepholes.grad.abs().sum() > 0


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


def test_torch_port_bilstm_matches_original_uci_graph():
    tf = pytest.importorskip('tensorflow')
    from research.modules.architectures.bilstmattention import build_model
    from types import SimpleNamespace
    source = (ROOT / 'research/vendor/bilstmattention.py').read_text()
    # Execute only the published model-definition block, never data loading/fit.
    source = source[source.index('class Attention'):source.index('model.compile')]
    namespace = dict(tf=tf, K=tf.keras.backend, Input=tf.keras.Input, Model=tf.keras.Model,
        Parameter=SimpleNamespace(kernel_size=3, pool_size=2, drop_rate=.2, number=6))
    namespace.update({name: getattr(tf.keras.layers, name) for name in ('Layer', 'Conv1D',
        'BatchNormalization', 'MaxPooling1D', 'Dropout', 'LSTM', 'Dense', 'Activation',
        'Concatenate', 'Bidirectional', 'LayerNormalization', 'Add')})
    exec(compile(source, 'original-model-block', 'exec'), namespace)
    reference = namespace['model']
    adapter = build_model(128 * 9, 6, dict(model_defaults('bilstmattention'), input_channels=9))
    from author_weight_transfer import transfer_bilstm
    transfer_bilstm(adapter, reference)
    adapter.eval()
    x = np.random.default_rng(13).normal(size=(2, 128, 9)).astype('float32')
    xf = tf.Variable(x)
    with tf.GradientTape() as tape:
        expected = reference([xf[:, :, i:i+3] for i in (0, 3, 6)], training=False)
        scalar = expected[:, 0]
    gradient = tape.gradient(scalar, xf).numpy()
    xt = torch.tensor(x, requires_grad=True)
    actual = adapter(xt).softmax(-1)
    actual[:, 0].sum().backward()
    np.testing.assert_allclose(actual.detach().numpy(), expected.numpy(), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(xt.grad.numpy(), gradient, atol=2e-6, rtol=3e-4)


@pytest.mark.parametrize('name,function', [('hart', 'HART'), ('mobilehart', 'mobileHART_XS')])
def test_torch_port_hart_matches_author_model(name, function):
    pytest.importorskip('tensorflow')
    from research.vendor import hart
    reference = getattr(hart, function)((128, 6), 5)
    adapter = importlib.import_module(f'research.modules.architectures.{name}').build_model(128 * 6, 5,
        dict(model_defaults(name), input_channels=6))
    from author_weight_transfer import transfer_hart, transfer_mobile
    (transfer_hart if name == 'hart' else transfer_mobile)(adapter, reference)
    adapter.eval()
    x = np.random.default_rng(13).normal(size=(2, 128, 6)).astype('float32')
    xt = torch.tensor(x, requires_grad=True)
    with __import__('tensorflow').GradientTape() as tape:
        xf = __import__('tensorflow').Variable(x)
        tape.watch(xf)
        expected = reference(xf, training=False)
        scalar = expected[:, 0]
    expected_gradient = tape.gradient(scalar, xf).numpy()
    actual = adapter(xt).softmax(-1)
    actual[:, 0].sum().backward()
    np.testing.assert_allclose(actual.detach().numpy(), expected.numpy(), atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(xt.grad.numpy(), expected_gradient, atol=3e-6, rtol=3e-4)


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
