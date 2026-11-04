import numpy as np
import pytest
import torch

from autohar_researcher.datasets import make_task
from autohar_researcher.harness import HARHarness
from autohar_researcher.patches import validate_pipeline
from autohar_researcher.types import Pipeline


def pipeline():
    p = Pipeline.from_dict({'data': {'window_seconds': 1, 'overlap': 0},
        'model': {'architecture': 'custom', 'optimizer': 'custom', 'loss': 'custom',
                  'max_epochs': 1, 'patience': 1, 'batch_size': 64}})
    p.code['research/components/model.py'] = '''import torch
class Network(torch.nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.head = torch.nn.Linear(channels, classes)
    def forward(self, x):
        return self.head(x.mean(dim=1))
def build_model(input_dim, num_classes, params):
    return Network(params['input_channels'], num_classes)
def build_optimizer(model, params):
    return torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
def build_loss(params, class_weights):
    return torch.nn.CrossEntropyLoss()
'''
    p.code['research/components/data.py'] = '''import numpy as np
def sample_indices(x, y, rng):
    return np.concatenate([rng.permutation(len(y)), np.arange(len(y))])
def fit_preprocessor(x, params):
    return {'center': np.median(x, axis=(0, 1), keepdims=True)}
def transform(x, state):
    return x - state['center']
'''
    p.augmentation['kind'] = 'custom'
    p.code['research/components/augmentation.py'] = '''import torch
def augment_labeled(x, labels, rng):
    return x + 0.001 * torch.randn(x.shape, generator=rng, device=x.device)
'''
    return p


def test_all_custom_components_train_and_refit(manifest, tmp_path):
    torch.set_num_threads(1)
    task = make_task(manifest, 'loso', 'uci_har', '0')
    h = HARHarness(task, 13, tmp_path, device='cpu')
    p = pipeline()
    h.preflight(p)
    result = h.evaluate(p)
    assert np.isfinite(result.score)
    assert h.finalize(p)['refit_epochs'] == 1


@pytest.mark.parametrize('field,hook', [('architecture', 'build_model'),
    ('optimizer', 'build_optimizer'), ('loss', 'build_loss')])
def test_custom_requires_implementation(field, hook):
    p = pipeline()
    p.code['research/components/model.py'] = p.code['research/components/model.py'].replace('def ' + hook, 'def unused')
    with pytest.raises(ValueError, match=hook):
        validate_pipeline(p, {'uci_har'})


@pytest.mark.parametrize('expression', ['[-1]', '[999]', '[0.5]', '[]', '[0]'])
def test_sampler_rejects_invalid_selection(manifest, tmp_path, expression):
    h = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path, device='cpu')
    p = pipeline()
    p.code['research/components/data.py'] = 'def sample_indices(x,y,rng):\n    return ' + expression
    with pytest.raises(ValueError):
        h._sample_training(p, np.zeros((2, 50, 6)), np.array([0, 1]))


def test_training_label_bindings_and_label_free_transform(manifest, tmp_path):
    h = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path, device='cpu')
    p = pipeline()
    p.code['research/components/data.py'] = '''import numpy as np
def fit_preprocessor(x, params):
    counts = training_class_counts.copy()
    center = x[training_labels == 0].mean(axis=(0, 1), keepdims=True)
    training_labels[:] = 0
    return {'center': center, 'counts': counts}
def transform(x, state):
    return x - state['center']
'''
    x = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    y = np.array([0, 1, 1, 1])
    h._fit_preprocessor(p, x, y)
    state = h._preprocessing_states[h._key(p)]
    assert state['counts'][:2].tolist() == [1, 3]
    assert y.tolist() == [0, 1, 1, 1]
    assert np.allclose(h._preprocess(p, x), x - x[:1].mean(axis=(0, 1), keepdims=True))
    h._training_labels[h._key(p)] = y.copy()
    bindings = h._label_bindings(p)
    bindings['training_labels'][:] = 0
    assert h._label_bindings(p)['training_class_counts'][:2].tolist() == [1, 3]


def test_all_roles_use_training_labels_during_search_and_refit(manifest, tmp_path):
    torch.set_num_threads(1)
    h = HARHarness(make_task(manifest, 'loso', 'uci_har', '0'), 13, tmp_path, device='cpu')
    p = pipeline()
    p.code['research/components/data.py'] = p.code['research/components/data.py'].replace(
        'np.median(x, axis=(0, 1), keepdims=True)',
        'x[training_labels == training_labels[0]].mean(axis=(0, 1), keepdims=True)')
    p.code['research/components/model.py'] = p.code['research/components/model.py'].replace(
        "return Network(params['input_channels'], num_classes)",
        "return Network(params['input_channels'], len(training_class_counts))").replace(
        'return torch.nn.CrossEntropyLoss()',
        'return torch.nn.CrossEntropyLoss(weight=torch.tensor(1.0 / (training_class_counts + 1), dtype=torch.float32))').replace(
        "lr=params['learning_rate']", "lr=params['learning_rate'] / len(activity_labels)")
    p.code['research/components/augmentation.py'] = p.code['research/components/augmentation.py'].replace(
        '0.001 * torch.randn', '0.001 / len(training_labels) * torch.randn')
    h.preflight(p)
    assert np.isfinite(h.evaluate(p).score)
    search_count = len(h._label_bindings(p)['training_labels'])
    h.finalize(p)
    assert len(h._label_bindings(p)['training_labels']) > search_count
