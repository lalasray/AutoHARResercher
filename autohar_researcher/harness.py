"""Protected source-only training, Macro-F1 selection, and one-time final test."""
from __future__ import annotations

import copy
import importlib
import json
import math
import random
from pathlib import Path
from statistics import median

import numpy as np

from .datasets import Task, windows
from .patches import OWNED_FILES, validate_pipeline
from .synthetic import MCPGenerator
from .types import Evaluation, Pipeline
from research.modules.architectures.provenance import model_provenance
from research.augmentations import policy_provenance


def macro_f1(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    if len(predictions) != len(labels) or not len(labels):
        raise ValueError('Invalid metric inputs')
    scores = []
    for c in range(n_classes):
        tp = np.sum((predictions == c) & (labels == c))
        fp = np.sum((predictions == c) & (labels != c))
        fn = np.sum((predictions != c) & (labels == c))
        scores.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores))


def domain_scores(pred: np.ndarray, y: np.ndarray, domains: np.ndarray, n: int) -> dict[str, float]:
    return {str(d): macro_f1(pred[domains == d], y[domains == d], n) for d in sorted(set(domains))}


class Normalizer:
    def __init__(self, mode: str):
        self.mode = mode

    def fit(self, x: np.ndarray) -> Normalizer:
        self.mean = x.mean(axis=(0, 1), keepdims=True) if self.mode == 'standard' else np.zeros((1, 1, x.shape[-1]))
        self.scale = x.std(axis=(0, 1), keepdims=True) if self.mode == 'standard' else np.ones_like(self.mean)
        if self.mode == 'minmax':
            self.mean = x.min(axis=(0, 1), keepdims=True)
            self.scale = x.max(axis=(0, 1), keepdims=True) - self.mean
        self.scale = np.maximum(self.scale, 1e-8)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


def custom_functions(pipeline: Pipeline, role: str) -> dict:
    source = pipeline.code.get(OWNED_FILES[role])
    namespace = {}
    if source:
        # Static checks are a correctness boundary, not an OS security sandbox.
        exec(compile(source, OWNED_FILES[role], 'exec'), namespace)
    return namespace


class HARHarness:
    def __init__(self, task: Task, seed: int, output: Path, device: str = 'cuda',
                 generator: MCPGenerator | None = None):
        self.task = task
        self.seed = seed
        self.output = output
        import torch
        requested = torch.device(device)
        if requested.type not in {'cpu', 'cuda'}:
            raise ValueError('Training device must be cpu or cuda[:index]')
        if requested.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA training requested but PyTorch cannot access a GPU. Install a CUDA-enabled PyTorch build and check the NVIDIA driver, or explicitly use --device cpu.')
            if requested.index is not None and requested.index >= torch.cuda.device_count():
                raise ValueError(f'CUDA device index {requested.index} is unavailable')
        self.device = str(requested)
        self.generator = generator
        self._finalized = False
        self._evaluations: dict[str, list[int]] = {}
        self._lr_schedules: dict[str, list[float]] = {}
        self._last_lr_schedule: list[float] = []
        self.output.mkdir(parents=True, exist_ok=True)

    @property
    def sources(self) -> set[str]:
        return {r.dataset for r in self.task.train}

    def _seed(self):
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)

    def _build_model(self, p: Pipeline):
        model_provenance(p.model, 'research/components/model.py' in p.code)
        params = dict(p.model, input_channels=len(self.task.channels), seed=self.seed, device=self.device)
        custom = custom_functions(p, 'model')
        if 'build_model' in custom:
            factory = custom['build_model']
        else:
            name = p.model['architecture']
            if not name.isidentifier() or name.startswith('_'):
                raise ValueError('Invalid architecture module')
            module = importlib.import_module(f'research.modules.architectures.{name}')
            factory = module.build_model
        return factory(round(p.data['window_seconds'] * 50) * len(self.task.channels),
                       len(self.task.labels), params).to(self.device)

    def preflight(self, p: Pipeline) -> None:
        """Shape/import/dependency checks occur before candidate budget is charged."""
        import torch
        validate_pipeline(p, self.sources)
        self._seed()
        from research.augmentations import validate_class_names
        validate_class_names(p.augmentation, self.task.labels)
        length = round(p.data['window_seconds'] * 50)
        count = max(2, len(self.task.labels))
        labels = torch.arange(count, device=self.device) % len(self.task.labels)
        x = np.zeros((count, length, len(self.task.channels)), dtype=np.float32)
        x = self._preprocess(p, x)
        x = self._augment(p, x, torch.Generator(device=self.device).manual_seed(self.seed), labels=labels)
        from research.torch_augmentations import mix_batch
        x, _ = mix_batch(x, labels, p.augmentation, torch.Generator(device=self.device).manual_seed(self.seed), len(self.task.labels), self.task.labels)
        model = self._build_model(p)
        model.eval()
        with torch.no_grad():
            y = model(torch.as_tensor(x, device=self.device))
        if tuple(y.shape) != (count, len(self.task.labels)) or not torch.isfinite(y).all():
            raise ValueError('Model must return finite [batch, fixed classes] logits')
        for request in p.augmentation['synthetic']:
            if request['label'] not in self.task.labels:
                raise ValueError('Synthetic request changes the fixed label space')
            if self.generator is None or request['generator'] not in self.generator.servers:
                raise ValueError('Requested MCP generator is not configured')
        # Class coverage/source eligibility cannot be inferred from tensor checks.
        train = self._records(p, fit=False)
        _, y, _ = windows(train, self.task.labels, p.data['window_seconds'], p.data['overlap'])
        self._validation_arrays(p)
        counts = np.bincount(y, minlength=len(self.task.labels))
        ratios = p.data.get('class_ratios', {})
        if set(ratios) - set(self.task.labels):
            raise ValueError('Class sampling cannot change label space')
        counts = np.array([int(n * ratios.get(label, 1)) for n, label in zip(counts, self.task.labels)])
        minimum = 100 if self.task.protocol == 'lodo' else 1
        if np.any(counts < minimum):
            raise ValueError(f'Source subset requires at least {minimum} training windows per class')

    def _records(self, p: Pipeline, fit: bool):
        selected = set(p.data['sources']) or self.sources
        records = self.task.train + (self.task.validation if fit else ())
        return tuple(r for r in records if r.dataset in selected)

    def _preprocess(self, p: Pipeline, x: np.ndarray) -> np.ndarray:
        custom = custom_functions(p, 'data')
        if 'preprocess' in custom:
            result = np.asarray(custom['preprocess'](x.copy()), dtype=np.float32)
            if result.shape != x.shape or not np.isfinite(result).all():
                raise ValueError('Preprocessing must preserve finite time/channel tensors')
            return result
        return x

    def _augment(self, p, x, rng, labels=None, normalizer=None):
        import torch
        from research.torch_augmentations import augment
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if p.augmentation['kind'] != 'custom':
            return augment(x, p.augmentation, rng, labels, normalizer, self.task.labels)
        result = custom_functions(p, 'augmentation')['augment'](x.clone(), rng)
        if not isinstance(result, torch.Tensor) or result.device != x.device:
            raise ValueError('Custom augmentation must return a PyTorch tensor on the input device')
        if result.shape != x.shape or not torch.isfinite(result).all():
            raise ValueError('Augmentation must preserve finite tensor shape')
        mask = torch.rand(len(x), device=x.device, generator=rng) < p.augmentation.get('probability', 1.)
        return torch.where(mask[:, None, None], result, x)

    def _training_arrays(self, p: Pipeline, fit: bool):
        records = self._records(p, fit)
        x, y, _ = windows(records, self.task.labels, p.data['window_seconds'], p.data['overlap'])
        rng = np.random.default_rng(self.seed)
        selected = []
        for i, label in enumerate(self.task.labels):
            indices = np.flatnonzero(y == i)
            count = int(len(indices) * p.data.get('class_ratios', {}).get(label, 1))
            selected.extend(rng.permutation(indices)[:count])
        x, y = x[selected], y[selected]
        x = self._preprocess(p, x)
        # Refit from source train+validation only in final stage.
        normalizer = Normalizer(p.data['normalization']).fit(x)
        real_counts = np.bincount(y, minlength=len(self.task.labels))
        for request in p.augmentation['synthetic']:
            if self.generator is None:
                raise ValueError('Synthetic generation requires an MCP server')
            generated = self.generator.generate(request, x.shape[1], x.shape[2], self.seed)
            generated = self._preprocess(p, generated)
            class_count = int(real_counts[self.task.labels.index(request['label'])])
            count = min(len(generated), int(class_count * request.get('ratio', 1.0)))
            x = np.concatenate((x, generated[:count]))
            y = np.concatenate((y, np.full(count, self.task.labels.index(request['label']), dtype=int)))
        return normalizer.transform(x), y, normalizer

    def _predict(self, model, x, batch_size):
        import torch
        from .torch_data import batches
        model.eval()
        chunks = []
        with torch.no_grad():
            for (batch,) in batches(x, batch_size=batch_size, device=self.device):
                chunks.append(model(batch).argmax(1).cpu().numpy())
        return np.concatenate(chunks)

    @staticmethod
    def _optimizer(model, p: Pipeline):
        import torch
        from .optimizers import LasagneRMSprop
        optimizers = {'rmsprop': LasagneRMSprop, 'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW,
                      'nadam': torch.optim.NAdam, 'sgd': torch.optim.SGD}
        options = {'lr': p.model['learning_rate'], 'weight_decay': p.model['weight_decay']}
        if p.model['optimizer'] != 'sgd':
            options['eps'] = p.model.get('optimizer_eps', 1e-8)
        if p.model['optimizer'] in ('adam', 'adamw', 'nadam'):
            options['betas'] = (p.model.get('optimizer_beta1', .9), p.model.get('optimizer_beta2', .999))
        if p.model['optimizer'] == 'rmsprop':
            options['alpha'] = p.model.get('optimizer_alpha', .9)
        return optimizers[p.model['optimizer']](model.parameters(), **options)

    def _fit(self, p: Pipeline, x: np.ndarray, y: np.ndarray, validation=None, epochs=None, normalizer=None):
        import torch
        self._seed()
        model = self._build_model(p)
        optimizer = self._optimizer(model, p)
        weights = None
        if p.model['loss'] == 'weighted_cross_entropy':
            counts = np.bincount(y, minlength=len(self.task.labels))
            weights = torch.tensor(len(y) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=p.model['label_smoothing'])
        rng = torch.Generator(device=self.device).manual_seed(self.seed)
        from .torch_data import batches
        best_score, best_epoch, best_state, stale = -math.inf, 0, None, 0
        batch_size = p.model['batch_size']
        schedule_best, schedule_stale = math.inf, 0
        self._last_lr_schedule = []
        replay = self._lr_schedules.get(self._key(p), []) if validation is None else []
        for epoch in range(1, (epochs or p.model['max_epochs']) + 1):
            if epoch <= len(replay):
                for group in optimizer.param_groups:
                    group['lr'] = replay[epoch - 1]
            self._last_lr_schedule.append(optimizer.param_groups[0]['lr'])
            model.train()
            for batch, targets in batches(x, y, batch_size=batch_size, device=self.device,
                    seed=self.seed+epoch, shuffle=True, balanced=p.data['balanced_sampling']):
                batch = self._augment(p, batch, rng, labels=targets, normalizer=normalizer)
                from research.torch_augmentations import mix_batch
                batch, targets = mix_batch(batch, targets, p.augmentation, rng, len(self.task.labels), self.task.labels)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                optimizer.step()
            if validation is None:
                best_epoch = epoch
                continue
            vx, vy, domains = validation
            pred = self._predict(model, vx, batch_size)
            score = float(np.mean(list(domain_scores(pred, vy, domains, len(self.task.labels)).values())))
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= p.model['patience']:
                break
            if p.model.get('scheduler') == 'plateau_validation_loss':
                # The author TinierHAR schedule monitors source validation loss;
                # checkpoint selection and promotion still use Macro-F1.
                losses = []
                model.eval()
                with torch.no_grad():
                    for batch, target in batches(vx, vy, batch_size=batch_size, device=self.device):
                        logits = model(batch)
                        loss = torch.nn.functional.cross_entropy(logits, target,
                            weight=weights.to(logits.device) if weights is not None else None,
                            label_smoothing=p.model['label_smoothing'])
                        losses.append(float(loss))
                current = float(np.mean(losses))
                if schedule_best == math.inf:
                    schedule_best, schedule_stale = current, 1
                elif current < schedule_best:
                    schedule_best, schedule_stale = current, 0
                else:
                    schedule_stale += 1
                if schedule_stale >= p.model['lr_patience']:
                    for group in optimizer.param_groups:
                        group['lr'] *= p.model['lr_factor']
                    schedule_stale = 0
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch

    @staticmethod
    def _key(p: Pipeline) -> str:
        return json.dumps(p.payload(), sort_keys=True)

    def _validation_arrays(self, p: Pipeline):
        arrays = windows(self.task.validation, self.task.labels,
                         p.data['window_seconds'], p.data['overlap'])
        expected = {r.dataset for r in self.task.validation}
        if set(arrays[2]) != expected:
            raise ValueError('Every fixed source-validation domain must produce windows')
        return arrays

    def evaluate(self, p: Pipeline) -> Evaluation:
        if self._finalized:
            raise RuntimeError('Search is closed after finalization')
        x, y, normalizer = self._training_arrays(p, fit=False)
        # Always evaluate ALL fixed source-validation domains, even when the
        # candidate selects fewer domains for training.
        vx, vy, domains = self._validation_arrays(p)
        vx = normalizer.transform(self._preprocess(p, vx))
        model, epoch = self._fit(p, x, y, (vx, vy, domains), normalizer=normalizer)
        scores = domain_scores(self._predict(model, vx, p.model['batch_size']), vy,
                               domains, len(self.task.labels))
        evaluation = Evaluation(float(np.mean(list(scores.values()))), scores, epoch)
        self._evaluations.setdefault(self._key(p), []).append(epoch)
        self._lr_schedules[self._key(p)] = list(self._last_lr_schedule)
        return evaluation

    def finalize(self, p: Pipeline) -> dict:
        import torch
        marker = self.output / 'final-test.started'
        if self._finalized or marker.exists():
            raise RuntimeError('Final target evaluation is allowed only once')
        epochs = self._evaluations.get(self._key(p))
        if not epochs:
            raise ValueError('Final pipeline must have source-validation evidence')
        self._finalized = True
        # Atomic persistent guard also prevents repeated testing after a crash.
        with marker.open('x') as f:
            f.write(self.task.fingerprint)
        fit_epochs = max(1, int(median(epochs)))
        x, y, normalizer = self._training_arrays(p, fit=True)
        model, _ = self._fit(p, x, y, epochs=fit_epochs, normalizer=normalizer)
        # First access to held-out signal arrays in the entire workflow.
        tx, ty, domains = windows(self.task.test, self.task.labels,
                                  p.data['window_seconds'], p.data['overlap'])
        tx = normalizer.transform(self._preprocess(p, tx))
        scores = domain_scores(self._predict(model, tx, p.model['batch_size']), ty,
                               domains, len(self.task.labels))
        report = {'test_macro_f1': float(np.mean(list(scores.values()))),
                  'dataset_scores': scores, 'refit_epochs': fit_epochs,
                  'validation_best_epochs': epochs, 'seed': self.seed,
                  'task_fingerprint': self.task.fingerprint, 'refit_learning_rates': self._last_lr_schedule}
        torch.save({'state_dict': model.state_dict(), 'pipeline': p.payload(),
                    'normalization_mean': normalizer.mean, 'normalization_scale': normalizer.scale,
                    'labels': self.task.labels, 'channels': self.task.channels,
                    'framework': getattr(model, 'framework', 'pytorch'),
                    'architecture_provenance': model_provenance(p.model, 'research/components/model.py' in p.code),
                    'augmentation_provenance': policy_provenance(p.augmentation),
                    'learning_rates': self._last_lr_schedule}, self.output / 'final-model.pt')
        (self.output / 'final-test.json').write_text(json.dumps(report, indent=2))
        return report
