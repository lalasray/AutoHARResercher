from __future__ import annotations

import copy
import importlib
import hashlib
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
from .preprocessing_artifacts import pipeline_id, snapshot_path, save_validation_snapshot, load_validation_snapshot
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


def custom_functions(pipeline: Pipeline, role: str, bindings=None) -> dict:
    source = pipeline.code.get(OWNED_FILES[role])
    namespace = {'component_params': copy.deepcopy(getattr(pipeline, role).get('custom_params', {})), **(bindings or {})}
    if source:

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
        self._validation_diagnostics: dict = {}
        self._raw_training_diagnostics: dict = {}
        self._preprocessing_states: dict = {}
        self._training_labels: dict = {}
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

    def _label_bindings(self, p):
        labels = self._training_labels.get(self._key(p), np.empty(0, dtype=np.int64))
        return {'activity_labels': list(self.task.labels),
                'training_labels': labels.copy(),
                'training_class_counts': np.bincount(labels, minlength=len(self.task.labels))}

    def _build_model(self, p: Pipeline):
        model_provenance(p.model, 'research/components/model.py' in p.code)
        params = dict(p.model, input_channels=len(self.task.channels), seed=self.seed, device=self.device)
        name = p.model['architecture']
        if not name.isidentifier() or name.startswith('_'):
            raise ValueError('Invalid architecture module')
        if name == 'custom':
            def factory(*args):
                raise ValueError('Custom architecture requires build_model; choose a registered model to use baseline_model')
        else:
            module = importlib.import_module(f'research.modules.architectures.{name}')
            factory = module.build_model
        def build_registered_model(architecture, overrides=None):
            from research.modules.architectures.presets import model_defaults
            settings = dict(model_defaults(architecture), input_channels=len(self.task.channels),
                            seed=self.seed, device=self.device)
            settings.update(overrides or {})
            module = importlib.import_module(f'research.modules.architectures.{architecture}')
            return module.build_model(round(p.data['window_seconds'] * self.task.sampling_hz) * len(self.task.channels),
                                      len(self.task.labels), settings)
        custom = custom_functions(p, 'model', {**self._label_bindings(p), 'baseline_model': factory,
                                             'build_registered_model': build_registered_model})
        if 'build_model' in custom:
            factory = custom['build_model']
        return factory(round(p.data['window_seconds'] * self.task.sampling_hz) * len(self.task.channels),
                       len(self.task.labels), params).to(self.device)

    def preflight(self, p: Pipeline) -> None:
        import torch
        validate_pipeline(p, self.sources)
        from research.comparisons import METHODS
        if p.model['architecture'] in METHODS:
            if self.task.protocol != 'lodo' or len(self.task.channels) != 6:
                raise ValueError('Comparison methods require six-channel LODO tasks')
            if p.model['architecture'] == 'sdmix' and len(p.data['sources'] or self.sources) < 2:
                raise ValueError('SDMix requires at least two source domains')
        self._seed()
        training_x, training_y, _ = windows(self._records(p, fit=False), self.task.labels,
                                           p.data['window_seconds'], p.data['overlap'], sampling_hz=self.task.sampling_hz)
        self._training_labels[self._key(p)] = training_y.copy()
        self._fit_preprocessor(p, training_x, training_y)
        from research.augmentations import validate_class_names
        validate_class_names(p.augmentation, self.task.labels)
        length = round(p.data['window_seconds'] * self.task.sampling_hz)
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

        train = self._records(p, fit=False)
        sample_x, y, _ = windows(train, self.task.labels, p.data['window_seconds'], p.data['overlap'], sampling_hz=self.task.sampling_hz)
        self._sample_training(p, sample_x, y)
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

    def _preprocess_stateless(self, p: Pipeline, x: np.ndarray) -> np.ndarray:
        custom = custom_functions(p, 'data')
        if 'preprocess' in custom:
            result = np.asarray(custom['preprocess'](x.copy()), dtype=np.float32)
            if result.shape != x.shape or not np.isfinite(result).all():
                raise ValueError('Preprocessing must preserve finite time/channel tensors')
            return result
        return x

    @staticmethod
    def _check_preprocessing_state(state):
        if state is None or isinstance(state, (str, bool, int)):
            return
        if isinstance(state, (float, np.number)):
            if not np.isfinite(state):
                raise ValueError('Preprocessor state must be finite')
            return
        if isinstance(state, np.ndarray):
            if state.dtype.kind not in 'biuf' or not np.isfinite(state).all():
                raise ValueError('Preprocessor arrays must be finite numeric arrays')
            return
        if isinstance(state, dict) and all(isinstance(k, str) for k in state):
            for value in state.values():
                HARHarness._check_preprocessing_state(value)
            return
        if isinstance(state, (list, tuple)):
            for value in state:
                HARHarness._check_preprocessing_state(value)
            return
        raise ValueError('Preprocessor state must contain only numeric arrays and plain serializable values')

    def _fit_preprocessor(self, p: Pipeline, training_x: np.ndarray, training_y=None):
        bindings = {'activity_labels': list(self.task.labels)}
        if training_y is not None:
            training_y = np.asarray(training_y, dtype=np.int64)
            if training_y.shape != (len(training_x),):
                raise ValueError('Training labels must align with preprocessing windows')
            bindings.update(training_labels=training_y.copy(),
                            training_class_counts=np.bincount(training_y, minlength=len(self.task.labels)))
        custom = custom_functions(p, 'data', bindings)
        has_fit, has_transform = 'fit_preprocessor' in custom, 'transform' in custom
        if has_fit != has_transform:
            raise ValueError('Fitted preprocessing requires both fit_preprocessor and transform')
        if has_fit:
            x = self._preprocess_stateless(p, training_x)
            state = custom['fit_preprocessor'](x.copy(), copy.deepcopy(p.data.get('custom_params', {})))
            self._check_preprocessing_state(state)
            self._preprocessing_states[self._key(p)] = copy.deepcopy(state)
        else:
            # Stateless pipelines have an explicit empty state, not a missing
            # entry. Evidence transfer must work the same way for all pipelines.
            self._preprocessing_states[self._key(p)] = None

    def _preprocess(self, p: Pipeline, x: np.ndarray) -> np.ndarray:
        result = self._preprocess_stateless(p, x)
        custom = custom_functions(p, 'data')
        if 'transform' in custom or 'fit_preprocessor' in custom:
            if 'transform' not in custom or 'fit_preprocessor' not in custom:
                raise ValueError('Fitted preprocessing requires both fit_preprocessor and transform')
            if self._key(p) not in self._preprocessing_states:
                raise ValueError('Preprocessor must be fitted on training data before transform')
            state = copy.deepcopy(self._preprocessing_states[self._key(p)])
            result = np.asarray(custom['transform'](result.copy(), state), dtype=np.float32)
            if result.shape != x.shape or not np.isfinite(result).all():
                raise ValueError('Fitted preprocessing must preserve finite time/channel tensors')
        return result

    def _augment(self, p, x, rng, labels=None, normalizer=None):
        import torch
        from research.torch_augmentations import augment
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if p.augmentation['kind'] != 'custom':
            return augment(x, p.augmentation, rng, labels, normalizer, self.task.labels)
        custom = custom_functions(p, 'augmentation', self._label_bindings(p))
        result = (custom['augment_labeled'](x.clone(), labels.clone(), rng)
                  if 'augment_labeled' in custom else custom['augment'](x.clone(), rng))
        if not isinstance(result, torch.Tensor) or result.device != x.device:
            raise ValueError('Custom augmentation must return a PyTorch tensor on the input device')
        if result.shape != x.shape or not torch.isfinite(result).all():
            raise ValueError('Augmentation must preserve finite tensor shape')
        mask = torch.rand(len(x), device=x.device, generator=rng) < p.augmentation.get('probability', 1.)
        return torch.where(mask[:, None, None], result, x)

    def _sample_training(self, p, x, y, *, return_indices=False):
        custom = custom_functions(p, 'data', {'activity_labels': self.task.labels})
        if 'sample_indices' not in custom:
            return (x, y, np.arange(len(y))) if return_indices else (x, y)
        indices = np.asarray(custom['sample_indices'](
            x.copy(), y.copy(), np.random.default_rng(self.seed)))
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= len(y))):
            raise ValueError('sample_indices must return nonempty in-range integer indices')

        if set(y[indices]) != set(y):
            raise ValueError('Custom sampling must preserve training class coverage')
        if self.task.protocol == 'lodo' and np.any(np.bincount(y[indices], minlength=len(self.task.labels)) < 100):
            raise ValueError('Custom LODO sampling requires at least 100 training windows per class')
        result = (x[indices], y[indices])
        return (*result, indices) if return_indices else result

    def _training_arrays(self, p: Pipeline, fit: bool):
        records = self._records(p, fit)
        x, y, domains = windows(records, self.task.labels, p.data['window_seconds'], p.data['overlap'], sampling_hz=self.task.sampling_hz)
        rng = np.random.default_rng(self.seed)
        selected = []
        for i, label in enumerate(self.task.labels):
            indices = np.flatnonzero(y == i)
            count = int(len(indices) * p.data.get('class_ratios', {}).get(label, 1))
            selected.extend(rng.permutation(indices)[:count])
        x, y = x[selected], y[selected]
        x, y, sample_indices = self._sample_training(p, x, y, return_indices=True)
        self._comparison_domains = domains[selected][sample_indices]
        if not fit:
            from .diagnostics import signal_summary
            self._raw_training_diagnostics[self._key(p)] = signal_summary(x, y, self.task.labels)
        self._training_labels[self._key(p)] = y.copy()
        self._fit_preprocessor(p, x, y)
        x = self._preprocess(p, x)

        normalizer = Normalizer(p.data['normalization']).fit(x)
        real_counts = np.bincount(y, minlength=len(self.task.labels))
        for request in p.augmentation['synthetic']:
            if self.generator is None:
                raise ValueError('Synthetic generation requires an MCP server')
            source_datasets = sorted({r.dataset for r in records})
            source_dataset = request.get('source_dataset', source_datasets[0])
            if source_dataset not in source_datasets:
                raise ValueError('Synthetic source_dataset must be a selected training source')
            generation_request = dict(request, source_dataset=source_dataset, channel_names=list(self.task.channels))
            generated = self.generator.generate(generation_request, x.shape[1], x.shape[2], self.seed, sampling_hz=self.task.sampling_hz)
            generated = self._preprocess(p, generated)
            class_count = int(real_counts[self.task.labels.index(request['label'])])
            count = min(len(generated), int(class_count * request.get('ratio', 1.0)))
            x = np.concatenate((x, generated[:count]))
            y = np.concatenate((y, np.full(count, self.task.labels.index(request['label']), dtype=int)))
            self._comparison_domains = np.concatenate((self._comparison_domains,
                                                       np.full(count, source_dataset)))
        self._training_labels[self._key(p)] = y.copy()
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
    def _optimizer(model, p: Pipeline, bindings=None):
        import torch
        from .optimizers import LasagneRMSprop
        optimizers = {'rmsprop': LasagneRMSprop, 'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW,
                      'nadam': torch.optim.NAdam, 'sgd': torch.optim.SGD}
        custom = custom_functions(p, 'model', bindings)
        if 'build_optimizer' in custom:
            optimizer = custom['build_optimizer'](model, copy.deepcopy(p.model))
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise ValueError('build_optimizer must return a torch optimizer')
            return optimizer
        options = {'lr': p.model['learning_rate'], 'weight_decay': p.model['weight_decay']}
        if p.model['optimizer'] != 'sgd':
            options['eps'] = p.model.get('optimizer_eps', 1e-8)
        if p.model['optimizer'] in ('adam', 'adamw', 'nadam'):
            options['betas'] = (p.model.get('optimizer_beta1', .9), p.model.get('optimizer_beta2', .999))
        if p.model['optimizer'] == 'rmsprop':
            options['alpha'] = p.model.get('optimizer_alpha', .9)
        return optimizers[p.model['optimizer']](model.parameters(), **options)

    def _fit(self, p: Pipeline, x: np.ndarray, y: np.ndarray, validation=None, epochs=None, normalizer=None, *, uncapped=False, monitor_name='validation_macro_f1', training_monitor=False):
        self._training_labels[self._key(p)] = np.asarray(y, dtype=np.int64).copy()
        from research.comparisons import METHODS
        if p.model['architecture'] in METHODS:
            if uncapped or training_monitor or monitor_name != 'validation_macro_f1':
                raise ValueError('Comparison methods use fixed pretraining budgets and source-validation selection')
            validate_pipeline(p, self.sources)
            from research.comparisons.training import fit
            return fit(self, p, x, y, validation, epochs)
        import torch
        import time
        import hashlib
        if training_monitor:
            if validation is not None:
                raise ValueError('Training-loss stopping cannot also monitor a validation/test set')
            monitor_name = 'training_loss'
        fit_started = time.monotonic()
        pipeline_id = hashlib.sha256(self._key(p).encode()).hexdigest()[:12]
        self._seed()
        model = self._build_model(p)
        optimizer = self._optimizer(model, p, self._label_bindings(p))
        weights = None
        if p.model['loss'] == 'weighted_cross_entropy':
            counts = np.bincount(y, minlength=len(self.task.labels))
            weights = torch.tensor(len(y) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=p.model['label_smoothing'])
        custom = custom_functions(p, 'model', self._label_bindings(p))
        if 'build_loss' in custom:
            criterion = custom['build_loss'](copy.deepcopy(p.model), weights)
            if isinstance(criterion, torch.nn.Module):
                criterion = criterion.to(self.device)
            if not callable(criterion):
                raise ValueError('build_loss must return a callable scalar loss')
        rng = torch.Generator(device=self.device).manual_seed(self.seed)
        from .torch_data import batches
        best_score, best_epoch, best_state, stale = -math.inf, 0, None, 0
        batch_size = p.model['batch_size']
        schedule_best, schedule_stale = math.inf, 0
        self._last_lr_schedule = []
        replay = self._lr_schedules.get(self._key(p), []) if validation is None else []
        def progress(epoch, validation_score=None):
            row = {'pipeline_id': pipeline_id, 'architecture': p.model['architecture'],
                   'stage': ('test_selected' if monitor_name == 'test_macro_f1' else 'source_search') if validation is not None else ('training_selected' if training_monitor else 'final_refit'),
                   'epoch': epoch, 'max_epochs': None if uncapped else (epochs or p.model['max_epochs']),
                   'elapsed_seconds': time.monotonic() - fit_started,
                   monitor_name: validation_score,
                   'best_' + monitor_name: (-best_score if training_monitor else best_score) if math.isfinite(best_score) else None}
            with (self.output / 'training-progress.jsonl').open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps(row), flush=True)
        if uncapped and validation is None and not training_monitor:
            raise ValueError('Uncapped training requires an early-stopping monitor')
        from itertools import count
        epoch_numbers = count(1) if uncapped else range(1, (epochs or p.model['max_epochs']) + 1)
        for epoch in epoch_numbers:
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
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError('Training loss must be a finite scalar tensor')
                loss.backward()
                optimizer.step()
            if training_monitor:


                model.eval()
                total_loss, count_samples = 0., 0
                with torch.no_grad():
                    for batch, target in batches(x, y, batch_size=batch_size, device=self.device):
                        value = criterion(model(batch), target)
                        if not torch.isfinite(value):
                            raise ValueError('Non-finite monitored training loss')
                        total_loss += float(value) * len(target)
                        count_samples += len(target)
                training_loss = total_loss / count_samples
                if -training_loss > best_score:
                    best_score, best_epoch, stale = -training_loss, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                progress(epoch, training_loss)
                if stale >= p.model['patience']:
                    break
                continue
            if validation is None:
                best_epoch = epoch
                progress(epoch)
                continue
            vx, vy, domains = validation
            pred = self._predict(model, vx, batch_size)
            score = float(np.mean(list(domain_scores(pred, vy, domains, len(self.task.labels)).values())))
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            progress(epoch, score)
            if stale >= p.model['patience']:
                break
            if p.model.get('scheduler') == 'plateau_validation_loss':
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

    def validation_preprocessing_artifacts(self, p: Pipeline):
        path = snapshot_path(self.output, p)
        return {'pipeline_id': pipeline_id(p), 'path': path.relative_to(self.output).as_posix(),
                'fit_scope': 'source_training_only', 'available': path.is_file()}

    def load_validation_preprocessing(self, p: Pipeline):
        return load_validation_snapshot(self.output, p, task_fingerprint=self.task.fingerprint, seed=self.seed)

    def _save_validation_preprocessing(self, p: Pipeline, payload):
        payload = {**payload, 'data_identity': self.task.identity_metadata()}
        return save_validation_snapshot(self.output, p, payload,
                                        task_fingerprint=self.task.fingerprint, seed=self.seed)

    def _validation_arrays(self, p: Pipeline):
        arrays = windows(self.task.validation, self.task.labels,
                         p.data['window_seconds'], p.data['overlap'], sampling_hz=self.task.sampling_hz)
        expected = {r.dataset for r in self.task.validation}
        if set(arrays[2]) != expected:
            raise ValueError('Every fixed source-validation domain must produce windows')
        return arrays

    def evaluate(self, p: Pipeline) -> Evaluation:
        if self._finalized:
            raise RuntimeError('Search is closed after finalization')
        x, y, normalizer = self._training_arrays(p, fit=False)


        vx, vy, domains = self._validation_arrays(p)
        from .diagnostics import signal_summary, classification_report
        raw_validation_stats = signal_summary(vx, vy, self.task.labels)
        vx = normalizer.transform(self._preprocess(p, vx))
        model, epoch = self._fit(p, x, y, (vx, vy, domains), normalizer=normalizer)
        predictions = self._predict(model, vx, p.model['batch_size'])
        scores = domain_scores(predictions, vy, domains, len(self.task.labels))
        training_predictions = self._predict(model, x, p.model['batch_size'])
        real_count = self._raw_training_diagnostics[self._key(p)]['window_count']
        diagnostics = {'labels': list(self.task.labels), 'channels': list(self.task.channels),
                       'validation_macro_f1': float(np.mean(list(scores.values()))),
                       **classification_report(predictions, vy, self.task.labels),
                       'training_class_counts': np.bincount(y, minlength=len(self.task.labels)).tolist(),
                       'validation_class_counts': np.bincount(vy, minlength=len(self.task.labels)).tolist(),
                       'training_predictions': classification_report(training_predictions, y, self.task.labels),
                       'real_training_predictions': classification_report(training_predictions[:real_count], y[:real_count], self.task.labels),
                       'real_training': self._raw_training_diagnostics.get(self._key(p)),
                       'real_validation': raw_validation_stats,
                       'training_model_inputs': signal_summary(x, y, self.task.labels),
                       'validation_model_inputs': signal_summary(vx, vy, self.task.labels),
                       'scope': 'source training and fixed source validation only; no test data'}

        for name, values, targets in [('training', x, y), ('validation', vx, vy)]:
            try:
                diagnostics[name + '_augmentation_probe'] = self._augmentation_diagnostics(
                    p, model, values, targets, normalizer)
            except Exception as exc:
                diagnostics[name + '_augmentation_probe'] = {'error': str(exc), 'diagnostic_only': True}
        self._save_validation_preprocessing(p, {
            'pipeline': p.payload(), 'state': self._preprocessing_states.get(self._key(p)),
            'normalization_mean': normalizer.mean, 'normalization_scale': normalizer.scale})
        diagnostics['validation_preprocessing'] = self.validation_preprocessing_artifacts(p)
        diagnostics['data_identity'] = self.task.identity_metadata()
        diagnostic_path = self.output / 'diagnostics'
        diagnostic_path.mkdir(exist_ok=True)
        (diagnostic_path / (hashlib.sha256(self._key(p).encode()).hexdigest() + '.json')).write_text(
            json.dumps(diagnostics, indent=2, allow_nan=False))
        self._validation_diagnostics[self._key(p)] = diagnostics
        (self.output / 'validation-diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        evaluation = Evaluation(float(np.mean(list(scores.values()))), scores, epoch)
        self._evaluations.setdefault(self._key(p), []).append(epoch)
        self._lr_schedules[self._key(p)] = list(self._last_lr_schedule)
        return evaluation

    def _augmentation_diagnostics(self, p, model, values, targets, normalizer):
        import torch
        from .diagnostics import probe_indices, signal_summary, classification_report
        from research.torch_augmentations import mix_batch
        indices = probe_indices(targets, len(self.task.labels))
        x, y = values[indices].copy(), targets[indices].copy()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        devices = [torch.device(self.device).index or torch.cuda.current_device()] if self.device.startswith('cuda') else []
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                rng = torch.Generator(device=self.device).manual_seed(self.seed + 7919)
                tensor_y = torch.as_tensor(y, device=self.device)
                augmented = self._augment(p, x, rng, tensor_y, normalizer)
                conventional = augmented.detach().cpu().numpy().copy()
                mixed, soft_targets = mix_batch(augmented, tensor_y, p.augmentation, rng,
                                                len(self.task.labels), self.task.labels)
                mixed_values = mixed.detach().cpu().numpy()
                masses = (torch.nn.functional.one_hot(soft_targets, len(self.task.labels)).float()
                          if soft_targets.ndim == 1 else soft_targets)
                return {'diagnostic_only': True, 'used_for_selection': False,
                        'before': signal_summary(x, y, self.task.labels),
                        'after_conventional': signal_summary(conventional, y, self.task.labels),
                        'after_mixing_by_recipient_class': signal_summary(mixed_values, y, self.task.labels),
                        'mixed_target_mass': masses.sum(0).cpu().tolist(),
                        'conventional_predictions': classification_report(
                            self._predict(model, conventional, p.model['batch_size']), y, self.task.labels)}
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def finalize(self, p: Pipeline) -> dict:
        if not self.task.test:
            raise ValueError('Final evaluation requires a held-out test partition')
        import torch
        marker = self.output / 'final-test.started'
        if self._finalized or marker.exists():
            raise RuntimeError('Final target evaluation is allowed only once')
        epochs = self._evaluations.get(self._key(p))
        if not epochs:
            raise ValueError('Final pipeline must have source-validation evidence')
        self._finalized = True

        with marker.open('x') as f:
            f.write(self.task.fingerprint)
        fit_epochs = max(1, int(median(epochs)))
        x, y, normalizer = self._training_arrays(p, fit=True)
        model, _ = self._fit(p, x, y, epochs=fit_epochs, normalizer=normalizer)

        tx, ty, domains, subjects = windows(self.task.test, self.task.labels,
                                  p.data['window_seconds'], p.data['overlap'], sampling_hz=self.task.sampling_hz,
                                  return_subjects=True)
        tx = normalizer.transform(self._preprocess(p, tx))
        predictions = self._predict(model, tx, p.model['batch_size'])
        scores = domain_scores(predictions, ty, domains, len(self.task.labels))
        report = {'test_macro_f1': float(np.mean(list(scores.values()))),
                  'data_identity': self.task.identity_metadata(),
                  'dataset_scores': scores, 'refit_epochs': fit_epochs,
                  'validation_best_epochs': epochs, 'seed': self.seed,
                  'task_fingerprint': self.task.fingerprint, 'refit_learning_rates': self._last_lr_schedule}
        if self.task.protocol == 'loso':
            report['test_subject_scores'] = domain_scores(predictions, ty, subjects, len(self.task.labels))
            report['test_subject_window_counts'] = {str(person): int(np.sum(subjects == person)) for person in set(subjects)}
        torch.save({'state_dict': model.state_dict(), 'pipeline': p.payload(),
                    'data_identity': self.task.identity_metadata(),
                    'preprocessing_fit_scope': 'source_training_and_validation_final_refit',
                    'validation_preprocessing': self.validation_preprocessing_artifacts(p),
                    'normalization_mean': normalizer.mean, 'normalization_scale': normalizer.scale,
                    'preprocessing_state': self._preprocessing_states.get(self._key(p)),
                    'labels': self.task.labels, 'channels': self.task.channels,
                    'framework': getattr(model, 'framework', 'pytorch'),
                    'architecture_provenance': model_provenance(p.model, 'research/components/model.py' in p.code),
                    'augmentation_provenance': policy_provenance(p.augmentation),
                    'learning_rates': self._last_lr_schedule}, self.output / 'final-model.pt')
        (self.output / 'final-test.json').write_text(json.dumps(report, indent=2))
        return report
