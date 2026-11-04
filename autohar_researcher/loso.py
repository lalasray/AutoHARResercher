from dataclasses import replace
import hashlib
import json
from statistics import median
import time
from urllib.parse import quote

import numpy as np

from .harness import HARHarness
from .types import Evaluation
from .benchmark_data import subject_order
from .evaluation_reporting import loso_test_statistics
from .data_identity import digest_json, task_fingerprint


class LOSOHarness:
    def __init__(self, tasks, seed, output, device='cuda', generator=None, *, allow_partial_folds=False):
        if not tasks or any(task.protocol != 'loso' for task in tasks):
            raise ValueError('Dataset search requires LOSO folds')
        first = tasks[0]
        if any(task.data_fingerprint != first.data_fingerprint for task in tasks):
            raise ValueError('LOSO folds must use the same verified recording data identity')
        if any((task.target, task.labels, task.channels, task.sampling_hz) !=
               (first.target, first.labels, first.channels, first.sampling_hz) for task in tasks):
            raise ValueError('LOSO folds must describe the same dataset and class space')
        self.folds = []
        self.subjects = []
        covered = set()
        for task in tasks:
            people = {r.subject for r in task.test}
            if not people:
                raise ValueError('Each LOSO fold must hold out at least one person')
            person = min(people, key=subject_order)
            if covered.intersection(people):
                raise ValueError('Duplicate held-out subject or overlapping test groups')
            covered.update(people)
            self.subjects.append(person)
            fold = HARHarness(task, seed, output / 'folds' / f'subject-{quote(person, safe="")}', device, generator)
            (fold.output / 'splits.json').write_text(json.dumps(task.split_manifest(), indent=2))
            self.folds.append(fold)
        all_people = {r.subject for task in tasks for r in task.train + task.validation + task.test}
        if not allow_partial_folds and covered != all_people:
            raise ValueError('Dataset search requires a held-out fold for every subject')
        self.available_subject_count = len(all_people)
        split = digest_json(dict(scheme='dataset-loso-splits-v1', folds=[t.split_fingerprint or t.fingerprint for t in tasks]))
        fingerprint = task_fingerprint(first.data_fingerprint, split, first.manifest_sha256)
        self.task = replace(first, protocol='dataset-loso', fingerprint=fingerprint, split_fingerprint=split, test=())
        self.seed, self.output, self.generator = seed, output, generator
        self.device = self.folds[0].device
        self._evaluations, self._lr_schedules, self._validation_diagnostics = {}, {}, {}
        self._evidence = {}
        self._finalized = False
        self.candidate_workers, self.worker_threads = 1, 1

    @property
    def sources(self):
        return self.folds[0].sources

    def _key(self, pipeline):
        return self.folds[0]._key(pipeline)

    def validation_preprocessing_artifacts(self, pipeline):
        artifacts = []
        for fold in self.folds:
            reference = fold.validation_preprocessing_artifacts(pipeline)
            reference['path'] = (fold.output / reference['path']).relative_to(self.output).as_posix()
            reference['test_subjects'] = sorted({r.subject for r in fold.task.test}, key=subject_order)
            artifacts.append(reference)
        return {'fit_scope': 'source_training_only_per_fold', 'folds': artifacts}

    def preflight(self, pipeline):
        if self._finalized:
            raise RuntimeError('Search is closed after finalization')
        for fold in self.folds:
            fold.preflight(pipeline)

    def _aggregate(self, pipeline, results, elapsed, batch_elapsed=None):

        if len(results) != len(self.folds) or any(not np.isfinite(r.score) for r in results):
            raise ValueError('Every fold must produce a finite validation score')
        key = self._key(pipeline)
        score = float(np.mean([r.score for r in results]))
        epoch = max(1, int(median(r.best_epoch for r in results)))
        self._evaluations.setdefault(key, []).append(epoch)
        self._lr_schedules[key] = []
        fold_diagnostics = [{'subject': person, 'test_subjects': sorted({r.subject for r in fold.task.test}, key=subject_order),
                             'diagnostics': fold._validation_diagnostics.get(key, {})}
                            for person, fold in zip(self.subjects, self.folds)]
        matrices = [item['diagnostics'].get('confusion_true_rows_predicted_columns') for item in fold_diagnostics]
        self._validation_diagnostics[key] = {'mean_validation_macro_f1': score,
            'fold_count': len(results), 'scope': 'source validation across all folds; pooled confusion is descriptive only',
            'labels': list(self.task.labels),
            'confusion_true_rows_predicted_columns': (np.sum(matrices, axis=0).tolist()
                if all(matrix is not None for matrix in matrices) else None),
            'folds': fold_diagnostics}
        evidence = {'pipeline_id': hashlib.sha256(key.encode()).hexdigest(),
                    'validation_macro_f1': score, 'elapsed_seconds': elapsed,
                    'batch_elapsed_seconds': batch_elapsed,
                    'folds': [{'subject': person, 'test_subjects': sorted({r.subject for r in fold.task.test}, key=subject_order),
                               'validation_macro_f1': result.score,
                               'best_epoch': result.best_epoch,
                               'epochs': fold._evaluations[key],
                               'learning_rates': fold._lr_schedules[key]}
                              for person, fold, result in zip(self.subjects, self.folds, results)]}
        self._evidence[key] = evidence
        with (self.output / 'candidate-folds.jsonl').open('a') as handle:
            handle.write(json.dumps(evidence) + '\n')

        return Evaluation(score, {self.task.target: score}, epoch)

    def evaluate(self, pipeline):
        if self._finalized:
            raise RuntimeError('Search is closed after finalization')
        start = time.monotonic()
        results = [fold.evaluate(pipeline) for fold in self.folds]
        return self._aggregate(pipeline, results, time.monotonic() - start)

    def evaluate_many(self, pipelines, workers, threads):
        from .parallel import evaluate_many
        if self._finalized:
            raise RuntimeError('Search is closed after finalization')
        start = time.monotonic()

        results = [list(evaluate_many(fold, pipelines, workers, threads)) for fold in self.folds]
        for index, pipeline in enumerate(pipelines):
            candidate_results = [fold_results[index] for fold_results in results]
            failure = next((r for r in candidate_results if isinstance(r, Exception)), None)
            if failure is not None:
                yield failure
            else:

                result = self._aggregate(pipeline, candidate_results, None, time.monotonic() - start)
                yield result

    def selection_evidence(self, pipeline):
        return self._evidence[self._key(pipeline)]

    def finalize(self, pipeline, subject=None):
        key = self._key(pipeline)
        if self._finalized or (self.output / 'final-test.started').exists():
            raise RuntimeError('Final target evaluation is allowed only once')
        if key not in self._evidence:
            raise ValueError('Final pipeline needs validation evidence from every fold')
        if subject is not None and not any(subject in {r.subject for r in fold.task.test} for fold in self.folds):
            raise ValueError('Unknown final evaluation subject')
        self._finalized = True
        with (self.output / 'final-test.started').open('x') as handle:
            handle.write(self.task.fingerprint)
        reports = [{'subject': person, 'test_subjects': sorted({r.subject for r in fold.task.test}, key=subject_order), **fold.finalize(pipeline)}
                   for person, fold in zip(self.subjects, self.folds)
                   if subject is None or subject in {r.subject for r in fold.task.test}]
        report = {**loso_test_statistics(reports),
                  'data_identity': self.task.identity_metadata(),
                  'tested_subjects': [person for r in reports for person in r['test_subjects']],
                  'available_subject_count': self.available_subject_count,
                  'final_loso': reports,
                  'evaluation_scope': 'exploratory shared pipeline selected by mean validation across subject folds; subjects participate in discovery in other folds'}
        (self.output / 'final-test.json').write_text(json.dumps(report, indent=2))
        return report
