from __future__ import annotations

import argparse
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .agents import Search, SingleAgentSearch
from .baselines import initial_candidates
from .benchmark_data import loso_groups, subject_order
from .datasets import load_dataset_snapshot, make_task, make_fixed_validation_task
from .harness import HARHarness
from .loso import LOSOHarness
from .evaluation_reporting import loso_test_statistics
from .planner import CodexBackend, LLMAgents, OpenCodeBackend
from .synthetic import MCPGenerator
from .types import Pipeline, SEEDS, LODO_DATASETS
from research.modules.architectures.provenance import model_provenance
from research.augmentations import policy_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description='AutoResearcHAR source-only LOSO/LODO search')
    parser.add_argument('command', choices=['splits', 'search'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--protocol', choices=['loso', 'lodo'], required=True)
    parser.add_argument('--target', default='uci_har', help='Dataset target, or all for all three LODO targets (default: uci_har)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--subject', help='Held-out test subject; dataset-level discovery still scores every fold')
    group.add_argument('--all-subjects', action='store_true', help='Evaluate all LOSO subjects after dataset-level discovery')
    parser.add_argument('--search-mode', choices=['fixed-validation', 'loso', 'independent-loso'], help='LOSO search scope: dataset-level loso (default), fixed validation, or independent search per test subject')
    parser.add_argument('--validation-subject', help='Fixed validation person; default: deterministic choice using split seed')
    parser.add_argument('--split-seed', type=int, default=13, help='Fixed across search seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--method', choices=['multi-agent', 'single-agent', 'fixed', 'contratsc', 'sdmix', 'crosshar'], default='multi-agent')
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--budget', type=int, default=300)
    parser.add_argument('--baseline', type=Path, help='Explicit initial pipeline JSON; omit to start from the generic CNN')
    parser.add_argument('--output', type=Path, default=Path('artifacts/run'))
    parser.add_argument('--device', default='cuda', help='Training device: cuda (default), cuda:N, or explicit cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--proposal-mode', choices=['directed', 'code', 'configuration'], default='directed', help='Directed broad exploration, mandatory code edits, or configuration-permitted search')
    parser.add_argument('--candidate-workers', type=int, choices=range(1, 5), default=1, help='Isolated parallel candidate fits per round (1–4)')
    parser.add_argument('--backend', choices=['codex', 'opencode'], default=os.getenv('AUTOHAR_BACKEND', 'codex'))
    parser.add_argument('--opencode-bin', default='opencode')
    parser.add_argument('--codex-bin', default='codex')
    parser.add_argument('--model', default=os.getenv('AUTOHAR_MODEL'))
    parser.add_argument('--mcp-config', type=Path, help='Fixed text/video generator server configuration')
    parser.add_argument('--final-test', action='store_true', help='Refit selected pipeline and evaluate target once')
    args = parser.parse_args()
    if args.protocol == 'lodo' and (args.subject or args.all_subjects):
        parser.error('LODO holds out a dataset, not a subject')
    if args.protocol == 'lodo' and (args.search_mode or args.validation_subject is not None):
        parser.error('Search mode and validation subject apply only to LOSO')
    if args.target == 'all' and args.protocol != 'lodo':
        parser.error('--target all applies only to LODO')
    mode = args.search_mode or 'loso'
    if args.protocol == 'loso' and mode == 'fixed-validation' and (args.subject is not None or args.all_subjects):
        parser.error('Use --validation-subject with fixed-validation search')
    if mode != 'fixed-validation' and args.validation_subject is not None:
        parser.error('--validation-subject requires --search-mode fixed-validation')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Search seeds must be unique')
    dataset_snapshot = load_dataset_snapshot(args.manifest)
    raw, records = dataset_snapshot.raw, dataset_snapshot.recordings
    subjects = sorted({r.subject for r in records if r.dataset == args.target}, key=subject_order) if args.protocol == 'loso' else [args.subject]
    if not subjects:
        parser.error('No people found for the requested dataset')
    available_subject_count = len(subjects)
    test_subjects = None
    if args.protocol == 'loso':
        try:
            groups = loso_groups(subjects, raw['datasets'][args.target].get('loso_groups'))
            subjects = [person for group in groups for person in group]
        except ValueError as exc:
            parser.error(str(exc))
        test_subjects = subjects
    if args.protocol == 'loso' and mode == 'independent-loso' and args.subject is not None:
        if args.subject not in subjects:
            parser.error('Unknown held-out subject')
        subjects = [args.subject]
    if args.protocol == 'lodo':
        targets = LODO_DATASETS if args.target == 'all' else (args.target,)
        tasks = [make_task(args.manifest, 'lodo', target, split_seed=args.split_seed, snapshot=dataset_snapshot) for target in targets]
        subjects = [None] * len(tasks)
    elif args.protocol == 'loso' and mode == 'fixed-validation':
        tasks = [make_fixed_validation_task(args.manifest, args.target, args.validation_subject, args.split_seed, snapshot=dataset_snapshot)]
        subjects = [None]
    else:
        tasks = [make_task(args.manifest, args.protocol, args.target, subject, args.split_seed, snapshot=dataset_snapshot) for subject in subjects]
    if args.protocol == 'loso' and args.subject is not None and args.subject not in subjects:
        parser.error('Unknown held-out subject')
    if args.protocol == 'loso' and mode != 'fixed-validation':
        unique = {}
        for person, task in zip(subjects, tasks):
            unique.setdefault(task.fingerprint, (person, task))
        subjects, tasks = map(list, zip(*unique.values()))
    if args.command == 'splits':
        print(json.dumps([task.split_manifest() for task in tasks], indent=2))
        return
    comparison_methods = ('contratsc', 'sdmix', 'crosshar')
    fixed_method = args.method == 'fixed' or args.method in comparison_methods
    if args.method in comparison_methods and args.protocol != 'lodo':
        parser.error('Comparison methods require --protocol lodo')
    if not fixed_method and not args.model:
        parser.error('Search requires explicit --model (or AUTOHAR_MODEL); use opencode for local models, codex for remote models')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance_file = args.output / 'data-provenance.json'
    provenance = dataset_snapshot.provenance()
    if provenance_file.exists():
        if json.loads(provenance_file.read_text(encoding='utf-8')) != json.loads(json.dumps(provenance)):
            raise ValueError('Existing dataset provenance differs; use a fresh output directory')
    else:
        provenance_file.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    import torch
    torch.set_num_threads(args.threads)
    baseline = Pipeline.from_dict(json.loads(args.baseline.read_text())) if args.baseline else None
    if args.method in comparison_methods:
        if baseline is None:
            baseline = Pipeline.from_dict({'model': {'architecture': args.method}})
        elif baseline.model['architecture'] != args.method:
            parser.error('--baseline architecture must match the comparison --method')
    backend = (OpenCodeBackend(args.model, args.opencode_bin) if args.backend == 'opencode' else
               CodexBackend(args.model, args.codex_bin))
    agents = LLMAgents(backend, code_edits=args.proposal_mode == 'code')
    generator = MCPGenerator(json.loads(args.mcp_config.read_text())) if args.mcp_config else None
    summaries = []
    dataset_loso = args.protocol == 'loso' and mode == 'loso'
    jobs = [(None, tasks[0])] if dataset_loso else list(zip(subjects, tasks))
    for seed in args.seeds:
        for subject, task in jobs:
            output = args.output / f'seed-{seed}' / (f'subject-{quote(str(subject), safe="")}' if subject is not None else ('fixed-validation' if task.protocol == 'fixed-validation' else task.target))
            if dataset_loso:
                output = args.output / f'seed-{seed}' / args.target
                harness = LOSOHarness(tasks, seed, output, args.device, generator,
                                      allow_partial_folds=len({r.subject for t in tasks for r in t.test}) < available_subject_count)
                task = harness.task
            else:
                harness = HARHarness(task, seed, output, args.device, generator)
            harness.candidate_workers = args.candidate_workers
            harness.worker_threads = args.threads
            initial = baseline if baseline is not None else initial_candidates(task)
            search_class = SingleAgentSearch if args.method == 'single-agent' else Search
            search = search_class(harness, agents, output, args.rounds, 1 if fixed_method else args.budget,
                                  exploration=args.proposal_mode == 'directed' and args.method == 'multi-agent')
            (output / 'splits.json').write_text(json.dumps([t.split_manifest() for t in tasks] if dataset_loso else task.split_manifest(), indent=2))
            (output / 'run.json').write_text(json.dumps({
                'seed': seed, 'split_seed': args.split_seed, 'rounds': args.rounds, 'search_mode': task.protocol,
                'data_identity': task.identity_metadata(),
                'data_provenance_file': os.path.relpath(provenance_file, output).replace('\\', '/'),
                'test_subjects': test_subjects, 'loso_groups': raw['datasets'][task.target].get('loso_groups'),
                'available_subject_count': available_subject_count,
                'proposal_mode': args.proposal_mode, 'candidate_workers': args.candidate_workers, 'worker_threads': args.threads,
                'budget': search.budget, 'method': args.method, 'backend': args.backend, 'model': args.model, 'llm_scope': 'local' if args.backend == 'opencode' else 'remote',
                'temperature': .2 if args.backend == 'opencode' else None,
                'top_p': .9 if args.backend == 'opencode' else None,
                'max_tokens': 4096 if args.backend == 'opencode' else None,
                'baseline': baseline.payload() if baseline is not None else None, 'device': args.device,
                'initialization_mode': 'explicit' if baseline is not None else 'generic_cnn',
                'initialization_candidates': initial if isinstance(initial, list) else None,
                'architecture_provenance': model_provenance(baseline.model, 'research/components/model.py' in baseline.code) if baseline is not None else None,
                'augmentation_provenance': policy_provenance(baseline.augmentation) if baseline is not None else None,
                'versions': {name: version(name) for name in ('numpy', 'scipy', 'torch')},
            }, indent=2))
            search_started = time.monotonic()
            selected, evaluation = search.run(initial)
            search_seconds = time.monotonic() - search_started
            (output / 'selection-training.json').write_text(json.dumps({
                'pipeline': selected.payload(), 'validation': evaluation.score,
                'epochs': harness._evaluations[harness._key(selected)],
                'learning_rates': harness._lr_schedules[harness._key(selected)],
                'selection_task_fingerprint': task.fingerprint,
                'data_identity': task.identity_metadata(),
                'fold_selection': harness.selection_evidence(selected) if dataset_loso else None,
            }, indent=2))
            metadata = json.loads((output / 'run.json').read_text())
            initialization = json.loads((output / 'initialization.json').read_text())
            metadata['baseline'] = initialization['selected_pipeline']
            chosen = Pipeline.from_dict(metadata['baseline'])
            metadata['architecture_provenance'] = model_provenance(chosen.model, 'research/components/model.py' in chosen.code)
            metadata['augmentation_provenance'] = policy_provenance(chosen.augmentation)
            metadata['initialization_selected_id'] = initialization['selected_id']
            (output / 'run.json').write_text(json.dumps(metadata, indent=2))
            report = {'seed': seed, 'subject': subject, 'target': task.target,
                      'validation_subjects': ({person: sorted({r.subject for r in fold.task.validation})
                          for person, fold in zip(harness.subjects, harness.folds)} if dataset_loso
                          else sorted({r.subject for r in task.validation})), 'validation_macro_f1': evaluation.score,
                      'evaluations': search.evaluated, 'search_seconds': search_seconds,
                      'fold_count': len(tasks) if dataset_loso else 1}
            if args.final_test:
                if dataset_loso:
                    report.update(harness.finalize(selected, args.subject))
                elif task.protocol == 'fixed-validation':
                    folds = []
                    seen_folds = set()
                    test_people = test_subjects
                    for person in test_people:
                        fold_task = make_task(args.manifest, 'loso', args.target, person, args.split_seed, snapshot=dataset_snapshot)
                        if fold_task.fingerprint in seen_folds:
                            continue
                        seen_folds.add(fold_task.fingerprint)
                        fold_output = output / 'final-loso' / f'subject-{quote(person, safe="")}'
                        final = HARHarness(fold_task, seed, fold_output, args.device, generator)
                        final._evaluations[final._key(selected)] = list(harness._evaluations[harness._key(selected)])
                        final._lr_schedules[final._key(selected)] = list(harness._lr_schedules[harness._key(selected)])
                        (fold_output / 'splits.json').write_text(json.dumps(fold_task.split_manifest(), indent=2))
                        folds.append({'subject': person, 'test_subjects': sorted({r.subject for r in fold_task.test}), **final.finalize(selected)})
                    report.update(**loso_test_statistics(folds),
                                  final_loso=folds, evaluation_scope='shared-pipeline exploratory LOSO; people participated in search')
                else:
                    report.update(harness.finalize(selected))
            summaries.append(report)
            print(json.dumps(report), flush=True)
    summary = {'runs': summaries}
    if args.final_test:
        per_seed = [float(np.mean([r['test_macro_f1'] for r in summaries if r['seed'] == seed])) for seed in args.seeds]
        summary.update(test_mean=float(np.mean(per_seed)),
                       test_std=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                       seed_scores=dict(zip(map(str, args.seeds), per_seed)))
    if args.protocol == 'lodo':
        summary['targets'] = {}
        for target in targets:
            runs = [r for r in summaries if r['target'] == target]
            entry = {'runs': runs}
            if args.final_test:
                scores = [r['test_macro_f1'] for r in runs]
                entry.update(test_mean=float(np.mean(scores)),
                             test_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
            summary['targets'][target] = entry
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
