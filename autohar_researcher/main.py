from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .agents import Search, SingleAgentSearch
from .baselines import initial_candidates
from .datasets import load_manifest, make_task
from .harness import HARHarness
from .planner import CodexBackend, LLMAgents, OpenCodeBackend
from .synthetic import MCPGenerator
from .types import Pipeline, SEEDS
from research.modules.architectures.provenance import model_provenance
from research.augmentations import policy_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description='AutoResearcHAR source-only LOSO/LODO search')
    parser.add_argument('command', choices=['splits', 'search'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--protocol', choices=['loso', 'lodo'], required=True)
    parser.add_argument('--target', default='uci_har', help='Dataset target (default: uci_har)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--subject', help='Held-out LOSO subject')
    group.add_argument('--all-subjects', action='store_true', help='Run every LOSO person fold (default unless --subject is given)')
    parser.add_argument('--split-seed', type=int, default=13, help='Fixed across search seeds')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--method', choices=['multi-agent', 'single-agent', 'fixed'], default='multi-agent')
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
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Search seeds must be unique')
    _, records = load_manifest(args.manifest)
    subjects = sorted({r.subject for r in records if r.dataset == args.target}) if args.protocol == 'loso' and args.subject is None else [args.subject]
    if not subjects:
        parser.error('No people found for the requested dataset')
    tasks = [make_task(args.manifest, args.protocol, args.target, subject, args.split_seed) for subject in subjects]
    if args.command == 'splits':
        print(json.dumps([task.split_manifest() for task in tasks], indent=2))
        return
    if args.method != 'fixed' and not args.model:
        parser.error('Search requires explicit --model (or AUTOHAR_MODEL); use opencode for local models, codex for remote models')
    import torch
    torch.set_num_threads(args.threads)
    baseline = Pipeline.from_dict(json.loads(args.baseline.read_text())) if args.baseline else None
    backend = (OpenCodeBackend(args.model, args.opencode_bin) if args.backend == 'opencode' else
               CodexBackend(args.model, args.codex_bin))
    agents = LLMAgents(backend, code_edits=args.proposal_mode == 'code')
    generator = MCPGenerator(json.loads(args.mcp_config.read_text())) if args.mcp_config else None
    summaries = []
    for seed in args.seeds:
        for subject, task in zip(subjects, tasks):
            output = args.output / f'seed-{seed}' / (f'subject-{quote(str(subject), safe="")}' if subject is not None else args.target)
            harness = HARHarness(task, seed, output, args.device, generator)
            harness.candidate_workers = args.candidate_workers
            harness.worker_threads = args.threads
            initial = baseline if baseline is not None else initial_candidates(task)
            search_class = SingleAgentSearch if args.method == 'single-agent' else Search
            search = search_class(harness, agents, output, args.rounds, 1 if args.method == 'fixed' else args.budget,
                                  exploration=args.proposal_mode == 'directed' and args.method == 'multi-agent')
            (output / 'splits.json').write_text(json.dumps(task.split_manifest(), indent=2))
            (output / 'run.json').write_text(json.dumps({
                'seed': seed, 'split_seed': args.split_seed, 'rounds': args.rounds,
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
            selected, evaluation = search.run(initial)
            (output / 'selection-training.json').write_text(json.dumps({
                'pipeline': selected.payload(), 'validation': evaluation.score,
                'epochs': harness._evaluations[harness._key(selected)],
                'learning_rates': harness._lr_schedules[harness._key(selected)],
                'selection_task_fingerprint': task.fingerprint,
            }, indent=2))
            metadata = json.loads((output / 'run.json').read_text())
            initialization = json.loads((output / 'initialization.json').read_text())
            metadata['baseline'] = initialization['selected_pipeline']
            chosen = Pipeline.from_dict(metadata['baseline'])
            metadata['architecture_provenance'] = model_provenance(chosen.model, 'research/components/model.py' in chosen.code)
            metadata['augmentation_provenance'] = policy_provenance(chosen.augmentation)
            metadata['initialization_selected_id'] = initialization['selected_id']
            (output / 'run.json').write_text(json.dumps(metadata, indent=2))
            report = {'seed': seed, 'subject': subject,
                      'validation_subjects': sorted({r.subject for r in task.validation}), 'validation_macro_f1': evaluation.score,
                      'evaluations': search.evaluated}
            if args.final_test:
                report.update(harness.finalize(selected))
            summaries.append(report)
            print(json.dumps(report), flush=True)
    summary = {'runs': summaries}
    if args.final_test:
        per_seed = [float(np.mean([r['test_macro_f1'] for r in summaries if r['seed'] == seed])) for seed in args.seeds]
        summary.update(test_mean=float(np.mean(per_seed)),
                       test_std=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                       seed_scores=dict(zip(map(str, args.seeds), per_seed)))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
