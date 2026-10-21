"""Literature-derived initial candidates, selected only through source validation."""
from copy import deepcopy
import json
from pathlib import Path
from .types import Pipeline


def registry():
    return json.loads((Path(__file__).parents[1] / 'research/baseline_registry.json').read_text())


def dataset_candidates(dataset):
    book = registry()
    if dataset not in book['datasets']:
        raise ValueError(f'No source-backed preset for {dataset}; supply --baseline explicitly')
    profile = book['datasets'][dataset]
    candidates = []
    for index, name in enumerate(profile['models']):
        base = Pipeline.from_dict({'data': {key: profile[key] for key in
            ('window_seconds', 'overlap', 'normalization', 'balanced_sampling')},
            'model': dict(architecture=name, **profile.get('model_overrides', {}).get(name, {}))})
        if index > 0 and 'alternative_window_seconds' in profile:
            base.data['window_seconds'] = profile['alternative_window_seconds']
        variants = [('reference', base)]
        if index == 0:
            scaled = deepcopy(base)
            scaled.data['normalization'] = 'minmax'
            augmented = deepcopy(base)
            augmented.augmentation.update(deepcopy(profile['augmentation']))
            variants += [('minmax', scaled), ('augmented', augmented)]
        for variant, pipeline in variants:
            candidates.append({'id': f'{dataset}/{name}/{variant}', 'pipeline': pipeline.payload(),
                'evidence': {'dataset': dataset, 'variant': variant,
                    'sources': {key: book['sources'][key] for key in profile['sources']},
                    'adaptations': profile['adaptations'], 'researched_on': book['researched_on'],
                    'architecture_source': json.loads((Path(__file__).parents[1] / 'research/vendor/ARCHITECTURES.json').read_text())[name].get('source', {})}})
    return candidates


def initial_candidates(task):
    # LODO pool derives exclusively from eligible SOURCE dataset identities.
    datasets = [task.target] if task.protocol == 'loso' else sorted({r.dataset for r in task.train})
    result = []
    for dataset in datasets:
        result.extend(dataset_candidates(dataset))
    if task.protocol == 'lodo':
        pipeline = Pipeline.from_dict({'model': {'architecture': 'mobilehart'}})
        result.append({'id': 'lodo/mobilehart/reference', 'pipeline': pipeline.payload(),
            'evidence': {'sources': {key: registry()['sources'][key] for key in ('hart', 'paper')},
                         'adaptations': ['Fixed 128-sample/six-channel source-only transfer candidate; target data are not used.']}})
    unique = {}
    for item in result:
        key = json.dumps(item['pipeline'], sort_keys=True)
        if key in unique:
            unique[key].setdefault('equivalent_candidates', []).append(item)
        else:
            unique[key] = item
    return list(unique.values())
