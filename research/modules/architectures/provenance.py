"""Machine-readable distinction between author models, ports and reconstructions."""
import copy
import hashlib
import json
from pathlib import Path
from .presets import model_defaults


def model_provenance(config, custom=False):
    if custom:
        return {'status': 'custom_candidate', 'architecture': config['architecture']}
    root = Path(__file__).parents[2] / 'vendor'
    entries = json.loads((root / 'ARCHITECTURES.json').read_text())
    result = copy.deepcopy(entries.get(config['architecture'], {'status': 'generic_experimental', 'framework': 'pytorch'}))
    result['architecture_name'] = config['architecture']
    defaults = model_defaults(config['architecture'])
    result['preset_overrides'] = {k: v for k, v in config.items() if defaults.get(k) != v}
    source = result.get('source', {})
    if 'sha256' in source:
        actual = hashlib.sha256((root / source['file']).read_bytes()).hexdigest()
        if actual != source['sha256']:
            raise ValueError('Pinned author source has changed; update provenance deliberately before running')
    return result
