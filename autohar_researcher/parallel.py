"""Spawn-isolated candidate fits with deterministic, submission-order results."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing


def fit_candidate(task, seed, output, device, generator, pipeline, threads):
    import torch
    from .harness import HARHarness
    torch.set_num_threads(threads)
    harness = HARHarness(task, seed, output, device, generator)
    result = harness.evaluate(pipeline)
    return result, harness._lr_schedules[harness._key(pipeline)], harness._validation_diagnostics[harness._key(pipeline)], harness._preprocessing_states.get(harness._key(pipeline))


def evaluate_many(harness, pipelines, workers, threads):
    if harness._finalized:
        raise RuntimeError('Search is closed after finalization')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = []
        for pipeline in pipelines:
            key = harness._key(pipeline)
            output = harness.output / 'candidate-workers' / hashlib.sha256(key.encode()).hexdigest()[:16]
            futures.append(pool.submit(fit_candidate, harness.task, harness.seed, output,
                                       harness.device, harness.generator, pipeline, threads))
        for pipeline, future in zip(pipelines, futures):
            try:
                result, schedule, diagnostics, preprocessing_state = future.result()
                key = harness._key(pipeline)
                harness._evaluations.setdefault(key, []).append(result.best_epoch)
                harness._lr_schedules[key] = schedule
                harness._validation_diagnostics[key] = diagnostics
                harness._preprocessing_states[key] = preprocessing_state
                yield result
            except Exception as exc:
                yield exc
