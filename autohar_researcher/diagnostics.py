import numpy as np


def probe_indices(y, classes, per_class=16):
    selected = []
    for index in range(classes):
        rows = np.flatnonzero(y == index)
        if len(rows):
            selected.extend(rows[np.linspace(0, len(rows) - 1, min(per_class, len(rows)), dtype=int)])
    return np.asarray(selected, dtype=int)


def signal_summary(x, y, labels):
    counts = np.bincount(y, minlength=len(labels))
    chosen = probe_indices(y, len(labels))
    def stats(values):
        if not len(values):
            return None
        flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        return {'mean': flat.mean(0).tolist(), 'std': flat.std(0).tolist(),
                'min': flat.min(0).tolist(), 'max': flat.max(0).tolist(),
                'rms': np.sqrt(np.mean(flat ** 2, axis=0)).tolist(),
                'quantiles_05_50_95': np.quantile(flat, [.05, .5, .95], axis=0).tolist()}
    sample, targets = x[chosen], y[chosen]
    return {'window_count': len(x), 'class_counts': dict(zip(labels, counts.tolist())),
            'probe_window_count': len(chosen), 'statistics_scope': 'up to 16 evenly spaced windows per class',
            'channels': stats(sample),
            'per_class': {name: stats(sample[targets == i]) for i, name in enumerate(labels)}}


def classification_report(predictions, targets, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(matrix, (targets, predictions), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    diagonal = matrix.diagonal()
    precision = np.divide(diagonal, predicted, out=np.zeros(len(labels)), where=predicted != 0)
    recall = np.divide(diagonal, support, out=np.zeros(len(labels)), where=support != 0)
    f1 = np.divide(2 * diagonal, support + predicted, out=np.zeros(len(labels)), where=support + predicted != 0)
    errors = [(int(matrix[i, j]), labels[i], labels[j]) for i in range(len(labels))
              for j in range(len(labels)) if i != j and matrix[i, j]]
    return {'confusion_true_rows_predicted_columns': matrix.tolist(),
            'per_class': {name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                'f1': float(f1[i]), 'support': int(support[i])}
                          for i, name in enumerate(labels)},
            'top_confusions': [{'true': a, 'predicted': b, 'count': n}
                               for n, a, b in sorted(errors, reverse=True)[:10]]}
