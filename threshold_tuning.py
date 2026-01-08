import numpy as np

def tune_threshold(results):
    """
    results = list of (score, true_label)
    """
    best_acc = 0
    best_thresh = 0

    scores = [r[0] for r in results]
    labels = [1 if r[1] == "consistent" else 0 for r in results]

    for t in range(min(scores), max(scores) + 1):
        preds = [1 if s >= t else 0 for s in scores]
        acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)

        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    return best_thresh, best_acc
