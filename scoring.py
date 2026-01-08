def score_label(label):
    return {
        "Supported": 2,
        "Constrained": -1,
        "Unconstrained": -1,
        "Contradicted": -3
    }.get(label, 0)
