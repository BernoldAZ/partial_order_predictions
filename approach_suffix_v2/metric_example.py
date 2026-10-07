####

# Running example comparing the suffix metrics (GES, bucket-order K^(1/2) and
# DL similarity) on one ground-truth suffix against a set of possible model
# predictions.
#
# Ground truth: A->{B,C,D}->E->{F,G}
#
# Notation: '->' separates blocks, '{...}' groups concurrent activities.
# To add or remove a case, edit CASES; to add or remove a metric, edit METRICS.
#
# Usage: python <path>/metric_example.py   (from any directory; needs networkx for GES)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find models_v2 from any working directory
from models_v2.partial_order_metrics import bucket_order_similarity, ges_compute_sample

TRUTH = "A->{B,C,D}->E->{F,G}"

# (group, description, prediction)
CASES = [
    ("Sanity",      "Perfect",                "A->{B,C,D}->E->{F,G}"),
    ("Sanity",      "Within-block shuffle",   "A->{D,B,C}->E->{G,F}"),
    ("Concurrency", "All sequential",         "A->B->C->D->E->F->G"),
    ("Concurrency", "Single block",           "{A,B,C,D,E,F,G}"),
    ("Concurrency", "Block split",            "A->{B,C}->D->E->{F,G}"),
    ("Concurrency", "Block merge",            "A->{B,C,D,E}->{F,G}"),
    ("Order",       "Blocks swapped",         "A->E->{B,C,D}->{F,G}"),
    ("Order",       "Full reversal",          "{F,G}->E->{B,C,D}->A"),
    ("Labels",      "One label substituted",  "A->{B,C,Y}->E->{F,G}"),
    ("Labels",      "All labels wrong",       "H->{I,J,K}->L->{M,N}"),
    ("Insertion",   "Extra X inside block",   "A->{B,C,D,X}->E->{F,G}"),
    ("Insertion",   "Extra X as own block",   "A->{B,C,D}->X->E->{F,G}"),
    ("Insertion",   "Repeated E at end",      "A->{B,C,D}->E->{F,G}->E"),
    ("Deletion",    "Missing D (in block)",   "A->{B,C}->E->{F,G}"),
    ("Deletion",    "Missing E (sequential)", "A->{B,C,D}->{F,G}"),
    ("Deletion",    "Early stop",             "A->{B,C,D}"),
    ("Deletion",    "Empty prediction",       ""),
    ("Mixed",       "Sequential + missing D", "A->B->C->E->{F,G}"),
    ("Mixed",       "Merge + substitution",   "A->{B,C,Y,E}->{F,G}"),
    ("Mixed",       "Swapped + extra X",      "A->E->{B,C,D}->X->{F,G}"),

]


# ─── Notation parser ──────────────────────────────────────────────────────────

_ACT_IDS = {}   # activity label → integer id (the metrics expect integer activities)


def parse(s):
    """'A->{B,C}->D' → (acts, nb): acts = integer ids in emitted order,
    nb = 1 for the first activity of each block, 0 for the others."""
    acts, nb = [], []
    for block in filter(None, (b.strip() for b in s.split('->'))):
        labels = [a.strip() for a in block.strip('{}').split(',')]
        for i, label in enumerate(labels):
            acts.append(_ACT_IDS.setdefault(label, len(_ACT_IDS) + 1))
            nb.append(1 if i == 0 else 0)
    return acts, nb


# ─── Metrics ──────────────────────────────────────────────────────────────────

def dl_similarity(acts_pred, acts_true):
    """Damerau-Levenshtein (optimal string alignment) similarity over the
    activities only (EOS excluded), as in _compute_metrics of the run scripts."""
    p, t = acts_pred, acts_true
    d = [[0] * (len(t) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p) + 1):
        d[i][0] = i
    for j in range(len(t) + 1):
        d[0][j] = j
    for i in range(1, len(p) + 1):
        for j in range(1, len(t) + 1):
            cost = 0 if p[i-1] == t[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
            if i > 1 and j > 1 and p[i-1] == t[j-2] and p[i-2] == t[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + cost)
    return 1.0 - d[len(p)][len(t)] / max(len(p), len(t), 1)


def _bo(pred, truth):
    return bucket_order_similarity(pred[0], pred[1], truth[0], truth[1])


# name → fn((acts_pred, nb_pred), (acts_true, nb_true))
METRICS = {
    "DL":          lambda pred, truth: dl_similarity(pred[0], truth[0]),
    "GES":         lambda pred, truth: ges_compute_sample((pred[0], pred[1], truth[0], truth[1])),
    "BO combined": lambda pred, truth: _bo(pred, truth)[2],
    "BO order":    lambda pred, truth: _bo(pred, truth)[0],
    "BO content":  lambda pred, truth: _bo(pred, truth)[1],
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    truth = parse(TRUTH)
    headers = ["Group", "Case", "Prediction"] + list(METRICS)
    rows = []
    for group, desc, pred_str in CASES:
        pred = parse(pred_str)
        scores = [f"{fn(pred, truth):.3f}" for fn in METRICS.values()]
        rows.append([group, desc, pred_str or "(empty)"] + scores)

    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" if i < 3 else f"{{:>{w}}}" for i, w in enumerate(widths))
    print(f"Ground truth: {TRUTH}\n")
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


if __name__ == '__main__':
    main()
