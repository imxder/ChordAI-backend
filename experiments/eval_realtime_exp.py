"""Roda evaluate_realtime.py com um modelo de experiments/results/<exp>/.

Uso:  python -m experiments.eval_realtime_exp aug2_chroma
      python -m experiments.eval_realtime_exp baseline_mfcc

So suporta experimentos com scaler 'baseline' (o mesmo formato que
evaluate_realtime.py/app.py usam).
"""
import json
import os
import sys

from experiments import common
from experiments.eval_degraded import feature_of
from experiments.features import EXTRACTORS

import evaluate_realtime  # noqa: E402  (common ja colocou a raiz no sys.path)


def main():
    exp = sys.argv[1]
    exp_dir = os.path.join(common.RESULTS_DIR, exp)
    with open(os.path.join(exp_dir, 'metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)
    assert metrics.get('scaler', 'baseline') == 'baseline', 'evaluate_realtime so aplica o scaler do baseline'
    feature = feature_of(metrics)
    evaluate_realtime.main(
        model_path=os.path.join(exp_dir, 'model.h5'),
        encoder_path=os.path.join(exp_dir, 'encoder.joblib'),
        scaler_path=os.path.join(exp_dir, 'scaler.joblib'),
        feature_fn=EXTRACTORS[feature],
        feature_label=f'Librosa/{feature}',
    )


if __name__ == '__main__':
    main()
