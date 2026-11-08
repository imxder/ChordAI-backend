"""Roda evaluate_realtime.py com um modelo de experiments/results/<exp>/.

Uso:  python -m experiments.eval_realtime_exp aug2_chroma_s42

Pastas de experimento treinadas antes da Parte 6 nao tem model_config.json;
neste caso ele e gerado a partir do metrics.json (sem limiar de confianca).
"""
import json
import os
import sys

import joblib

import evaluate_realtime
import pipeline as pl
from experiments import common
from experiments.eval_degraded import feature_of


def ensure_config(exp_dir):
    if os.path.exists(os.path.join(exp_dir, pl.CONFIG_FILENAME)):
        return
    metrics = json.load(open(os.path.join(exp_dir, 'metrics.json'), encoding='utf-8'))
    classes = joblib.load(os.path.join(exp_dir, 'encoder.joblib')).classes_
    cfg = pl.make_model_config(feature_of(metrics), metrics.get('scaler', 'baseline'), metrics.get('arch', 'baseline'),
                               classes, metrics.get('with_none', False), None,
                               extra={'generated_from': 'metrics.json (eval_realtime_exp)'})
    with open(os.path.join(exp_dir, pl.CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def main():
    exp_dir = os.path.join(common.RESULTS_DIR, sys.argv[1])
    ensure_config(exp_dir)
    evaluate_realtime.main(exp_dir)


if __name__ == '__main__':
    main()
