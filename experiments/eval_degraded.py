"""Avalia um experimento ja treinado no TESTE DEGRADADO (mesmos arquivos e
janelas do teste, com ruido rosa a 15 dB + passa-banda 200 Hz-5 kHz) e grava
'accuracy_degraded' / 'macro_f1_degraded' no metrics.json do experimento.
O relatorio completo fica em results/<exp>/degraded/.

Uso:  python -m experiments.eval_degraded baseline_mfcc feat_chroma ...
"""
import json
import os
import sys

import joblib
import tensorflow as tf

from experiments import common
from experiments.augmented_data import load_degraded_test
from experiments.features import EXTRACTORS


def feature_of(metrics):
    key = metrics.get('feature_key') or metrics.get('features')
    return key if key in EXTRACTORS else 'mfcc'  # baseline_mfcc guarda uma descricao textual


def evaluate_degraded(exp_name, model=None, encoder=None, scaler=None):
    exp_dir = os.path.join(common.RESULTS_DIR, exp_name)
    metrics_path = os.path.join(exp_dir, 'metrics.json')
    with open(metrics_path, encoding='utf-8') as f:
        metrics = json.load(f)

    model = model or tf.keras.models.load_model(os.path.join(exp_dir, 'model.h5'))
    encoder = encoder or joblib.load(os.path.join(exp_dir, 'encoder.joblib'))
    scaler = scaler or joblib.load(os.path.join(exp_dir, 'scaler.joblib'))

    X, y = load_degraded_test(feature_of(metrics))
    X_f = common.apply_scaler(scaler, X, metrics.get('scaler', 'baseline'))
    deg = common.evaluate(model, X_f, y, encoder, os.path.join(exp_dir, 'degraded'),
                          extra={'condition': 'teste degradado: ruido rosa 15 dB + passa-banda 200-5000 Hz'})

    metrics['accuracy_degraded'] = deg['accuracy']
    metrics['macro_f1_degraded'] = deg['macro_f1']
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return deg


def main():
    for name in sys.argv[1:]:
        evaluate_degraded(name)


if __name__ == '__main__':
    main()
