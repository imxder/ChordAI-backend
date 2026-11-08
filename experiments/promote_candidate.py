"""Monta o candidato a producao (Parte 6) a partir de um experimento ja treinado.

Usa os pesos da seed 42 de feat_chroma_scalerfeat_gapbn_none -- seed escolhida
ANTES de olhar os resultados (nao e a "melhor seed" no teste) -- e o limiar de
confianca calibrado na VALIDACAO para esse mesmo modelo, com o criterio da Parte 5
(pipeline.choose_threshold), exatamente como train_model.py faz ao treinar.
O limiar da Parte 5 (0,45) era a media das 3 seeds; fica registrado no metrics.json.

Gera candidates/candidate_chroma_gapbn_none/ com model.h5, encoder.joblib,
scaler.joblib, model_config.json e metrics.json.

Uso:  python -m experiments.promote_candidate
"""
import json
import os
import shutil

import joblib
import numpy as np
import tensorflow as tf

import pipeline as pl
from experiments import common
from experiments.threshold import load_eval_set

SOURCE_RUN = 'feat_chroma_scalerfeat_gapbn_none_s42'
OUT_DIR = os.path.join(common.ROOT_DIR, 'candidates', 'candidate_chroma_gapbn_none')


def calibrate_on_validation(src, scaler_kind):
    """Limiar do proprio modelo na validacao (janelas de silencio fora), como train_model.py."""
    train_files, _, _ = common.load_split()
    _, val_files = pl.split_train_val(train_files)
    s = load_eval_set('val', val_files)
    model = tf.keras.models.load_model(os.path.join(src, 'model.h5'))
    scaler = joblib.load(os.path.join(src, 'scaler.joblib'))
    classes = np.array(joblib.load(os.path.join(src, 'encoder.joblib')).classes_)
    probs_c = model.predict(pl.apply_scaler(scaler, s['X'], scaler_kind), verbose=0)
    probs_n = model.predict(pl.apply_scaler(scaler, s['X_neg'], scaler_kind), verbose=0)
    rows = pl.threshold_curve(probs_c, s['y'], probs_n, classes)
    threshold = pl.choose_threshold(rows)
    chosen = next(r for r in rows if r['threshold'] == threshold)
    return threshold, {k: float(v) for k, v in chosen.items()}


def main():
    src = os.path.join(common.RESULTS_DIR, SOURCE_RUN)
    metrics = json.load(open(os.path.join(src, 'metrics.json'), encoding='utf-8'))
    suggestion = json.load(open(os.path.join(common.RESULTS_DIR, 'threshold', 'suggestion.json'), encoding='utf-8'))
    encoder = joblib.load(os.path.join(src, 'encoder.joblib'))
    threshold, val_row = calibrate_on_validation(src, metrics['scaler'])
    print(f'limiar calibrado na validacao (seed {metrics["seed"]}): {threshold}  {val_row}')

    config = pl.make_model_config(
        feature=metrics['feature_key'], scaler=metrics['scaler'], arch=metrics['arch'],
        classes=encoder.classes_, with_none=metrics['with_none'],
        confidence_threshold=threshold,
        extra={'seed': metrics['seed'], 'source_experiment': f'experiments/results/{SOURCE_RUN}',
               'threshold_source': 'calibrado na validacao para este modelo (seed 42), criterio da Parte 5; '
                                   'mesmo valor que train_model.py calibra sozinho',
               'threshold_parte5_mean_3_seeds': suggestion['limiar']},
    )
    pl.validate_config(config)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ('model.h5', 'encoder.joblib', 'scaler.joblib'):
        shutil.copy2(os.path.join(src, name), os.path.join(OUT_DIR, name))
    with open(os.path.join(OUT_DIR, pl.CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    out_metrics = {
        'source_experiment': SOURCE_RUN,
        'seed': metrics['seed'],
        'classes': metrics['classes'],
        'n_params': metrics['n_params'],
        'accuracy_chords': metrics['accuracy_chords'],
        'accuracy_degraded': metrics['accuracy_degraded'],
        'macro_f1': metrics['macro_f1'],
        'f1_per_class': metrics['f1_per_class'],
        'none_recall': metrics['none_recall'],
        'none_recall_by_kind': metrics['none_recall_by_kind'],
        'chords_predicted_as_none': metrics['chords_predicted_as_none'],
        'confidence_threshold': threshold,
        'threshold_validation_seed42': val_row,
        'threshold_parte5_mean_3_seeds': suggestion['limiar'],
        'threshold_parte5_validation_mean_3_seeds': suggestion['validacao'],
        'threshold_parte5_test_mean_3_seeds': suggestion['teste'],
        'latency': metrics.get('latency'),
        'note': 'acuracias sem limiar (argmax); com o limiar de producao, ver experiments/results/parte6/equivalence.json',
    }
    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)
    print(f'candidato salvo em {OUT_DIR}')
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
