"""Treina a MESMA CNN 1D do baseline variando a representacao de entrada, o
scaler e (opcionalmente) o data augmentation.

Uso:
  python -m experiments.exp_features --feature {mfcc,chroma,cqt,mfcc+chroma}
      [--scaler {baseline,per_feature}] [--n-aug {0,1,2}] [--detune]
      [--arch {baseline,gap_bn,cnn2d}] [--seed 42]

--scaler baseline     StandardScaler identico ao de train_model.py (normaliza por frame)
--scaler per_feature  StandardScaler por linha de feature (coeficiente/bin)
--n-aug N             N versoes aumentadas por janela de treino (ver augment.py)
--detune              inclui pitch shift de -40..+40 cents mantendo o rotulo
--arch                baseline = CNN 1D com Flatten (train_model.py);
                      gap_bn = CNN 1D com BatchNorm + GlobalAveragePooling1D;
                      cnn2d = CNN 2D pequena (ver common.py)

Validacao: 15% dos arquivos de treino (ver data.py), usada nos callbacks.
O teste so e usado no evaluate() final. Resultado em
experiments/results/<grupo>_s<seed>/, onde <grupo> descreve a configuracao.
"""
import argparse
import os
import time

import numpy as np

from experiments import augment, common
import pipeline as pl
from experiments.data import VAL_SIZE, load_dataset
from experiments.eval_degraded import evaluate_degraded
from experiments.features import EXTRACTORS


def group_name(feature, scaler='baseline', n_aug=0, detune=False, arch='baseline', with_none=False):
    name = ('feat_' if not n_aug else f'aug{n_aug}_') + feature.replace('+', '_')
    if scaler == 'per_feature':
        name += '_scalerfeat'
    if detune:
        name += '_detune'
    if arch != 'baseline':
        name += '_' + arch.replace('_', '')
    if with_none:
        name += '_none'
    return name


def none_metrics(model, X_test_f, y_test_raw, none_kinds, encoder, out_dir):
    """Metricas extras de um modelo com classe _none, separando as janelas de
    acorde (as MESMAS 1545 dos outros experimentos) das negativas."""
    from experiments.negatives import NONE_LABEL
    pred = encoder.inverse_transform(np.argmax(model.predict(X_test_f, verbose=0), axis=1))
    y = np.asarray(y_test_raw)
    is_none = y == NONE_LABEL
    chords_pred, chords_true = pred[~is_none], y[~is_none]
    neg_pred = pred[is_none]
    return {
        'accuracy_chords': float(np.mean(chords_pred == chords_true)),
        'chords_predicted_as_none': float(np.mean(chords_pred == NONE_LABEL)),
        'none_recall': float(np.mean(neg_pred == NONE_LABEL)),
        'none_recall_by_kind': {k: float(np.mean(neg_pred[none_kinds == k] == NONE_LABEL))
                                for k in sorted(set(none_kinds))},
        'n_test_chord_windows': int((~is_none).sum()),
        'n_test_none_windows': int(is_none.sum()),
    }


def run(feature, scaler='baseline', n_aug=0, detune=False, seed=common.DEFAULT_SEED, arch='baseline',
        with_none=False):
    assert not detune or n_aug, '--detune exige --n-aug >= 1'
    group = group_name(feature, scaler, n_aug, detune, arch, with_none)
    name = f'{group}_s{seed}'
    out_dir = os.path.join(common.RESULTS_DIR, name)
    print(f'=== {name} ===', flush=True)
    common.set_seeds(seed)

    d = load_dataset(feature, n_aug=n_aug, detune=detune, seed=seed, with_none=with_none)
    encoder, y_train, y_val, y_test = common.encode_labels(d['labels'], d['y_train'], d['y_val'], d['y_test'])
    X_train_f, scaler_obj = common.fit_scaler(d['X_train'], scaler)
    X_val_f = common.apply_scaler(scaler_obj, d['X_val'], scaler)
    X_test_f = common.apply_scaler(scaler_obj, d['X_test'], scaler)

    model = common.build_model(arch, X_train_f.shape[1:], len(d['labels']))
    t0 = time.time()
    history = model.fit(
        X_train_f, y_train, epochs=200, batch_size=32,
        validation_data=(X_val_f, y_val),          # validacao, NAO o teste
        callbacks=common.baseline_callbacks(), verbose=2,
    )
    train_seconds = time.time() - t0

    config = pl.make_model_config(feature, scaler, arch, encoder.classes_, with_none, confidence_threshold=None,
                                  extra={'group': group, 'seed': seed, 'trained_by': 'experiments/exp_features.py'})
    common.save_artifacts(model, encoder, scaler_obj, out_dir, config=config)
    extra_none = {}
    if with_none:
        extra_none = none_metrics(model, X_test_f, d['y_test'], d['test_none_kinds'], encoder, out_dir)
        extra_none['none_counts'] = d['none_counts']
    common.evaluate(model, X_test_f, y_test, encoder, out_dir, extra={
        'with_none': with_none,
        **extra_none,
        'group': group,
        'seed': seed,
        'features': feature,
        'feature_key': feature,
        'input_shape': list(X_train_f.shape[1:]),
        'scaler': scaler,
        'n_aug': n_aug,
        'augmentation': (f'{n_aug} versao(oes)/janela: ruido branco/rosa SNR 10-30 dB, ganho -12..+6 dB, '
                         'passa-banda ~150 Hz-7 kHz, deslocamento ate 0,15 s') if n_aug else None,
        'detune_cents': list(augment.DETUNE_CENTS_RANGE) if detune else None,
        'validation': f'{int(VAL_SIZE * 100)}% dos arquivos de treino (estratificado, random_state=42)',
        'n_train_files': d['n_train_files'],
        'n_val_files': d['n_val_files'],
        'n_train_windows': int(len(X_train_f)),
        'n_val_windows': int(len(X_val_f)),
        'arch': arch,
        'model': {'baseline': 'CNN 1D (Conv64-Conv128-Flatten-Dense128)',
                  'gap_bn': 'CNN 1D (Conv64-BN-Conv128-BN-GAP-Dense128)',
                  'cnn2d': 'CNN 2D (Conv32-BN-Conv64-BN-Conv128-BN-GAP)'}[arch],
        'epochs_trained': len(history.history['loss']),
        'best_val_loss': float(min(history.history['val_loss'])),
        'train_seconds': round(train_seconds, 1),
        'window_seconds': pl.WINDOW_SECONDS,
    })
    evaluate_degraded(name, model, encoder, scaler_obj)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', required=True, choices=sorted(EXTRACTORS))
    parser.add_argument('--scaler', default='baseline', choices=['baseline', 'per_feature'])
    parser.add_argument('--n-aug', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--detune', action='store_true')
    parser.add_argument('--arch', default='baseline', choices=sorted(common.ARCHITECTURES))
    parser.add_argument('--with-none', action='store_true', help='acrescenta a classe _none (Parte 5)')
    parser.add_argument('--seed', type=int, default=common.DEFAULT_SEED)
    args = parser.parse_args()
    run(args.feature, args.scaler, args.n_aug, args.detune, args.seed, args.arch, args.with_none)


if __name__ == '__main__':
    main()
