"""Treina a CNN do baseline com a melhor feature da Parte 2 (chroma) + data
augmentation no audio bruto, SO no treino.

Uso:  python -m experiments.exp_augment [--feature chroma] [--n-aug 1|2] [--seed 42]

Treino = janelas originais + n_aug versoes aumentadas de cada janela.
Teste  = janelas originais, sem augmentation (mesmo split da Parte 1).
Ao final avalia tambem no teste degradado (ver eval_degraded.py).

--detune adiciona um pitch shift pequeno (-40..+40 cents) MANTENDO o rotulo,
simulando violao desafinado; as demais augmentations ficam identicas as da
execucao sem --detune (mesmas sementes).

Pitch shift com troca de rotulo NAO foi implementado: com as classes do dataset
(Am, Bb, Bdim, C, Dm, Em, F, G), +-1 semitom nunca cai em outra classe existente,
entao todas as amostras transpostas seriam descartadas.
"""
import argparse
import os
import time

import numpy as np

from experiments import augment, common
from experiments.augmented_data import build
from experiments.common import tm
from experiments.eval_degraded import evaluate_degraded
from experiments.exp_features import load_features
from experiments.features import EXTRACTORS

MAX_N_AUG = 2  # o cache guarda 2 versoes; --n-aug 1 usa so a primeira (k=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', default='chroma', choices=sorted(set(EXTRACTORS) - {'mfcc+chroma'}))
    parser.add_argument('--n-aug', type=int, default=2, choices=[1, 2])
    parser.add_argument('--scaler', default='baseline', choices=['baseline', 'per_feature'])
    parser.add_argument('--detune', action='store_true',
                        help='inclui pitch shift de -40..+40 cents mantendo o rotulo')
    parser.add_argument('--seed', type=int, default=common.DEFAULT_SEED)
    parser.add_argument('--name')
    args = parser.parse_args()

    name = args.name or f'aug{args.n_aug}_{args.feature}' + ('_detune' if args.detune else '')
    out_dir = os.path.join(common.RESULTS_DIR, name)
    common.set_seeds(args.seed)

    train_files, test_files, labels = common.load_split()
    X_clean, y_clean, X_test, y_test_raw = load_features(args.feature, train_files, test_files)
    X_aug, y_aug, k = build(train_files, args.feature,
                          'train_aug_detune' if args.detune else 'train_aug', n_versions=MAX_N_AUG, seed=args.seed)
    keep = k < args.n_aug
    X_train = np.concatenate([X_clean, X_aug[keep]])
    y_train_raw = np.concatenate([y_clean, y_aug[keep]])
    print(f'  treino: {len(X_clean)} originais + {int(keep.sum())} aumentadas = {len(X_train)}; '
          f'teste: {len(X_test)} (sem augmentation)')

    encoder, y_train, y_test = common.encode_labels(labels, y_train_raw, y_test_raw)
    if args.scaler == 'baseline':
        X_train_f, X_test_f, scaler = common.fit_scaler_like_baseline(X_train, X_test)
    else:
        X_train_f, X_test_f, scaler = common.fit_scaler_per_feature(X_train, X_test)

    model = common.build_baseline_cnn(X_train_f.shape[1:], len(labels))
    t0 = time.time()
    history = model.fit(
        X_train_f, y_train, epochs=200, batch_size=32,
        validation_data=(X_test_f, y_test),
        callbacks=common.baseline_callbacks(), verbose=2,
    )
    train_seconds = time.time() - t0

    common.save_artifacts(model, encoder, scaler, out_dir)
    common.evaluate(model, X_test_f, y_test, encoder, out_dir, extra={
        'seed': args.seed,
        'features': args.feature,
        'feature_key': args.feature,
        'scaler': args.scaler,
        'augmentation': f'{args.n_aug} versao(oes)/janela: ruido branco/rosa SNR 10-30 dB, '
                        'ganho -12..+6 dB, passa-banda ~150 Hz-7 kHz, deslocamento ate 0,15 s',
        'pitch_shift': False,
        'detune_cents': list(augment.DETUNE_CENTS_RANGE) if args.detune else None,
        'model': 'CNN 1D (Conv64-Conv128-Dense128)',
        'epochs_trained': len(history.history['loss']),
        'train_seconds': round(train_seconds, 1),
        'n_train_windows': int(len(X_train_f)),
        'window_seconds': tm.WINDOW_SECONDS,
    })
    evaluate_degraded(name, model, encoder, scaler)


if __name__ == '__main__':
    main()
