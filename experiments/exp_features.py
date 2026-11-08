"""Treina a MESMA CNN 1D do baseline trocando so a representacao de entrada.

Uso:  python -m experiments.exp_features --feature {mfcc,chroma,cqt,mfcc+chroma}
          [--scaler {baseline,per_feature}] [--seed 42]

--scaler baseline    (padrao) StandardScaler identico ao de train_model.py
--scaler per_feature StandardScaler por linha de feature (coeficiente/bin)
"""
import argparse
import os
import time

import numpy as np

from experiments import common
from experiments.common import tm
from experiments.features import CACHE_NAMES, EXTRACTORS


def load_features(feature, train_files, test_files):
    if feature == 'mfcc+chroma':
        # mesmas janelas, na mesma ordem: basta concatenar os caches de cada parte
        parts = [load_features(f, train_files, test_files) for f in ('mfcc', 'chroma')]
        (Xm_tr, y_tr, Xm_te, y_te), (Xc_tr, y_tr2, Xc_te, y_te2) = parts
        assert np.array_equal(y_tr, y_tr2) and np.array_equal(y_te, y_te2)
        return (np.concatenate([Xm_tr, Xc_tr], axis=1), y_tr,
                np.concatenate([Xm_te, Xc_te], axis=1), y_te)

    name = CACHE_NAMES[feature]
    fn = None if feature == 'mfcc' else EXTRACTORS[feature]
    X_tr, y_tr = common.build_windowed_dataset_cached(train_files, f'{name}_train', feature_fn=fn)
    X_te, y_te = common.build_windowed_dataset_cached(test_files, f'{name}_test', feature_fn=fn)
    return X_tr, y_tr, X_te, y_te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', required=True, choices=sorted(EXTRACTORS))
    parser.add_argument('--scaler', default='baseline', choices=['baseline', 'per_feature'])
    parser.add_argument('--seed', type=int, default=common.DEFAULT_SEED)
    parser.add_argument('--name')
    args = parser.parse_args()

    name = args.name or ('feat_' + args.feature.replace('+', '_')
                         + ('' if args.scaler == 'baseline' else '_scalerfeat'))
    out_dir = os.path.join(common.RESULTS_DIR, name)
    common.set_seeds(args.seed)

    train_files, test_files, labels = common.load_split()
    print(f'Extraindo "{args.feature}" das janelas de {tm.WINDOW_SECONDS}s...')
    X_train, y_train_raw, X_test, y_test_raw = load_features(args.feature, train_files, test_files)
    print(f'  janelas: treino={len(X_train)} teste={len(X_test)} shape={X_train.shape[1:]}')

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
        'input_shape': list(X_train_f.shape[1:]),
        'scaler': args.scaler,
        'model': 'CNN 1D (Conv64-Conv128-Dense128)',
        'epochs_trained': len(history.history['loss']),
        'train_seconds': round(train_seconds, 1),
        'n_train_windows': int(len(X_train_f)),
    })


if __name__ == '__main__':
    main()
