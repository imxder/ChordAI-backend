"""Treina a MESMA CNN 1D do baseline variando a representacao de entrada, o
scaler e (opcionalmente) o data augmentation.

Uso:
  python -m experiments.exp_features --feature {mfcc,chroma,cqt,mfcc+chroma}
      [--scaler {baseline,per_feature}] [--n-aug {0,1,2}] [--detune] [--seed 42]

--scaler baseline     StandardScaler identico ao de train_model.py (normaliza por frame)
--scaler per_feature  StandardScaler por linha de feature (coeficiente/bin)
--n-aug N             N versoes aumentadas por janela de treino (ver augment.py)
--detune              inclui pitch shift de -40..+40 cents mantendo o rotulo

Validacao: 15% dos arquivos de treino (ver data.py), usada nos callbacks.
O teste so e usado no evaluate() final. Resultado em
experiments/results/<grupo>_s<seed>/, onde <grupo> descreve a configuracao.
"""
import argparse
import os
import time

from experiments import augment, common
from experiments.common import tm
from experiments.data import VAL_SIZE, load_dataset
from experiments.eval_degraded import evaluate_degraded
from experiments.features import EXTRACTORS


def group_name(feature, scaler='baseline', n_aug=0, detune=False):
    name = ('feat_' if not n_aug else f'aug{n_aug}_') + feature.replace('+', '_')
    if scaler == 'per_feature':
        name += '_scalerfeat'
    if detune:
        name += '_detune'
    return name


def run(feature, scaler='baseline', n_aug=0, detune=False, seed=common.DEFAULT_SEED):
    assert not detune or n_aug, '--detune exige --n-aug >= 1'
    group = group_name(feature, scaler, n_aug, detune)
    name = f'{group}_s{seed}'
    out_dir = os.path.join(common.RESULTS_DIR, name)
    print(f'=== {name} ===', flush=True)
    common.set_seeds(seed)

    d = load_dataset(feature, n_aug=n_aug, detune=detune, seed=seed)
    encoder, y_train, y_val, y_test = common.encode_labels(d['labels'], d['y_train'], d['y_val'], d['y_test'])
    X_train_f, scaler_obj = common.fit_scaler(d['X_train'], scaler)
    X_val_f = common.apply_scaler(scaler_obj, d['X_val'], scaler)
    X_test_f = common.apply_scaler(scaler_obj, d['X_test'], scaler)

    model = common.build_baseline_cnn(X_train_f.shape[1:], len(d['labels']))
    t0 = time.time()
    history = model.fit(
        X_train_f, y_train, epochs=200, batch_size=32,
        validation_data=(X_val_f, y_val),          # validacao, NAO o teste
        callbacks=common.baseline_callbacks(), verbose=2,
    )
    train_seconds = time.time() - t0

    common.save_artifacts(model, encoder, scaler_obj, out_dir)
    common.evaluate(model, X_test_f, y_test, encoder, out_dir, extra={
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
        'model': 'CNN 1D (Conv64-Conv128-Dense128)',
        'epochs_trained': len(history.history['loss']),
        'best_val_loss': float(min(history.history['val_loss'])),
        'train_seconds': round(train_seconds, 1),
        'window_seconds': tm.WINDOW_SECONDS,
    })
    evaluate_degraded(name, model, encoder, scaler_obj)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', required=True, choices=sorted(EXTRACTORS))
    parser.add_argument('--scaler', default='baseline', choices=['baseline', 'per_feature'])
    parser.add_argument('--n-aug', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--detune', action='store_true')
    parser.add_argument('--seed', type=int, default=common.DEFAULT_SEED)
    args = parser.parse_args()
    run(args.feature, args.scaler, args.n_aug, args.detune, args.seed)


if __name__ == '__main__':
    main()
