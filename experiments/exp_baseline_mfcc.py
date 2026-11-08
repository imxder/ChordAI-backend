"""Re-treina o baseline (40 MFCC + CNN 1D, hiperparametros de train_model.py)
com o mesmo split por arquivo e avalia com experiments.common.evaluate().

Uso (na raiz do projeto):  python -m experiments.exp_baseline_mfcc [--seed 42]

Obs.: assim como train_model.py, usa o conjunto de teste como validation_data
(EarlyStopping/ReduceLROnPlateau). Isso foi mantido de proposito para
reproduzir o baseline fielmente.
"""
import argparse
import os
import time

from experiments import common
from experiments.common import tm

NAME = 'baseline_mfcc'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=common.DEFAULT_SEED)
    parser.add_argument('--name', default=NAME)
    args = parser.parse_args()

    common.set_seeds(args.seed)
    out_dir = os.path.join(common.RESULTS_DIR, args.name)

    print('Split por arquivo (random_state=42)...')
    train_files, test_files, labels = common.load_split()
    print(f'  {len(train_files)} arquivos de treino, {len(test_files)} de teste, classes={labels}')

    print(f'Extraindo MFCC das janelas de {tm.WINDOW_SECONDS}s...')
    X_train, y_train_raw = common.build_windowed_dataset_cached(train_files, 'mfcc40_train')
    X_test, y_test_raw = common.build_windowed_dataset_cached(test_files, 'mfcc40_test')
    print(f'  janelas: treino={len(X_train)}  teste={len(X_test)}  shape={X_train.shape[1:]}')

    encoder, y_train, y_test = common.encode_labels(labels, y_train_raw, y_test_raw)
    X_train_f, X_test_f, scaler = common.fit_scaler_like_baseline(X_train, X_test)

    model = common.build_baseline_cnn(X_train_f.shape[1:], len(labels))
    model.summary()

    t0 = time.time()
    history = model.fit(
        X_train_f, y_train,
        epochs=200, batch_size=32,
        validation_data=(X_test_f, y_test),
        callbacks=common.baseline_callbacks(),
        verbose=2,
    )
    train_seconds = time.time() - t0

    common.save_artifacts(model, encoder, scaler, out_dir)
    common.evaluate(model, X_test_f, y_test, encoder, out_dir, extra={
        'seed': args.seed,
        'features': f'{tm.N_MFCC} MFCC, janela {tm.WINDOW_SECONDS}s @ {tm.SAMPLE_RATE} Hz',
        'model': 'CNN 1D (Conv64-Conv128-Dense128)',
        'epochs_trained': len(history.history['loss']),
        'train_seconds': round(train_seconds, 1),
        'n_train_windows': int(len(X_train_f)),
    })
    print(f'Artefatos em {out_dir}')


if __name__ == '__main__':
    main()
