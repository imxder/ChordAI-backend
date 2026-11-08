"""Avalia um modelo JA treinado (ex.: candidates/candidate_91_81) com o mesmo
split e a mesma funcao evaluate(), so leitura -- nada em candidates/ e alterado.

Uso:  python -m experiments.eval_saved_model candidates/candidate_91_81 [--name ref_candidate_91_81]
"""
import argparse
import os

import joblib
import numpy as np
import tensorflow as tf

from experiments import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--name')
    args = parser.parse_args()

    model_dir = os.path.join(common.ROOT_DIR, args.model_dir)
    name = args.name or 'ref_' + os.path.basename(os.path.normpath(model_dir))
    out_dir = os.path.join(common.RESULTS_DIR, name)

    model = tf.keras.models.load_model(os.path.join(model_dir, 'chord_recognizer_cnn_model.h5'))
    encoder = joblib.load(os.path.join(model_dir, 'label_encoder_chords.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler_chords.joblib'))

    _, test_files, _ = common.load_split()
    X_test, y_test_raw = common.build_windowed_dataset_cached(test_files, 'mfcc40_test')
    X_test_f = np.swapaxes(scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape), 1, 2)

    common.evaluate(model, X_test_f, y_test_raw, encoder, out_dir, extra={
        'source_model_dir': args.model_dir,
        'note': 'modelo pre-treinado, apenas avaliado',
    })


if __name__ == '__main__':
    main()
