"""Avalia um modelo JA treinado (ex.: candidates/candidate_91_81) com o mesmo
split e a mesma funcao evaluate(), so leitura -- nada em candidates/ e alterado.

Obs.: o candidate_91_81 foi treinado com outro split (deu 97,24% aqui, contra
~91,8% documentado), entao NAO e comparavel com os experimentos.

Uso:  python -m experiments.eval_saved_model candidates/candidate_91_81 [--name ref_candidate_91_81]
"""
import argparse
import os

import joblib
import tensorflow as tf

from experiments import common
from experiments.data import build


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
    X_test, y_test_raw, _, _ = build(test_files, 'mfcc', 'clean')
    X_test_f = common.apply_scaler(scaler, X_test, 'baseline')

    common.evaluate(model, X_test_f, y_test_raw, encoder, out_dir, extra={
        'source_model_dir': args.model_dir,
        'note': 'modelo pre-treinado, apenas avaliado',
    })


if __name__ == '__main__':
    main()
