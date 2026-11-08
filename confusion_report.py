"""Matriz de confusao e relatorio por classe do modelo salvo, no split de teste.
Usa o mesmo caminho de inferencia de producao (model_config.json do modelo).

Uso:  python confusion_report.py [--model-dir trained_model]
"""
import argparse

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import pipeline
import train_model
from audio_capture import RECORD_DURATION
from train_model import WINDOW_SECONDS, make_windowed_segments, split_dataset_files

assert abs(RECORD_DURATION - WINDOW_SECONDS) < 1e-6


def predict(segment, model, encoder, scaler, config):
    """Classe de maior probabilidade (argmax), sem limiar."""
    features = pipeline.features_for_model(segment, scaler, config)
    if features is None:
        return None
    prediction = model.predict(features, verbose=0)
    return encoder.inverse_transform([int(np.argmax(prediction[0]))])[0]


def main(model_dir=train_model.MODEL_DIR):
    print("Carregando modelo, encoder e scaler salvos...")
    model, encoder, scaler = train_model.load_trained_model(model_dir)
    config = encoder.model_config_

    _, test_files, _ = split_dataset_files()
    labels = [str(c) for c in encoder.classes_]

    y_true, y_pred = [], []
    for file_path, true_label in test_files:
        for segment in make_windowed_segments(file_path):
            predicted = predict(segment, model, encoder, scaler, config)
            if predicted is None:
                continue
            y_true.append(true_label)
            y_pred.append(predicted)

    print()
    print("=" * 64)
    print("MATRIZ DE CONFUSAO (linhas = esperado, colunas = previsto)")
    print("=" * 64)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("        " + "".join(f"{l:>6}" for l in labels))
    for label, row in zip(labels, cm):
        print(f"{label:>7} " + "".join(f"{v:>6}" for v in row))

    print()
    print("=" * 64)
    print("RELATORIO POR CLASSE")
    print("=" * 64)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default=train_model.MODEL_DIR)
    main(parser.parse_args().model_dir)
