"""
Mede a acuracia real do modelo de acordes em condicoes de captura ao vivo
(janelas de RECORD_DURATION segundos, como em audio_capture.py).

Reusa o split de teste e o corte em janelas de pipeline.py e o mesmo caminho de
inferencia de producao: a feature, o scaler, a classe "_none" e o limiar de
confianca vem do model_config.json do modelo (pastas sem esse arquivo seguem o
pipeline antigo: MFCC, sem _none e sem limiar).

Uso:  python evaluate_realtime.py [--model-dir trained_model]
"""
import argparse
import time

import numpy as np

import pipeline
import train_model
from audio_capture import RECORD_DURATION
from train_model import WINDOW_SECONDS, make_windowed_segments, split_dataset_files

assert abs(RECORD_DURATION - WINDOW_SECONDS) < 1e-6, (
    f"RECORD_DURATION (audio_capture.py, {RECORD_DURATION}s) != WINDOW_SECONDS "
    f"(train_model.py, {WINDOW_SECONDS}s) - ajuste os dois para o mesmo valor "
    "antes de avaliar, senao a medicao nao reflete a captura ao vivo real."
)


def predict_with_timing(segment, model, encoder, scaler, config):
    """Replica _predict_raw() mas mede separadamente a extracao de features
    (Librosa + scaler) e a inferencia do modelo. Retorna
    (acorde_argmax, desconhecido?, t_extracao, t_inferencia)."""
    t0 = time.perf_counter()
    features = pipeline.features_for_model(segment, scaler, config)
    extraction_time = time.perf_counter() - t0
    if features is None:
        return None, True, extraction_time, None

    t1 = time.perf_counter()
    prediction = model.predict(features, verbose=0)[0]
    inference_time = time.perf_counter() - t1

    idx = int(np.argmax(prediction))
    label = encoder.inverse_transform([idx])[0]
    return label, train_model._is_unknown(label, float(prediction[idx]), config), extraction_time, inference_time


def print_timing_block(title, times_seconds):
    times_ms = np.array(times_seconds) * 1000
    print(f"{title}")
    print(f"  media:   {times_ms.mean():.2f} ms")
    print(f"  mediana: {np.median(times_ms):.2f} ms")
    print(f"  p95:     {np.percentile(times_ms, 95):.2f} ms")


def main(model_dir=train_model.MODEL_DIR):
    model, encoder, scaler = train_model.load_trained_model(model_dir)
    if model is None:
        raise SystemExit(f"Nao foi possivel carregar o modelo de {model_dir}")
    config = encoder.model_config_

    print("Reconstruindo split de teste (20% nao usado no treino, mesmo random_state=42)...")
    _, test_files, _ = split_dataset_files()
    print(f"{len(test_files)} arquivos de teste identificados.\n")
    print(f"Simulando captura ao vivo: janelas de {RECORD_DURATION}s...\n")

    extraction_times, inference_times = [], []
    total = correct_argmax = unknown = accepted_correct = 0
    for file_path, true_label in test_files:
        for segment in make_windowed_segments(file_path):
            predicted, is_unknown, extraction_time, inference_time = predict_with_timing(
                segment, model, encoder, scaler, config)
            extraction_times.append(extraction_time)
            if inference_time is None:
                continue
            inference_times.append(inference_time)
            total += 1
            correct_argmax += predicted == true_label
            unknown += is_unknown
            accepted_correct += (not is_unknown) and predicted == true_label

    feature_label = f"Librosa/{config['feature']}"
    print("=" * 64)
    print("RELATORIO - Acuracia e latencia em condicoes de captura ao vivo")
    print("=" * 64)
    print(f"Modelo:                            {model_dir}")
    print(f"Pipeline:                          feature={config['feature']} scaler={config['scaler']} "
          f"arch={config['arch']} _none={config['with_none']} limiar={config['confidence_threshold']}")
    print(f"Arquivos de teste utilizados:      {len(test_files)}")
    print(f"Segmentos de {RECORD_DURATION}s avaliados:       {total}")
    if total:
        accepted = total - unknown
        print(f"Acuracia em segmentos curtos:      {correct_argmax / total:.4f}  ({correct_argmax}/{total})  [argmax]")
        print(f"Taxa de 'desconhecido':            {unknown / total:.4f}  ({unknown}/{total})  [_none ou confianca baixa]")
        if accepted:
            print(f"Acuracia nos aceitos:              {accepted_correct / accepted:.4f}  ({accepted_correct}/{accepted})")
        print(f"Acertos com desconhecido = erro:   {accepted_correct / total:.4f}  ({accepted_correct}/{total})")
    else:
        print("Acuracia em segmentos curtos:      N/A (nenhum segmento valido)")
    print("-" * 64)
    print_timing_block(f"Tempo de extracao de features ({feature_label}):", extraction_times)
    print("-" * 64)
    print_timing_block("Tempo de inferencia do modelo (model.predict):", inference_times)
    print("-" * 64)

    if extraction_times and inference_times:
        avg_total = np.mean(extraction_times) + np.mean(inference_times)
        print(f"Tempo total medio por predicao:   {avg_total * 1000:.2f} ms")
        print(f"Orcamento por janela (RECORD_DURATION): {RECORD_DURATION * 1000:.0f} ms")
        status = "OK (cabe dentro da janela)" if avg_total < RECORD_DURATION else "ESTOURA o orcamento da janela"
        print(f"Viabilidade para tempo real:       {status}")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default=train_model.MODEL_DIR)
    main(parser.parse_args().model_dir)
