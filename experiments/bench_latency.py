"""Mede tamanho e latencia de cada modelo em experiments/results/*/ nas MESMAS
condicoes (um modelo por vez, mesma maquina, mesmas janelas), e grava o
resultado em metrics.json['latency'].

Por janela de 1,5 s, batch 1, so o modelo (sem extracao de features):
- predict_ms:  model.predict(x)  -> como app.py faz hoje (muito overhead do Keras)
- compiled_ms: tf.function(model)(x) -> custo real do calculo no TensorFlow
- tflite_ms:   tf.lite.Interpreter.invoke() -> o formato que iria para o celular
Tamanhos: h5_kb (arquivo Keras) e tflite_kb (TFLite float32, sem quantizacao).

Os tempos sao do PC (CPU), nao do celular: servem para comparar modelos entre si.

Uso:  python -m experiments.bench_latency [grupo ...]   (padrao: todos os grupos)
"""
import glob
import json
import os
import sys
import tempfile
import time

import joblib
import numpy as np
import tensorflow as tf

from experiments import common
from experiments.data import build
from experiments.eval_degraded import feature_of

N_SAMPLES = 200
WARMUP = 20


def _timeit(fn, xs):
    for x in xs[:WARMUP]:
        fn(x)
    times = []
    for x in xs:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.median(times))


def to_tflite(model):
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, verbose=False)
        return tf.lite.TFLiteConverter.from_saved_model(tmp).convert()


def bench(run_dir, test_cache):
    with open(os.path.join(run_dir, 'metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)
    feature = feature_of(metrics)
    if feature not in test_cache:
        _, test_files, _ = common.load_split()
        test_cache[feature] = build(test_files, feature, 'clean')[0]
    X = test_cache[feature]

    model = tf.keras.models.load_model(os.path.join(run_dir, 'model.h5'))
    scaler = joblib.load(os.path.join(run_dir, 'scaler.joblib'))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=N_SAMPLES, replace=False)
    Xf = common.apply_scaler(scaler, X[idx], metrics.get('scaler', 'baseline')).astype(np.float32)
    xs = [Xf[i:i + 1] for i in range(len(Xf))]

    predict_ms = _timeit(lambda x: model.predict(x, verbose=0), xs)

    spec = tf.TensorSpec((1, *Xf.shape[1:]), tf.float32)
    compiled = tf.function(lambda x: model(x, training=False), input_signature=[spec])
    compiled_ms = _timeit(lambda x: compiled(x).numpy(), xs)

    tflite_bytes = to_tflite(model)
    interp = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]['index']
    out = interp.get_output_details()[0]['index']

    def run_tflite(x):
        interp.set_tensor(inp, x)
        interp.invoke()
        return interp.get_tensor(out)

    tflite_ms = _timeit(run_tflite, xs)
    agree = float(np.mean([np.argmax(run_tflite(x)) == np.argmax(compiled(x).numpy()) for x in xs]))

    metrics['latency'] = {
        'predict_ms': predict_ms[0], 'predict_ms_median': predict_ms[1],
        'compiled_ms': compiled_ms[0], 'compiled_ms_median': compiled_ms[1],
        'tflite_ms': tflite_ms[0], 'tflite_ms_median': tflite_ms[1],
        'tflite_threads': 1,
        'tflite_agreement_with_keras': agree,
        'h5_kb': os.path.getsize(os.path.join(run_dir, 'model.h5')) / 1024,
        'tflite_kb': len(tflite_bytes) / 1024,
        'n_samples': N_SAMPLES,
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    lat = metrics['latency']
    print(f"{os.path.basename(run_dir):36s} predict={lat['predict_ms']:6.1f}  compilado={lat['compiled_ms']:5.2f}  "
          f"tflite={lat['tflite_ms']:5.2f} ms  tflite={lat['tflite_kb']:7.1f} KB  concordancia={agree:.3f}",
          flush=True)


def main():
    groups = set(sys.argv[1:])
    test_cache = {}
    for path in sorted(glob.glob(os.path.join(common.RESULTS_DIR, '*', 'metrics.json'))):
        run_dir = os.path.dirname(path)
        with open(path, encoding='utf-8') as f:
            group = json.load(f).get('group')
        if group and (not groups or group in groups):
            bench(run_dir, test_cache)


if __name__ == '__main__':
    main()
