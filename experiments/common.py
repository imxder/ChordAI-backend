"""Utilidades compartilhadas pelos experimentos do ChordAI.

O que producao e experimentos precisam fazer IGUAL (janelas, split, features,
scalers, modelos, negativos, model_config.json) fica em pipeline.py, na raiz;
aqui ficam so a avaliacao (metrics.json, matriz de confusao) e os caminhos dos
experimentos. A montagem com cache fica em data.py.
"""
import json
import os
import sys
import time

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import pipeline as pl  # noqa: E402
from pipeline import (  # noqa: E402,F401
    DEFAULT_SEED,
    SPLIT_RANDOM_STATE,
    apply_scaler,
    encode_labels,
    fit_scaler,
    set_seeds,
)

# CHORDAI_RESULTS_DIR / CHORDAI_CACHE_DIR permitem rodar em outra pasta (ex.: a
# conferencia de equivalencia da Parte 6) sem mexer nos resultados oficiais.
RESULTS_DIR = os.environ.get('CHORDAI_RESULTS_DIR') or os.path.join(ROOT_DIR, 'experiments', 'results')
CACHE_DIR = os.environ.get('CHORDAI_CACHE_DIR') or os.path.join(ROOT_DIR, 'experiments', 'cache')
DATASET_PATH = pl.DATASET_PATH

# nomes de arquitetura usados na linha de comando / nomes dos grupos dos experimentos
ARCHITECTURES = {'baseline': 'flatten', 'gap_bn': 'gap_bn', 'cnn2d': 'cnn2d'}


def load_split():
    """O MESMO split por arquivo do baseline (random_state=42)."""
    return pl.split_dataset_files(dataset_path=DATASET_PATH)


def build_model(arch, input_shape, n_classes):
    return pl.build_model(arch, input_shape, n_classes)


def build_baseline_cnn(input_shape, n_classes):
    return pl.build_flatten_cnn(tuple(input_shape), n_classes)


def baseline_callbacks():
    return pl.training_callbacks()


def save_artifacts(model, encoder, scaler, out_dir, config=None):
    """Salva model.h5, encoder.joblib, scaler.joblib e, se dado, model_config.json."""
    if config is None:
        os.makedirs(out_dir, exist_ok=True)
        model.save(os.path.join(out_dir, 'model.h5'))
        import joblib
        joblib.dump(encoder, os.path.join(out_dir, 'encoder.joblib'))
        joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
    else:
        pl.save_model_bundle(model, encoder, scaler, config, out_dir)


def _measure_inference_ms(model, X_test, n_samples=200, warmup=10):
    """Tempo medio de model.predict para UMA janela (batch=1), como em app.py."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    for i in idx[:warmup]:
        model.predict(X_test[i:i + 1], verbose=0)
    times = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict(X_test[i:i + 1], verbose=0)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.median(times))


def _save_confusion(cm, labels, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    csv_path = os.path.join(out_dir, 'confusion_matrix.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('esperado\\previsto,' + ','.join(labels) + '\n')
        for label, row in zip(labels, cm):
            f.write(label + ',' + ','.join(str(int(v)) for v in row) + '\n')

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Esperado')
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > threshold else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Matriz de confusão (janelas de teste)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)


def evaluate(model, X_test, y_test, encoder, out_dir, extra=None):
    """Avalia no conjunto de teste e grava em out_dir:
    metrics.json, confusion_matrix.csv/.png e classification_report.txt.

    y_test pode ser one-hot, indices inteiros ou nomes de classe.
    extra: dict opcional mesclado ao metrics.json (ex.: epocas treinadas).
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = [str(c) for c in encoder.classes_]

    y_test = np.asarray(y_test)
    if y_test.ndim == 2:
        y_true_idx = np.argmax(y_test, axis=1)
    elif y_test.dtype.kind in ('U', 'S', 'O'):
        y_true_idx = encoder.transform(y_test)
    else:
        y_true_idx = y_test.astype(int)

    y_pred_idx = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = encoder.inverse_transform(y_true_idx)
    y_pred = encoder.inverse_transform(y_pred_idx)

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mean_ms, median_ms = _measure_inference_ms(model, X_test)

    metrics = {
        'experiment': os.path.basename(os.path.normpath(out_dir)),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'f1_per_class': {label: float(v) for label, v in zip(labels, per_class)},
        'n_params': int(model.count_params()),
        'inference_ms_per_window': mean_ms,
        'inference_ms_per_window_median': median_ms,
        'n_test_windows': int(len(y_true)),
        'classes': labels,
    }
    if extra:
        metrics.update(extra)

    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    _save_confusion(cm, labels, out_dir)

    print(f"\n[{metrics['experiment']}] accuracy={accuracy:.4f}  macro-F1={macro_f1:.4f}  "
          f"params={metrics['n_params']}  {mean_ms:.2f} ms/janela")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    return metrics
