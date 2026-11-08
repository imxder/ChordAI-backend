"""Utilidades compartilhadas pelos experimentos do ChordAI.

Tudo aqui reaproveita train_model.py (split por arquivo, janelas, MFCC) sem
alterar o seu comportamento. Os experimentos salvam artefatos em
experiments/results/<nome>/. A montagem treino/validacao/teste fica em data.py.
"""
import json
import os
import random
import sys
import time

import joblib
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import tensorflow as tf  # noqa: E402
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402
from keras.layers import (  # noqa: E402
    Activation,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    MaxPooling1D,
    MaxPooling2D,
    Permute,
    Reshape,
)
from keras.models import Sequential  # noqa: E402
from keras.utils import to_categorical  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

import train_model as tm  # noqa: E402

RESULTS_DIR = os.path.join(ROOT_DIR, 'experiments', 'results')
CACHE_DIR = os.path.join(ROOT_DIR, 'experiments', 'cache')
DATASET_PATH = os.path.join(ROOT_DIR, tm.DATASET_PATH)
SPLIT_RANDOM_STATE = 42
DEFAULT_SEED = 42


def set_seeds(seed=DEFAULT_SEED):
    """Fixa as sementes para que o re-treino seja reproduzivel."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_split():
    """O MESMO split por arquivo do baseline (random_state=42)."""
    return tm.split_dataset_files(dataset_path=DATASET_PATH, random_state=SPLIT_RANDOM_STATE)


def fit_scaler(X_train, kind='baseline'):
    """Ajusta o StandardScaler SO no treino e devolve (X_train escalado, scaler).

    kind='baseline': exatamente como train_model.train_model(). Atencao: como o
      reshape e (-1, n_frames), a normalizacao e feita por POSICAO DE FRAME
      (media/desvio de todos os coeficientes naquele instante), nao por
      coeficiente. Mantido assim para reproduzir o baseline.
    kind='per_feature': normaliza cada linha de feature (coeficiente/bin).
    """
    scaler = StandardScaler()
    if kind == 'baseline':
        scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    else:
        scaler.fit(np.swapaxes(X_train, 1, 2).reshape(-1, X_train.shape[1]))
    return apply_scaler(scaler, X_train, kind), scaler


def apply_scaler(scaler, X, kind='baseline'):
    """Aplica um scaler ja treinado. X: (n, n_features, n_frames) ->
    (n, n_frames, n_features), pronto para a CNN."""
    if kind == 'baseline':
        return np.swapaxes(scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape), 1, 2)
    n_feat = X.shape[1]
    return scaler.transform(np.swapaxes(X, 1, 2).reshape(-1, n_feat)).reshape(len(X), -1, n_feat)


def encode_labels(labels, *y_sets):
    """LabelEncoder nas classes + one-hot de cada conjunto. Retorna (encoder, *y_onehot)."""
    encoder = LabelEncoder()
    encoder.fit(labels)
    n = len(labels)
    return (encoder, *[to_categorical(encoder.transform(y), num_classes=n) for y in y_sets])


def build_baseline_cnn(input_shape, n_classes):
    """Mesma arquitetura de train_model.train_model() (la ela esta inline, por
    isso e replicada aqui sem mudar nada)."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_gap_bn_cnn(input_shape, n_classes):
    """Variante A (Parte 4): a CNN 1D do baseline com BatchNormalization depois de
    cada Conv1D (Conv sem ativacao -> BN -> ReLU) e GlobalAveragePooling1D no
    lugar do Flatten, para a saida nao depender da posicao do acorde na janela."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=5),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=3),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cnn2d(input_shape, n_classes):
    """Variante B (Parte 4): CNN 2D pequena tratando a feature como imagem
    [n_bins x n_frames x 1]. Recebe o mesmo tensor (n_frames, n_bins) das CNNs
    1D e faz a transposicao dentro do modelo, entao scaler/evaluate/app nao mudam.
    padding='same' para o eixo de 12 bins nao encolher (12 -> 6 -> 3)."""
    n_frames, n_bins = input_shape

    def conv_bn(filters):
        return [Conv2D(filters, (3, 3), padding='same'), BatchNormalization(), Activation('relu')]

    model = Sequential([
        Input(shape=input_shape),
        Permute((2, 1)),                      # (n_frames, n_bins) -> (n_bins, n_frames)
        Reshape((n_bins, n_frames, 1)),
        *conv_bn(32),
        MaxPooling2D((2, 2)),
        *conv_bn(64),
        MaxPooling2D((2, 2)),
        *conv_bn(128),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


ARCHITECTURES = {
    'baseline': build_baseline_cnn,
    'gap_bn': build_gap_bn_cnn,
    'cnn2d': build_cnn2d,
}


def build_model(arch, input_shape, n_classes):
    return ARCHITECTURES[arch](tuple(input_shape), n_classes)


def baseline_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
    ]


def save_artifacts(model, encoder, scaler, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, 'model.h5'))
    joblib.dump(encoder, os.path.join(out_dir, 'encoder.joblib'))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))


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
