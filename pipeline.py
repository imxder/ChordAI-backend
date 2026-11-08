"""Pipeline compartilhado do ChordAI: tudo que PRODUCAO (train_model.py, app.py)
e EXPERIMENTOS (experiments/) precisam fazer exatamente igual.

- janelas de audio e split por arquivo (treino / validacao / teste)
- extratores de feature: mfcc, chroma, cqt, mfcc+chroma
- scalers: 'frame' (o original de train_model.py, normaliza por posicao de
  frame) e 'feature' (normaliza cada coeficiente/bin)
- construtores de modelo: 'flatten' (CNN 1D original), 'gap_bn', 'cnn2d'
- negativos da classe "_none" (Parte 5)
- model_config.json: a configuracao salva junto de cada modelo, para a
  inferencia extrair a feature e aplicar o scaler EXATAMENTE como no treino

Este modulo nao depende de experiments/.
"""
import hashlib
import json
import os
import random

import joblib
import librosa
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset')

# ---------------------------------------------------------------------------
# Constantes de audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
N_MFCC = 40
HOP_LENGTH = 512  # padrao do librosa; igual para todas as features
# WINDOW_SECONDS precisa ser igual a RECORD_DURATION em audio_capture.py: o
# modelo so acerta ao vivo se treinar com o mesmo tamanho de janela que vai
# receber na captura em tempo real.
WINDOW_SECONDS = 1.5
CROP_HOP_SECONDS = WINDOW_SECONDS / 2  # 50% de sobreposicao entre janelas de treino
MAX_CROPS_PER_FILE = 8  # evita que arquivos longos dominem o dataset
MIN_VALID_DURATION = 0.05
SILENCE_THRESHOLD = 0.003  # RMS abaixo disso: "silence", o modelo nem e chamado

SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
VAL_SIZE = 0.15
VAL_RANDOM_STATE = 42
DEFAULT_SEED = 42
N_JOBS = min(8, os.cpu_count() or 1)  # cada worker carrega librosa/TF; 28 estouravam a RAM


def _compute_max_pad_len(window_seconds=WINDOW_SECONDS, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
    dummy = np.zeros(int(sample_rate * window_seconds), dtype=np.float32)
    return librosa.feature.mfcc(y=dummy, sr=sample_rate, n_mfcc=n_mfcc).shape[1]


MAX_PAD_LEN = _compute_max_pad_len()  # 65 frames para 1,5 s


def set_seeds(seed=DEFAULT_SEED):
    """Fixa as sementes (Python, NumPy, TF) e liga o determinismo do TF."""
    import tensorflow as tf
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset, split e janelas
# ---------------------------------------------------------------------------
def list_dataset_files(dataset_path=DATASET_PATH):
    """Lista (file_path, label) de todo o dataset, com a mesma filtragem por
    duracao minima usada no treino (sem decodificar o audio inteiro)."""
    file_paths, labels = [], []
    valid_labels = sorted(
        d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
    )
    for label in valid_labels:
        label_path = os.path.join(dataset_path, label)
        # sorted(): os.listdir() nao garante ordem (NTFS ordena sem diferenciar
        # maiusculas, ext4 por hash, APFS de outro jeito), e a ordem define o split.
        for filename in sorted(os.listdir(label_path)):
            if not filename.endswith(".wav"):
                continue
            file_path = os.path.join(label_path, filename)
            try:
                info = sf.info(file_path)
                duration = info.frames / info.samplerate
            except Exception:
                continue
            if duration >= MIN_VALID_DURATION:
                file_paths.append(file_path)
                labels.append(label)
    return file_paths, labels, valid_labels


def split_dataset_files(dataset_path=DATASET_PATH, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE):
    """Divide por ARQUIVO (nao por janela/crop), para nenhum recorte do mesmo
    audio vazar entre treino e teste."""
    file_paths, labels, valid_labels = list_dataset_files(dataset_path)
    train_idx, test_idx = train_test_split(
        np.arange(len(file_paths)), test_size=test_size, random_state=random_state, stratify=labels
    )
    train_files = [(file_paths[i], labels[i]) for i in train_idx]
    test_files = [(file_paths[i], labels[i]) for i in test_idx]
    return train_files, test_files, valid_labels


def split_train_val(train_files, val_size=VAL_SIZE, random_state=VAL_RANDOM_STATE):
    """Validacao = val_size dos arquivos de TREINO (por arquivo, estratificado)."""
    labels = [l for _, l in train_files]
    tr, va = train_test_split(np.arange(len(train_files)), test_size=val_size,
                              random_state=random_state, stratify=labels)
    tr, va = np.sort(tr), np.sort(va)
    return [train_files[i] for i in tr], [train_files[i] for i in va]


def make_windowed_segments(file_path, window_seconds=WINDOW_SECONDS, hop_seconds=CROP_HOP_SECONDS,
                           sample_rate=SAMPLE_RATE, max_crops=MAX_CROPS_PER_FILE):
    """Recorta o arquivo em janelas deslizantes do tamanho da captura ao vivo,
    simulando os offsets que get_audio_segment() entrega em audio_capture.py."""
    audio, _ = librosa.load(file_path, sr=sample_rate)
    window_len = int(sample_rate * window_seconds)

    if len(audio) < window_len:
        audio = np.pad(audio, (0, window_len - len(audio)))

    hop_len = max(1, int(sample_rate * hop_seconds))
    starts = list(range(0, len(audio) - window_len + 1, hop_len))
    if not starts:
        starts = [0]

    if len(starts) > max_crops:
        idx = sorted(set(np.linspace(0, len(starts) - 1, max_crops).round().astype(int)))
        starts = [starts[i] for i in idx]

    return [audio[s:s + window_len] for s in starts]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# ---------------------------------------------------------------------------
# Features: entrada = segmento de audio; saida = [n_features, MAX_PAD_LEN] ou None
# ---------------------------------------------------------------------------
CQT_FMIN = librosa.note_to_hz('C1')  # 32,7 Hz
CQT_BINS_PER_OCTAVE = 12
CQT_N_OCTAVES = 7  # C1..B7 (~3,9 kHz), abaixo de Nyquist (11025 Hz)
CQT_N_BINS = CQT_BINS_PER_OCTAVE * CQT_N_OCTAVES  # 84


def _fix_frames(mat, max_pad_len=MAX_PAD_LEN):
    if mat.shape[1] > max_pad_len:
        return mat[:, :max_pad_len]
    return np.pad(mat, ((0, 0), (0, max_pad_len - mat.shape[1])), mode='constant')


def extract_features(audio_data, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN, is_file=True):
    """40 MFCC (o extrator original de train_model.py)."""
    try:
        if is_file:
            audio, sr = librosa.load(audio_data, sr=sample_rate)
        else:
            audio = audio_data
            sr = sample_rate

        if len(audio) < sample_rate * MIN_VALID_DURATION:
            return None

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Erro ao processar áudio para features: {e}")
        return None


def extract_mfcc(audio, sr=SAMPLE_RATE):
    return extract_features(audio, sample_rate=sr, is_file=False)


def extract_chroma(audio, sr=SAMPLE_RATE):
    if len(audio) < sr * MIN_VALID_DURATION:
        return None
    return _fix_frames(librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP_LENGTH))


def extract_cqt(audio, sr=SAMPLE_RATE):
    if len(audio) < sr * MIN_VALID_DURATION:
        return None
    C = np.abs(librosa.cqt(y=audio, sr=sr, hop_length=HOP_LENGTH, fmin=CQT_FMIN,
                           n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE))
    return _fix_frames(librosa.amplitude_to_db(C, ref=np.max))


def extract_mfcc_chroma(audio, sr=SAMPLE_RATE):
    mfcc = extract_mfcc(audio, sr)
    chroma = extract_chroma(audio, sr)
    if mfcc is None or chroma is None:
        return None
    return np.concatenate([mfcc, chroma], axis=0)


EXTRACTORS = {
    'mfcc': extract_mfcc,
    'chroma': extract_chroma,
    'cqt': extract_cqt,
    'mfcc+chroma': extract_mfcc_chroma,
}
N_FEATURES = {'mfcc': N_MFCC, 'chroma': 12, 'cqt': CQT_N_BINS, 'mfcc+chroma': N_MFCC + 12}


def build_windowed_dataset(file_label_pairs):
    """MFCC de todas as janelas (o formato original de train_model.py)."""
    X, y = [], []
    for file_path, label in file_label_pairs:
        for segment in make_windowed_segments(file_path):
            features = extract_features(segment, is_file=False)
            if features is not None:
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)


def _process_file(file_idx, file_path, label, feature, transform, n_versions, seed):
    extractor = EXTRACTORS[feature]
    out = []
    for win_idx, segment in enumerate(make_windowed_segments(file_path)):
        for k in range(n_versions):
            audio = segment if transform is None else transform(segment, seed, file_idx, win_idx, k)
            feats = extractor(audio)
            if feats is not None:
                out.append((k, file_idx, feats, label, rms(audio)))
    return out


def build_features(file_label_pairs, feature, transform=None, n_versions=1, seed=DEFAULT_SEED,
                   n_jobs=N_JOBS):
    """Features de cada janela de cada arquivo, em paralelo.
    transform(segment, seed, file_idx, win_idx, k) -> audio permite aplicar uma
    transformacao (augmentation/degradacao) antes da extracao; None = janela limpa.
    Retorna X [n, n_features, n_frames], y, k (versao), fidx (arquivo), rms (da janela)."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_file)(i, p, l, feature, transform, n_versions, seed)
        for i, (p, l) in enumerate(file_label_pairs)
    )
    flat = [r for file_res in results for r in file_res]
    return (np.array([r[2] for r in flat]), np.array([r[3] for r in flat]),
            np.array([r[0] for r in flat]), np.array([r[1] for r in flat]),
            np.array([r[4] for r in flat]))


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------
SCALER_ALIASES = {'frame': 'frame', 'baseline': 'frame', 'feature': 'feature', 'per_feature': 'feature'}


def scaler_kind(kind):
    return SCALER_ALIASES[kind]


def fit_scaler(X_train, kind='frame'):
    """Ajusta o StandardScaler SO no treino e devolve (X_train escalado, scaler).

    kind='frame' (alias 'baseline'): exatamente como o train_model.py original.
      Como o reshape e (-1, n_frames), normaliza por POSICAO DE FRAME (media/desvio
      de todos os coeficientes naquele instante), nao por coeficiente.
    kind='feature' (alias 'per_feature'): normaliza cada linha de feature.
    """
    kind = scaler_kind(kind)
    scaler = StandardScaler()
    if kind == 'frame':
        scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    else:
        scaler.fit(np.swapaxes(X_train, 1, 2).reshape(-1, X_train.shape[1]))
    return apply_scaler(scaler, X_train, kind), scaler


def apply_scaler(scaler, X, kind='frame'):
    """Aplica um scaler ja treinado. X: (n, n_features, n_frames) ->
    (n, n_frames, n_features), pronto para a CNN."""
    if scaler_kind(kind) == 'frame':
        return np.swapaxes(scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape), 1, 2)
    n_feat = X.shape[1]
    return scaler.transform(np.swapaxes(X, 1, 2).reshape(-1, n_feat)).reshape(len(X), -1, n_feat)


def encode_labels(labels, *y_sets):
    """LabelEncoder nas classes + one-hot de cada conjunto. Retorna (encoder, *y_onehot)."""
    from keras.utils import to_categorical
    encoder = LabelEncoder()
    encoder.fit(labels)
    n = len(labels)
    return (encoder, *[to_categorical(encoder.transform(y), num_classes=n) for y in y_sets])


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
ARCH_ALIASES = {'flatten': 'flatten', 'baseline': 'flatten', 'gap_bn': 'gap_bn', 'cnn2d': 'cnn2d'}


def arch_name(arch):
    return ARCH_ALIASES[arch]


def build_flatten_cnn(input_shape, n_classes):
    """CNN 1D original de train_model.py (Conv64 -> Conv128 -> Flatten -> Dense128)."""
    from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
    from keras.models import Sequential
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
    """Variante A (Parte 4): CNN 1D com BatchNorm depois de cada Conv1D (Conv sem
    ativacao -> BN -> ReLU) e GlobalAveragePooling1D no lugar do Flatten."""
    from keras.layers import (Activation, BatchNormalization, Conv1D, Dense, Dropout,
                              GlobalAveragePooling1D, Input, MaxPooling1D)
    from keras.models import Sequential
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
    [n_bins x n_frames x 1]. Recebe o mesmo tensor (n_frames, n_bins) das CNNs 1D
    e transpoe dentro do modelo. padding='same' (12 -> 6 -> 3 bins)."""
    from keras.layers import (Activation, BatchNormalization, Conv2D, Dense, Dropout,
                              GlobalAveragePooling2D, Input, MaxPooling2D, Permute, Reshape)
    from keras.models import Sequential
    n_frames, n_bins = input_shape

    def conv_bn(filters):
        return [Conv2D(filters, (3, 3), padding='same'), BatchNormalization(), Activation('relu')]

    model = Sequential([
        Input(shape=input_shape),
        Permute((2, 1)),
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
    'flatten': build_flatten_cnn,
    'gap_bn': build_gap_bn_cnn,
    'cnn2d': build_cnn2d,
}


def build_model(arch, input_shape, n_classes):
    return ARCHITECTURES[arch_name(arch)](tuple(input_shape), n_classes)


def training_callbacks():
    """EarlyStopping / ReduceLROnPlateau do baseline (monitoram a VALIDACAO)."""
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    return [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
    ]


# ---------------------------------------------------------------------------
# Ruido e filtros (usados pelos negativos e pelo augmentation dos experimentos)
# ---------------------------------------------------------------------------
def white_noise(n, rng):
    return rng.standard_normal(n)


def pink_noise(n, rng):
    """Ruido 1/f gerado no dominio da frequencia."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.arange(len(spectrum), dtype=np.float64)
    freqs[0] = 1.0
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n=n)
    return noise / (np.std(noise) + 1e-12)


def add_noise(x, rng, snr_db, color):
    noise = white_noise(len(x), rng) if color == 'white' else pink_noise(len(x), rng)
    noise *= (rms(x) + 1e-12) / ((rms(noise) + 1e-12) * 10 ** (snr_db / 20))
    return x + noise


def bandpass(x, low_hz, high_hz, sr=SAMPLE_RATE, order=4):
    sos = butter(order, [low_hz, high_hz], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, x)


# ---------------------------------------------------------------------------
# Classe "_none" (Parte 5): exemplos negativos
# ---------------------------------------------------------------------------
NONE_LABEL = '_none'
NEGATIVE_KINDS = ('low_volume', 'noise', 'percussion')
N_SYNTH_NEGATIVES = 370  # por tipo sintetico (~ o mesmo numero de low_volume)
LOW_VOLUME_EVERY = 4     # 1 exemplo low_volume a cada 4 arquivos do split
_WIN = int(SAMPLE_RATE * WINDOW_SECONDS)


def _set_rms(x, target):
    return x * (target / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def _brown_noise(n, rng):
    x = np.cumsum(rng.standard_normal(n))
    x -= np.linspace(x[0], x[-1], n)
    return x / (np.std(x) + 1e-12)


def _gen_low_volume(file_path, rng):
    """Janela real de acorde atenuada e enterrada em ruido (SNR -10 a -3 dB),
    RMS final 0,004-0,012: passa pelo filtro de silencio, mas o acorde some."""
    segments = make_windowed_segments(file_path)
    x = np.asarray(segments[len(segments) // 2], dtype=np.float64)
    if np.sqrt(np.mean(x ** 2)) < 1e-6:
        x = np.asarray(max(segments, key=lambda s: np.mean(s ** 2)), dtype=np.float64)
    x = add_noise(x, rng, snr_db=rng.uniform(-10, -3), color=rng.choice(['white', 'pink']))
    return _set_rms(x, rng.uniform(0.004, 0.012))


def _gen_noise(rng):
    """Ruido branco, rosa, marrom ou zumbido eletrico (50/60 Hz + harmonicos)."""
    kind = rng.choice(['white', 'pink', 'brown', 'hum'])
    if kind == 'white':
        x = white_noise(_WIN, rng)
    elif kind == 'pink':
        x = pink_noise(_WIN, rng)
    elif kind == 'brown':
        x = _brown_noise(_WIN, rng)
    else:
        t = np.arange(_WIN) / SAMPLE_RATE
        f0 = rng.choice([50.0, 60.0])
        x = sum(rng.uniform(0.2, 1.0) / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
                for h in range(1, 9))
        x = _set_rms(x, 1.0) + _set_rms(pink_noise(_WIN, rng), 10 ** (-rng.uniform(10, 20) / 20))
    return _set_rms(x, np.exp(rng.uniform(np.log(0.004), np.log(0.1))))


def _hit(kind, rng):
    n = int(SAMPLE_RATE * 0.4)
    t = np.arange(n) / SAMPLE_RATE
    if kind == 'knock':
        env = np.exp(-t / rng.uniform(0.02, 0.08))
        y = bandpass(rng.standard_normal(n), rng.uniform(80, 150), rng.uniform(250, 400)) * env
    elif kind == 'click':
        env = np.exp(-t / rng.uniform(0.002, 0.008))
        y = bandpass(rng.standard_normal(n), 2000, 9000) * env
    elif kind == 'kick':
        f = 50 + 100 * np.exp(-t / 0.03)
        y = np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE) * np.exp(-t / rng.uniform(0.06, 0.15))
    else:  # snare
        env = np.exp(-t / rng.uniform(0.04, 0.12))
        y = (bandpass(rng.standard_normal(n), 1000, 5000) + 0.5 * np.sin(2 * np.pi * 190 * t)) * env
    return y / (np.max(np.abs(y)) + 1e-12)


def _gen_percussion(rng):
    """1 a 6 batidas (knock no corpo, click, bumbo, caixa) sobre ruido rosa."""
    x = _set_rms(pink_noise(_WIN, rng), rng.uniform(0.001, 0.005))
    for _ in range(int(rng.integers(1, 7))):
        h = _hit(rng.choice(['knock', 'click', 'kick', 'snare']), rng) * rng.uniform(0.1, 0.8)
        start = int(rng.integers(0, _WIN - len(h) // 4))
        end = min(_WIN, start + len(h))
        x[start:end] += h[:end - start]
    return x / max(1.0, np.max(np.abs(x)) / 0.95)


def synth_negative_ids(split, n=N_SYNTH_NEGATIVES, seed=42):
    """Divide os pseudo-arquivos sinteticos 0..n-1 nas proporcoes do split
    (20% teste; 15% do resto para validacao)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_test = round(0.2 * n)
    n_val = round(0.15 * (n - n_test))
    parts = {'test': perm[:n_test], 'val': perm[n_test:n_test + n_val], 'train': perm[n_test + n_val:]}
    return sorted(int(i) for i in parts[split])


def _source_key(kind, source):
    """Chave estavel entre sistemas operacionais: caminho relativo a raiz com '\\'
    (a forma usada desde a Parte 5, no Windows), para gerar os MESMOS negativos
    em Windows, Linux e macOS."""
    if kind != 'low_volume':
        return source
    return os.path.relpath(source, ROOT_DIR).replace('/', '\\')


def negative_specs(split, files):
    """(tipo, origem) dos negativos de um split. files = arquivos de acorde do split;
    os low_volume herdam o split do arquivo de origem."""
    out = [('low_volume', p) for i, (p, _) in enumerate(files) if i % LOW_VOLUME_EVERY == 0]
    for kind in ('noise', 'percussion'):
        out += [(kind, str(i)) for i in synth_negative_ids(split)]
    return out


def make_negative_audio(kind, source):
    seed = int(hashlib.md5(f'{kind}|{_source_key(kind, source)}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    if kind == 'low_volume':
        x = _gen_low_volume(source, rng)
    elif kind == 'noise':
        x = _gen_noise(rng)
    else:
        x = _gen_percussion(rng)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _negative_features(kind, source, feature):
    x = make_negative_audio(kind, source)
    return EXTRACTORS[feature](x), rms(x)


def negative_specs_digest(split, files):
    keys = [f'{k}|{_source_key(k, s)}' for k, s in negative_specs(split, files)]
    return hashlib.md5('\n'.join(keys).encode()).hexdigest()[:10]


def build_negatives(split, files, feature, n_jobs=N_JOBS):
    """Features dos negativos do split. Retorna X, kinds, rms."""
    sp = negative_specs(split, files)
    res = Parallel(n_jobs=n_jobs)(delayed(_negative_features)(k, s, feature) for k, s in sp)
    return (np.array([r[0] for r in res]), np.array([k for k, _ in sp]), np.array([r[1] for r in res]))


# ---------------------------------------------------------------------------
# Montagem treino / validacao / teste
# ---------------------------------------------------------------------------
def load_split(dataset_path=DATASET_PATH):
    return split_dataset_files(dataset_path=dataset_path)


def load_dataset(feature, with_none=False, dataset_path=DATASET_PATH, build_fn=None, negatives_fn=None,
                 verbose=True):
    """Treino (85% dos arquivos de treino), validacao (15%) e teste (o split do
    baseline), janelas limpas, com a classe '_none' opcional.

    build_fn(files, feature) -> (X, y, ...) e negatives_fn(split, files, feature) ->
    (X, kinds, rms) permitem trocar a extracao (ex.: com cache, nos experimentos).
    Retorna dict com X_*/y_* (features [n, n_features, n_frames], antes do scaler),
    labels, os arquivos de cada parte e, com _none, as contagens de negativos."""
    build_fn = build_fn or (lambda files, feat: build_features(files, feat))
    negatives_fn = negatives_fn or build_negatives
    train_files, test_files, labels = split_dataset_files(dataset_path)
    train_sub, val_files = split_train_val(train_files)

    out = {'labels': list(labels), 'train_files': train_sub, 'val_files': val_files, 'test_files': test_files,
           'n_train_files': len(train_sub), 'n_val_files': len(val_files)}
    for name, files in (('train', train_sub), ('val', val_files), ('test', test_files)):
        res = build_fn(files, feature)
        out[f'X_{name}'], out[f'y_{name}'] = res[0], res[1]
        out[f'rms_{name}'] = res[4] if len(res) > 4 else None  # RMS das janelas de ACORDE

    if with_none:
        counts = {}
        for split in ('train', 'val', 'test'):
            files = {'train': train_sub, 'val': val_files, 'test': test_files}[split]
            X_n, kinds, _ = negatives_fn(split, files, feature)
            out[f'X_{split}'] = np.concatenate([out[f'X_{split}'], X_n])
            out[f'y_{split}'] = np.concatenate([out[f'y_{split}'], np.full(len(X_n), NONE_LABEL)])
            counts[split] = {k: int((kinds == k).sum()) for k in sorted(set(kinds))}
            if split == 'test':
                out['test_none_kinds'] = kinds
        out['labels'] = list(labels) + [NONE_LABEL]
        out['none_counts'] = counts

    if verbose:
        print(f"  arquivos: treino={len(train_sub)} val={len(val_files)} teste={len(test_files)} | "
              f"janelas: treino={len(out['X_train'])} val={len(out['X_val'])} teste={len(out['X_test'])}"
              + (f" | negativos: {out['none_counts']}" if with_none else ''), flush=True)
    return out


# ---------------------------------------------------------------------------
# Limiar de confianca
# ---------------------------------------------------------------------------
THRESHOLD_GRID = np.round(np.arange(0.30, 0.951, 0.05), 2)
MAX_FALSE_ACCEPT = 0.05
MIN_ACCEPTED_ACCURACY = 0.98


def threshold_curve(probs_chords, y_chords, probs_neg, classes, thresholds=THRESHOLD_GRID):
    """Para cada limiar: cobertura e acuracia nas janelas de acorde aceitas e
    falso aceite dos negativos (aceito = classe != _none e confianca >= limiar)."""
    classes = np.asarray(classes)
    conf_c, pred_c = probs_chords.max(axis=1), classes[probs_chords.argmax(axis=1)]
    conf_n, pred_n = probs_neg.max(axis=1), classes[probs_neg.argmax(axis=1)]
    rows = []
    for t in thresholds:
        acc_c = (pred_c != NONE_LABEL) & (conf_c >= t)
        acc_n = (pred_n != NONE_LABEL) & (conf_n >= t)
        rows.append({'threshold': float(t), 'coverage': float(acc_c.mean()),
                     'accuracy': float((acc_c & (pred_c == y_chords)).sum() / max(1, acc_c.sum())),
                     'false_accept': float(acc_n.mean())})
    return rows


def choose_threshold(rows, max_false_accept=MAX_FALSE_ACCEPT, min_accuracy=MIN_ACCEPTED_ACCURACY):
    """Maior cobertura com falso aceite <= max e acuracia nos aceitos >= min (o
    criterio da Parte 5). Se nenhum limiar atender, o de menor falso aceite."""
    ok = [r for r in rows if r['false_accept'] <= max_false_accept and r['accuracy'] >= min_accuracy]
    if ok:
        return max(ok, key=lambda r: r['coverage'])['threshold']
    return min(rows, key=lambda r: (r['false_accept'], -r['coverage']))['threshold']


# ---------------------------------------------------------------------------
# model_config.json
# ---------------------------------------------------------------------------
CONFIG_FILENAME = 'model_config.json'
CONFIG_VERSION = 1
MODEL_FILES = {'model': 'model.h5', 'encoder': 'encoder.joblib', 'scaler': 'scaler.joblib'}
LEGACY_FILES = {'model': 'chord_recognizer_cnn_model.h5', 'encoder': 'label_encoder_chords.joblib',
                'scaler': 'scaler_chords.joblib'}


def make_model_config(feature, scaler, arch, classes, with_none, confidence_threshold=None, files=None,
                      extra=None):
    cfg = {
        'version': CONFIG_VERSION,
        'feature': feature,
        'scaler': scaler_kind(scaler),
        'arch': arch_name(arch),
        'classes': [str(c) for c in classes],
        'with_none': bool(with_none),
        'none_label': NONE_LABEL if with_none else None,
        'confidence_threshold': None if confidence_threshold is None else float(confidence_threshold),
        'sample_rate': SAMPLE_RATE,
        'window_seconds': WINDOW_SECONDS,
        'hop_length': HOP_LENGTH,
        'n_frames': MAX_PAD_LEN,
        'n_features': N_FEATURES[feature],
        'n_mfcc': N_MFCC if 'mfcc' in feature else None,
        'files': dict(files or MODEL_FILES),
    }
    if extra:
        cfg.update(extra)
    return cfg


def legacy_model_config(classes=None):
    """Pipeline antigo (pastas sem model_config.json, ex.: candidate_91_81):
    40 MFCC + scaler por frame + CNN 1D com Flatten, sem _none e sem limiar."""
    cfg = make_model_config('mfcc', 'frame', 'flatten', classes or [], with_none=False,
                            confidence_threshold=None, files=LEGACY_FILES)
    cfg['version'] = 0
    cfg['legacy'] = True
    return cfg


def validate_config(cfg):
    """Garante que a inferencia vai processar o audio como no treino."""
    expected = {'sample_rate': SAMPLE_RATE, 'window_seconds': WINDOW_SECONDS, 'hop_length': HOP_LENGTH,
                'n_frames': MAX_PAD_LEN}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'model_config.json: {key}={cfg.get(key)!r}, mas este pipeline usa {value!r}')
    if cfg['feature'] not in EXTRACTORS:
        raise ValueError(f"model_config.json: feature desconhecida {cfg['feature']!r}")
    scaler_kind(cfg['scaler'])
    arch_name(cfg['arch'])


def load_model_config(model_dir):
    path = os.path.join(model_dir, CONFIG_FILENAME)
    if not os.path.exists(path):
        return legacy_model_config()
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


def save_model_bundle(model, encoder, scaler, config, out_dir, metrics=None):
    """Salva modelo, encoder, scaler e model_config.json (e metrics.json, se dado)."""
    os.makedirs(out_dir, exist_ok=True)
    files = config['files']
    model.save(os.path.join(out_dir, files['model']))
    joblib.dump(encoder, os.path.join(out_dir, files['encoder']))
    joblib.dump(scaler, os.path.join(out_dir, files['scaler']))
    with open(os.path.join(out_dir, CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if metrics is not None:
        with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


def model_bundle_exists(model_dir):
    files = load_model_config(model_dir)['files']
    return all(os.path.exists(os.path.join(model_dir, f)) for f in files.values())


def load_model_bundle(model_dir):
    """Carrega (model, encoder, scaler, config). Sem model_config.json, assume o pipeline antigo."""
    import tensorflow as tf
    cfg = load_model_config(model_dir)
    files = cfg['files']
    model = tf.keras.models.load_model(os.path.join(model_dir, files['model']))
    encoder = joblib.load(os.path.join(model_dir, files['encoder']))
    scaler = joblib.load(os.path.join(model_dir, files['scaler']))
    if not cfg['classes']:
        cfg['classes'] = [str(c) for c in encoder.classes_]
    if list(cfg['classes']) != [str(c) for c in encoder.classes_]:
        raise ValueError(f"model_config.json: classes {cfg['classes']} != encoder {list(encoder.classes_)}")
    return model, encoder, scaler, cfg


def features_for_model(audio, scaler, config, sample_rate=SAMPLE_RATE):
    """Audio de uma janela -> tensor (1, n_frames, n_features) para o modelo,
    com a feature e o scaler definidos no config. None se o audio for curto."""
    if sample_rate != config['sample_rate']:
        audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sample_rate,
                                 target_sr=config['sample_rate'])
    feats = EXTRACTORS[config['feature']](audio, config['sample_rate'])
    if feats is None:
        return None
    return apply_scaler(scaler, feats[np.newaxis], config['scaler'])
