"""Exemplos negativos para a classe "_none" (Parte 5): janelas de 1,5 s em que
NAO ha acorde para reconhecer.

Tres tipos (o dataset so tem acordes, entao nao ha "notas soltas" para usar):

- low_volume  (do PROPRIO dataset): uma janela de acorde real atenuada e
  enterrada em ruido de fundo (SNR -10 a -3 dB), com RMS final entre 0,004 e
  0,012 -- logo acima do SILENCE_THRESHOLD (0,003) do app, ou seja, audio que
  passa pelo filtro de silencio mas em que o acorde nao e mais reconhecivel.
  Um exemplo para cada 4 arquivos de cada split (i % 4 == 0); o exemplo herda
  o split do arquivo de origem, entao o split por arquivo e mantido.
- noise       (sintetico): ruido branco, rosa, marrom, ou zumbido eletrico
  (50/60 Hz + harmonicos + ruido rosa), com RMS log-uniforme entre 0,004 e 0,1.
- percussion  (sintetico): 1 a 6 batidas aleatorias -- "knock" no corpo do
  violao (rajada de ruido 80-400 Hz), click, bumbo (senoide 150->50 Hz) e caixa
  (ruido 1-5 kHz + tom de 190 Hz) -- sobre um piso de ruido rosa.

Os sinteticos sao "pseudo-arquivos" (1 janela cada, gerados por semente) e sao
divididos entre treino/validacao/teste nas mesmas proporcoes do split por
arquivo (20% teste; 15% do resto para validacao).
"""
import hashlib
import os

import numpy as np
from joblib import Parallel, delayed

from experiments import augment, common
from experiments.common import tm
from experiments.features import CACHE_NAMES, EXTRACTORS

NONE_LABEL = '_none'
KINDS = ('low_volume', 'noise', 'percussion')
N_SYNTH = 370           # por tipo sintetico (~ o mesmo numero de low_volume)
LOW_VOLUME_EVERY = 4    # 1 exemplo low_volume a cada 4 arquivos
SR = tm.SAMPLE_RATE
WIN = int(SR * tm.WINDOW_SECONDS)
N_JOBS = min(8, os.cpu_count() or 1)


def _set_rms(x, target):
    return x * (target / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def _brown_noise(n, rng):
    x = np.cumsum(rng.standard_normal(n))
    x -= np.linspace(x[0], x[-1], n)  # remove a deriva
    return x / (np.std(x) + 1e-12)


def gen_low_volume(file_path, rng):
    segments = tm.make_windowed_segments(file_path)
    x = np.asarray(segments[len(segments) // 2], dtype=np.float64)
    if np.sqrt(np.mean(x ** 2)) < 1e-6:          # janela muda: usa a mais alta
        x = np.asarray(max(segments, key=lambda s: np.mean(s ** 2)), dtype=np.float64)
    x = augment.add_noise(x, rng, snr_db=rng.uniform(-10, -3), color=rng.choice(['white', 'pink']))
    return _set_rms(x, rng.uniform(0.004, 0.012))


def gen_noise(rng):
    kind = rng.choice(['white', 'pink', 'brown', 'hum'])
    if kind == 'white':
        x = augment.white_noise(WIN, rng)
    elif kind == 'pink':
        x = augment.pink_noise(WIN, rng)
    elif kind == 'brown':
        x = _brown_noise(WIN, rng)
    else:
        t = np.arange(WIN) / SR
        f0 = rng.choice([50.0, 60.0])
        x = sum(rng.uniform(0.2, 1.0) / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
                for h in range(1, 9))
        x = _set_rms(x, 1.0) + _set_rms(augment.pink_noise(WIN, rng), 10 ** (-rng.uniform(10, 20) / 20))
    return _set_rms(x, np.exp(rng.uniform(np.log(0.004), np.log(0.1))))


def _hit(kind, rng):
    n = int(SR * 0.4)
    t = np.arange(n) / SR
    if kind == 'knock':
        env = np.exp(-t / rng.uniform(0.02, 0.08))
        y = augment.bandpass(rng.standard_normal(n), rng.uniform(80, 150), rng.uniform(250, 400)) * env
    elif kind == 'click':
        env = np.exp(-t / rng.uniform(0.002, 0.008))
        y = augment.bandpass(rng.standard_normal(n), 2000, 9000) * env
    elif kind == 'kick':
        f = 50 + 100 * np.exp(-t / 0.03)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / rng.uniform(0.06, 0.15))
    else:  # snare
        env = np.exp(-t / rng.uniform(0.04, 0.12))
        y = (augment.bandpass(rng.standard_normal(n), 1000, 5000) + 0.5 * np.sin(2 * np.pi * 190 * t)) * env
    return y / (np.max(np.abs(y)) + 1e-12)


def gen_percussion(rng):
    x = _set_rms(augment.pink_noise(WIN, rng), rng.uniform(0.001, 0.005))
    for _ in range(int(rng.integers(1, 7))):
        h = _hit(rng.choice(['knock', 'click', 'kick', 'snare']), rng) * rng.uniform(0.1, 0.8)
        start = int(rng.integers(0, WIN - len(h) // 4))
        end = min(WIN, start + len(h))
        x[start:end] += h[:end - start]
    return x / max(1.0, np.max(np.abs(x)) / 0.95)


def synth_ids(split, n=N_SYNTH, seed=42):
    """Divide os pseudo-arquivos sinteticos 0..n-1 nas proporcoes do split."""
    perm = np.random.default_rng(seed).permutation(n)
    n_test = round(0.2 * n)
    n_val = round(0.15 * (n - n_test))
    parts = {'test': perm[:n_test], 'val': perm[n_test:n_test + n_val], 'train': perm[n_test + n_val:]}
    return sorted(int(i) for i in parts[split])


def specs(split, files):
    """Lista de (tipo, origem) dos negativos de um split. files = arquivos de acorde do split."""
    out = [('low_volume', p) for i, (p, _) in enumerate(files) if i % LOW_VOLUME_EVERY == 0]
    for kind in ('noise', 'percussion'):
        out += [(kind, str(i)) for i in synth_ids(split)]
    return out


def make_audio(kind, source):
    key = os.path.relpath(source, common.ROOT_DIR) if kind == 'low_volume' else source
    seed = int(hashlib.md5(f'{kind}|{key}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    if kind == 'low_volume':
        x = gen_low_volume(source, rng)
    elif kind == 'noise':
        x = gen_noise(rng)
    else:
        x = gen_percussion(rng)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _features(kind, source, feature):
    x = make_audio(kind, source)
    return EXTRACTORS[feature](x), float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def build_negatives(split, files, feature):
    """Features dos negativos do split. Retorna X, kinds, rms."""
    sp = specs(split, files)
    keys = [f'{k}|{os.path.relpath(s, common.ROOT_DIR) if k == "low_volume" else s}' for k, s in sp]
    digest = hashlib.md5('\n'.join(keys).encode()).hexdigest()[:10]
    cache_path = os.path.join(common.CACHE_DIR, f'{CACHE_NAMES[feature]}_none_{split}_{digest}.npz')
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=False)
        return d['X'], d['kinds'], d['rms']
    print(f'  gerando negativos {split} ({len(sp)} janelas)...', flush=True)
    res = Parallel(n_jobs=N_JOBS)(delayed(_features)(k, s, feature) for k, s in sp)
    X = np.array([r[0] for r in res])
    rms = np.array([r[1] for r in res])
    kinds = np.array([k for k, _ in sp])
    os.makedirs(common.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, X=X, kinds=kinds, rms=rms)
    return X, kinds, rms
