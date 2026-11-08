"""Data augmentation no audio bruto (antes da extracao de features).
Aplicado SO no conjunto de treino; o teste continua limpo.

Cada versao aumentada de uma janela usa um gerador com semente derivada de
(seed, indice_do_arquivo, indice_da_janela, k), entao o resultado e
reproduzivel e a versao k=0 e a mesma tanto em "1 versao" quanto em "2 versoes".

Tambem define degrade_for_test(): uma degradacao FIXA (ruido rosa a 15 dB +
passa-banda 200 Hz-5 kHz) usada para montar um "teste degradado" e medir
robustez. Os parametros dela ficam fora das faixas sorteadas no treino de
proposito, para nao avaliar exatamente a mesma distribuicao do treino.
"""
import librosa
import numpy as np

import pipeline as pl

SR = pl.SAMPLE_RATE

# probabilidade de cada augmentation ser aplicada a uma versao aumentada
P_NOISE = 0.8
P_GAIN = 0.8
P_BANDPASS = 0.5
P_SHIFT = 0.5

SNR_DB_RANGE = (10.0, 30.0)
GAIN_DB_RANGE = (-12.0, 6.0)
BANDPASS_LOW_RANGE = (120.0, 200.0)     # ~150 Hz
BANDPASS_HIGH_RANGE = (6000.0, 8000.0)  # ~7 kHz
MAX_SHIFT_SECONDS = 0.15
DETUNE_CENTS_RANGE = (-40.0, 40.0)  # violao levemente desafinado; rotulo mantido


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


white_noise = pl.white_noise
pink_noise = pl.pink_noise
bandpass = pl.bandpass


def add_noise(x, rng, snr_db=None, color=None):
    """pl.add_noise com SNR e cor sorteados quando nao informados."""
    snr_db = rng.uniform(*SNR_DB_RANGE) if snr_db is None else snr_db
    color = rng.choice(['white', 'pink']) if color is None else color
    return pl.add_noise(x, rng, snr_db, color)


def random_gain(x, rng, gain_db=None):
    gain_db = rng.uniform(*GAIN_DB_RANGE) if gain_db is None else gain_db
    return np.clip(x * 10 ** (gain_db / 20), -1.0, 1.0)


def phone_mic(x, rng, sr=SR):
    """Passa-banda simulando microfone de celular (~150 Hz a ~7 kHz)."""
    return bandpass(x, rng.uniform(*BANDPASS_LOW_RANGE), rng.uniform(*BANDPASS_HIGH_RANGE), sr)


def time_shift(x, rng, sr=SR, max_shift_s=MAX_SHIFT_SECONDS):
    """Desloca o sinal dentro da janela, completando com silencio (nao circular)."""
    k = int(rng.integers(-int(max_shift_s * sr), int(max_shift_s * sr) + 1))
    if k == 0:
        return x
    out = np.zeros_like(x)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def detune(x, rng, sr=SR, cents=None):
    """Pitch shift pequeno (-40..+40 cents) MANTENDO o rotulo."""
    cents = rng.uniform(*DETUNE_CENTS_RANGE) if cents is None else cents
    y = librosa.effects.pitch_shift(np.asarray(x, dtype=np.float32), sr=sr, n_steps=cents / 100.0)
    return y.astype(np.float64)


def augment(x, rng, sr=SR, detune_rng=None):
    """Uma versao aumentada: sorteia quais augmentations aplicar (pelo menos uma).
    Ordem fisica: desafinacao -> deslocamento -> ruido ambiente -> microfone -> ganho.

    detune_rng: se informado, aplica detune() SEMPRE, usando esse gerador
    separado. Assim as demais augmentations (sorteadas com rng) ficam identicas
    com e sem detune, e a comparacao isola o efeito dele."""
    x = np.asarray(x, dtype=np.float64)
    if detune_rng is not None:
        x = detune(x, detune_rng, sr)
    flags = rng.random(4) < [P_SHIFT, P_NOISE, P_BANDPASS, P_GAIN]
    if not flags.any():
        flags[rng.integers(4)] = True
    do_shift, do_noise, do_bp, do_gain = flags
    if do_shift:
        x = time_shift(x, rng, sr)
    if do_noise:
        x = add_noise(x, rng)
    if do_bp:
        x = phone_mic(x, rng, sr)
    if do_gain:
        x = random_gain(x, rng)
    return x.astype(np.float32)


def degrade_for_test(x, rng, sr=SR):
    """Degradacao fixa do 'teste degradado' (so o ruido em si e aleatorio)."""
    x = add_noise(np.asarray(x, dtype=np.float64), rng, snr_db=15.0, color='pink')
    x = bandpass(x, 200.0, 5000.0, sr)
    return x.astype(np.float32)
