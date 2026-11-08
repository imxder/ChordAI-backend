"""Extratores de features com a mesma interface de train_model.extract_features:
entrada = segmento de audio (np.ndarray, 22050 Hz); saida = matriz
[n_features, n_frames] com padding/corte fixo em MAX_PAD_LEN frames (ou None
se o audio for curto demais).

Todos usam hop_length=512 (o padrao do librosa, o mesmo do MFCC atual), entao
uma janela de 1,5 s (33075 amostras) sempre vira MAX_PAD_LEN = 65 frames.
"""
import numpy as np
import librosa

from experiments.common import tm

HOP_LENGTH = 512
CQT_FMIN = librosa.note_to_hz('C1')  # 32,7 Hz
CQT_BINS_PER_OCTAVE = 12
CQT_N_OCTAVES = 7  # C1..B7 (~3,9 kHz), abaixo de Nyquist (11025 Hz)
CQT_N_BINS = CQT_BINS_PER_OCTAVE * CQT_N_OCTAVES  # 84


def _fix_frames(mat, max_pad_len=tm.MAX_PAD_LEN):
    if mat.shape[1] > max_pad_len:
        return mat[:, :max_pad_len]
    return np.pad(mat, ((0, 0), (0, max_pad_len - mat.shape[1])), mode='constant')


def _too_short(audio, sr):
    return len(audio) < sr * tm.MIN_VALID_DURATION


def extract_mfcc(audio, sr=tm.SAMPLE_RATE):
    """O extrator atual (40 MFCC), reaproveitado de train_model.py."""
    return tm.extract_features(audio, sample_rate=sr, is_file=False)


def extract_chroma(audio, sr=tm.SAMPLE_RATE):
    if _too_short(audio, sr):
        return None
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=HOP_LENGTH)
    return _fix_frames(chroma)


def extract_cqt(audio, sr=tm.SAMPLE_RATE):
    if _too_short(audio, sr):
        return None
    C = np.abs(librosa.cqt(y=audio, sr=sr, hop_length=HOP_LENGTH, fmin=CQT_FMIN,
                           n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE))
    return _fix_frames(librosa.amplitude_to_db(C, ref=np.max))


def extract_mfcc_chroma(audio, sr=tm.SAMPLE_RATE):
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

# nome do cache em experiments/cache/ (mfcc reaproveita o cache do baseline)
CACHE_NAMES = {
    'mfcc': 'mfcc40',
    'chroma': 'chroma12',
    'cqt': 'cqt84',
    'mfcc+chroma': 'mfcc40_chroma12',
}
