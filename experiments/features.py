"""Extratores de features dos experimentos -- reexportados de pipeline.py, que e
o MESMO codigo usado em producao. Mesma interface de extract_features:
entrada = segmento de audio; saida = [n_features, MAX_PAD_LEN] ou None.
"""
from pipeline import (  # noqa: F401
    CQT_BINS_PER_OCTAVE,
    CQT_FMIN,
    CQT_N_BINS,
    CQT_N_OCTAVES,
    EXTRACTORS,
    HOP_LENGTH,
    extract_chroma,
    extract_chroma_stft,
    extract_cqt,
    extract_mfcc,
    extract_mfcc_chroma,
)

# nome do cache em experiments/cache/ (mfcc reaproveita o cache do baseline)
CACHE_NAMES = {
    'mfcc': 'mfcc40',
    'chroma': 'chroma12',
    'chroma_stft': 'chroma_stft12',
    'cqt': 'cqt84',
    'mfcc+chroma': 'mfcc40_chroma12',
}
