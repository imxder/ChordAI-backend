"""Exemplos negativos da classe "_none" (Parte 5). A geracao fica em
pipeline.py (a mesma usada por train_model.py); aqui so se acrescenta o cache
em experiments/cache/. Ver pipeline.negative_specs / make_negative_audio para a
descricao dos tres tipos (low_volume, noise, percussion).
"""
import os

import numpy as np

import pipeline as pl
from experiments import common
from experiments.features import CACHE_NAMES
from pipeline import (  # noqa: F401
    LOW_VOLUME_EVERY,
    NEGATIVE_KINDS as KINDS,
    NONE_LABEL,
    N_SYNTH_NEGATIVES as N_SYNTH,
    make_negative_audio as make_audio,
    negative_specs as specs,
    synth_negative_ids as synth_ids,
)


def build_negatives(split, files, feature):
    """Features dos negativos do split (com cache). Retorna X, kinds, rms."""
    digest = pl.negative_specs_digest(split, files)
    cache_path = os.path.join(common.CACHE_DIR, f'{CACHE_NAMES[feature]}_none_{split}_{digest}.npz')
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=False)
        return d['X'], d['kinds'], d['rms']
    print(f'  gerando negativos {split} ({len(specs(split, files))} janelas)...', flush=True)
    X, kinds, rms = pl.build_negatives(split, files, feature)
    os.makedirs(common.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, X=X, kinds=kinds, rms=rms)
    return X, kinds, rms
