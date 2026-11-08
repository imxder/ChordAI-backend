"""Monta (em paralelo e com cache) as features das janelas aumentadas de treino
e do teste degradado. As janelas sao as mesmas de tm.make_windowed_segments."""
import os

import numpy as np
from joblib import Parallel, delayed

from experiments import augment, common
from experiments.common import tm
from experiments.features import CACHE_NAMES, EXTRACTORS


def _files_key(file_label_pairs):
    return np.array([f'{os.path.relpath(p, common.ROOT_DIR)}|{l}' for p, l in file_label_pairs])


def _process_file(file_idx, file_path, label, feature, mode, n_versions, seed):
    extractor = EXTRACTORS[feature]
    out = []
    for win_idx, segment in enumerate(tm.make_windowed_segments(file_path)):
        for k in range(n_versions):
            rng = np.random.default_rng([seed, file_idx, win_idx, k])
            if mode == 'test_degraded':
                audio = augment.degrade_for_test(segment, rng)
            else:
                detune_rng = np.random.default_rng([seed, file_idx, win_idx, k, 1]) if mode == 'train_aug_detune' else None
                audio = augment.augment(segment, rng, detune_rng=detune_rng)
            feats = extractor(audio)
            if feats is not None:
                out.append((k, feats, label))
    return out


def build(file_label_pairs, feature, mode, n_versions=1, seed=common.DEFAULT_SEED, n_jobs=-1):
    """mode='train_aug' / 'train_aug_detune' (mesmo sorteio + detune): n_versions versoes aumentadas por janela (devolve tambem
    o indice k da versao, para filtrar 1 ou 2 versoes a partir do mesmo cache).
    mode='test_degraded': 1 versao degradada por janela.
    Retorna X, y, k."""
    assert mode in ('train_aug', 'train_aug_detune', 'test_degraded')
    name = f'{CACHE_NAMES[feature]}_{mode}_v{n_versions}_s{seed}'
    cache_path = os.path.join(common.CACHE_DIR, f'{name}.npz')
    key = _files_key(file_label_pairs)
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if np.array_equal(data['files_key'], key):
            print(f'  cache: {name}')
            return data['X'], data['y'], data['k']

    print(f'  extraindo {name} ({len(file_label_pairs)} arquivos, paralelo)...')
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_file)(i, p, l, feature, mode, n_versions, seed)
        for i, (p, l) in enumerate(file_label_pairs)
    )
    flat = [r for file_res in results for r in file_res]
    k = np.array([r[0] for r in flat])
    X = np.array([r[1] for r in flat])
    y = np.array([r[2] for r in flat])
    os.makedirs(common.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, k=k, files_key=key)
    return X, y, k


def load_degraded_test(feature):
    """Features do teste degradado; 'mfcc+chroma' e montado concatenando as partes."""
    _, test_files, _ = common.load_split()
    if feature == 'mfcc+chroma':
        Xm, ym, _ = build(test_files, 'mfcc', 'test_degraded')
        Xc, yc, _ = build(test_files, 'chroma', 'test_degraded')
        assert np.array_equal(ym, yc)
        return np.concatenate([Xm, Xc], axis=1), ym
    X, y, _ = build(test_files, feature, 'test_degraded')
    return X, y
