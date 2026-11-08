"""Montagem dos conjuntos dos experimentos: a mesma de producao
(pipeline.load_dataset) + cache em experiments/cache/ + augmentation e teste
degradado, que so os experimentos usam.

- Teste: o MESMO 20% de arquivos do baseline (split_dataset_files, random_state=42),
  janelas limpas. Usado SO no evaluate() final.
- Validacao: 15% dos arquivos de TREINO (split por arquivo, estratificado,
  random_state=42), janelas limpas. Usada nos callbacks.
- Treino: o restante dos arquivos de treino (+ versoes aumentadas, se pedido).
"""
import hashlib
import os

import numpy as np

import pipeline as pl
from experiments import augment, common
from experiments.features import CACHE_NAMES
from experiments.negatives import build_negatives
from pipeline import N_JOBS, split_train_val  # noqa: F401

VAL_SIZE = pl.VAL_SIZE
VAL_RANDOM_STATE = pl.VAL_RANDOM_STATE
# tamanho do teste com o split ordenado (sorted(os.listdir()), hash em
# results/split_hash.txt). Era 1557 com a ordem do NTFS, antes da correcao.
N_TEST_WINDOWS = 1545
DEGRADED_SEED = 42      # o teste degradado e uma condicao FIXA (nao varia com a seed do treino)
MAX_N_AUG = 2           # o cache guarda 2 versoes; n_aug=1 usa so k=0
MODES = ('clean', 'train_aug', 'train_aug_detune', 'test_degraded')


# transformacoes por janela (funcoes de modulo, para o joblib conseguir enviar aos workers)
def _rng(seed, file_idx, win_idx, k, *extra):
    return np.random.default_rng([seed, file_idx, win_idx, k, *extra])


def transform_train_aug(segment, seed, file_idx, win_idx, k):
    return augment.augment(segment, _rng(seed, file_idx, win_idx, k))


def transform_train_aug_detune(segment, seed, file_idx, win_idx, k):
    return augment.augment(segment, _rng(seed, file_idx, win_idx, k),
                           detune_rng=_rng(seed, file_idx, win_idx, k, 1))


def transform_test_degraded(segment, seed, file_idx, win_idx, k):
    return augment.degrade_for_test(segment, _rng(seed, file_idx, win_idx, k))


TRANSFORMS = {
    'clean': None,
    'train_aug': transform_train_aug,
    'train_aug_detune': transform_train_aug_detune,
    'test_degraded': transform_test_degraded,
}


def _files_key(file_label_pairs):
    return np.array([f'{os.path.relpath(p, common.ROOT_DIR)}|{l}' for p, l in file_label_pairs])


def build(file_label_pairs, feature, mode='clean', n_versions=1, seed=common.DEFAULT_SEED, n_jobs=N_JOBS):
    """pl.build_features com cache. Retorna X, y, k (versao), fidx (arquivo)."""
    assert mode in MODES, mode
    if feature == 'mfcc+chroma':
        Xm, ym, km, fm = build(file_label_pairs, 'mfcc', mode, n_versions, seed, n_jobs)
        Xc, yc, kc, fc = build(file_label_pairs, 'chroma', mode, n_versions, seed, n_jobs)
        assert np.array_equal(ym, yc) and np.array_equal(km, kc) and np.array_equal(fm, fc)
        return np.concatenate([Xm, Xc], axis=1), ym, km, fm

    key = _files_key(file_label_pairs)
    digest = hashlib.md5('\n'.join(key).encode()).hexdigest()[:10]
    seed_tag = '' if mode == 'clean' else f'_s{seed}'
    name = f'{CACHE_NAMES[feature]}_{mode}_v{n_versions}{seed_tag}_{digest}'
    cache_path = os.path.join(common.CACHE_DIR, f'{name}.npz')
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if np.array_equal(data['files_key'], key):
            return data['X'], data['y'], data['k'], data['fidx']

    print(f'  extraindo {name} ({len(file_label_pairs)} arquivos, paralelo)...', flush=True)
    X, y, k, fidx, rms = pl.build_features(file_label_pairs, feature, TRANSFORMS[mode], n_versions, seed, n_jobs)
    os.makedirs(common.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, k=k, fidx=fidx, rms=rms, files_key=key)
    return X, y, k, fidx


def load_dataset(feature, n_aug=0, detune=False, seed=common.DEFAULT_SEED, with_none=False):
    """pl.load_dataset (o mesmo de producao) com cache, + augmentation opcional.
    Retorna dict com X_train, y_train, X_val, y_val, X_test, y_test, labels, ...
    (features no formato [n, n_features, n_frames], antes do scaler)."""
    d = pl.load_dataset(feature, with_none=with_none, dataset_path=common.DATASET_PATH,
                        build_fn=lambda files, feat: build(files, feat, 'clean'),
                        negatives_fn=build_negatives, verbose=False)
    n_test_chords = int((np.asarray(d['y_test']) != pl.NONE_LABEL).sum())
    assert n_test_chords == N_TEST_WINDOWS, f'teste mudou: {n_test_chords} janelas (esperado {N_TEST_WINDOWS})'
    load_degraded_test(feature)  # extrai/cacheia agora, antes de o modelo ocupar memoria

    n_before = len(d['X_train'])
    if n_aug:
        mode = 'train_aug_detune' if detune else 'train_aug'
        X_a, y_a, k, _ = build(d['train_files'], feature, mode, n_versions=MAX_N_AUG, seed=seed)
        keep = k < n_aug
        d['X_train'] = np.concatenate([d['X_train'], X_a[keep]])
        d['y_train'] = np.concatenate([d['y_train'], y_a[keep]])

    print(f"  arquivos: treino={d['n_train_files']} val={d['n_val_files']} teste={len(d['test_files'])} | "
          f"janelas: treino={n_before}+{len(d['X_train']) - n_before} aum. val={len(d['X_val'])} "
          f"teste={len(d['X_test'])}" + (f" | negativos: {d['none_counts']}" if with_none else ''), flush=True)
    return d


def load_degraded_test(feature):
    _, test_files, _ = common.load_split()
    X, y, _, _ = build(test_files, feature, 'test_degraded', seed=DEGRADED_SEED)
    assert len(X) == N_TEST_WINDOWS
    return X, y
