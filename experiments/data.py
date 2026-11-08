"""Montagem dos conjuntos de treino / validacao / teste dos experimentos.

- Teste: o MESMO 20% de arquivos do baseline (split_dataset_files, random_state=42),
  janelas limpas. Usado SO no evaluate() final.
- Validacao: 15% dos arquivos de TREINO (split por arquivo, estratificado,
  random_state=42), janelas limpas. Usada nos callbacks (EarlyStopping /
  ReduceLROnPlateau).
- Treino: o restante dos arquivos de treino (+ versoes aumentadas, se pedido).

As features sao extraidas em paralelo e guardadas em experiments/cache/, com o
indice do arquivo de origem de cada janela (fidx), para separar treino e
validacao sem reextrair.
"""
import hashlib
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from experiments import augment, common
from experiments.common import tm
from experiments.features import CACHE_NAMES, EXTRACTORS

VAL_SIZE = 0.15
VAL_RANDOM_STATE = 42
N_TEST_WINDOWS = 1557   # tamanho do teste do baseline; nao pode mudar
DEGRADED_SEED = 42      # o teste degradado e uma condicao FIXA (nao varia com a seed do treino)
MAX_N_AUG = 2           # o cache guarda 2 versoes; n_aug=1 usa so k=0
# cada worker importa train_model (e portanto o TensorFlow, ~0,6 GB); 28 workers
# estouravam a RAM. 8 mantem a extracao rapida sem matar processos.
N_JOBS = min(8, os.cpu_count() or 1)
MODES = ('clean', 'train_aug', 'train_aug_detune', 'test_degraded')


def split_train_val(train_files, val_size=VAL_SIZE, random_state=VAL_RANDOM_STATE):
    labels = [l for _, l in train_files]
    tr, va = train_test_split(np.arange(len(train_files)), test_size=val_size,
                              random_state=random_state, stratify=labels)
    tr, va = np.sort(tr), np.sort(va)
    return [train_files[i] for i in tr], [train_files[i] for i in va]


def _files_key(file_label_pairs):
    return np.array([f'{os.path.relpath(p, common.ROOT_DIR)}|{l}' for p, l in file_label_pairs])


def _process_file(file_idx, file_path, label, feature, mode, n_versions, seed):
    extractor = EXTRACTORS[feature]
    out = []
    for win_idx, segment in enumerate(tm.make_windowed_segments(file_path)):
        for k in range(n_versions):
            if mode == 'clean':
                audio = segment
            else:
                rng = np.random.default_rng([seed, file_idx, win_idx, k])
                if mode == 'test_degraded':
                    audio = augment.degrade_for_test(segment, rng)
                else:
                    detune_rng = (np.random.default_rng([seed, file_idx, win_idx, k, 1])
                                  if mode == 'train_aug_detune' else None)
                    audio = augment.augment(segment, rng, detune_rng=detune_rng)
            feats = extractor(audio)
            if feats is not None:
                out.append((k, file_idx, feats, label))
    return out


def build(file_label_pairs, feature, mode='clean', n_versions=1, seed=common.DEFAULT_SEED, n_jobs=N_JOBS):
    """Features de cada janela de cada arquivo. Retorna X, y, k (versao), fidx (arquivo)."""
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
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_file)(i, p, l, feature, mode, n_versions, seed)
        for i, (p, l) in enumerate(file_label_pairs)
    )
    flat = [r for file_res in results for r in file_res]
    k = np.array([r[0] for r in flat])
    fidx = np.array([r[1] for r in flat])
    X = np.array([r[2] for r in flat])
    y = np.array([r[3] for r in flat])
    os.makedirs(common.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, k=k, fidx=fidx, files_key=key)
    return X, y, k, fidx


def load_dataset(feature, n_aug=0, detune=False, seed=common.DEFAULT_SEED):
    """Retorna dict com X_train, y_train, X_val, y_val, X_test, y_test, labels
    (features no formato [n, n_features, n_frames], antes do scaler)."""
    train_files, test_files, labels = common.load_split()
    train_sub, val_files = split_train_val(train_files)

    X_tr, y_tr, _, _ = build(train_sub, feature, 'clean')
    X_va, y_va, _, _ = build(val_files, feature, 'clean')
    X_te, y_te, _, _ = build(test_files, feature, 'clean')
    assert len(X_te) == N_TEST_WINDOWS, f'teste mudou: {len(X_te)} janelas (esperado {N_TEST_WINDOWS})'
    load_degraded_test(feature)  # extrai/cacheia agora, antes de o modelo ocupar memoria

    n_clean = len(X_tr)
    if n_aug:
        mode = 'train_aug_detune' if detune else 'train_aug'
        X_a, y_a, k, _ = build(train_sub, feature, mode, n_versions=MAX_N_AUG, seed=seed)
        keep = k < n_aug
        X_tr = np.concatenate([X_tr, X_a[keep]])
        y_tr = np.concatenate([y_tr, y_a[keep]])

    print(f'  arquivos: treino={len(train_sub)} val={len(val_files)} teste={len(test_files)} | '
          f'janelas: treino={n_clean}+{len(X_tr) - n_clean} aum. val={len(X_va)} teste={len(X_te)}',
          flush=True)
    return dict(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te,
                labels=labels, n_train_files=len(train_sub), n_val_files=len(val_files))


def load_degraded_test(feature):
    _, test_files, _ = common.load_split()
    X, y, _, _ = build(test_files, feature, 'test_degraded', seed=DEGRADED_SEED)
    assert len(X) == N_TEST_WINDOWS
    return X, y
