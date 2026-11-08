"""Le experiments/results/*/metrics.json, agrupa as seeds de cada configuracao
(campo 'group') e grava experiments/results/comparativo.md com media ± desvio
padrao (amostral, ddof=1).

Uso:  python -m experiments.compare
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BASELINE = 'feat_mfcc'
METRICS = [('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1'), ('accuracy_degraded', 'Acc. degradado')]

# (rotulo, grupo_A, grupo_B): efeito = B - A, pareado por seed
EFFECTS = [
    ('Scaler por feature (MFCC)', 'feat_mfcc', 'feat_mfcc_scalerfeat'),
    ('Scaler por feature (chroma)', 'feat_chroma', 'feat_chroma_scalerfeat'),
    ('Chroma no lugar de MFCC (scaler do baseline)', 'feat_mfcc', 'feat_chroma'),
    ('Chroma no lugar de MFCC (scaler por feature)', 'feat_mfcc_scalerfeat', 'feat_chroma_scalerfeat'),
    ('CQT no lugar de chroma (scaler por feature)', 'feat_chroma_scalerfeat', 'feat_cqt_scalerfeat'),
    ('Augmentation (aug2, chroma)', 'feat_chroma', 'aug2_chroma'),
    ('Augmentation + detune (aug2, chroma)', 'feat_chroma', 'aug2_chroma_detune'),
    ('Detune (sobre aug2)', 'aug2_chroma', 'aug2_chroma_detune'),
    ('Total: baseline → melhor chroma sem aug', 'feat_mfcc', 'feat_chroma_scalerfeat'),
]


def load_groups():
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, '*', 'metrics.json'))):
        with open(path, encoding='utf-8') as f:
            m = json.load(f)
        if m.get('group'):  # ignora ref_* e resultados sem seed
            groups[m['group']].append(m)
    return {g: sorted(rs, key=lambda m: m['seed']) for g, rs in groups.items()}


def stats(runs, key):
    vals = np.array([r[key] for r in runs if key in r], dtype=float) * 100
    if not len(vals):
        return None
    return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)


def fmt(st, signed=False):
    if st is None:
        return '—'
    mean, std = st
    return (f'{mean:+.2f}' if signed else f'{mean:.2f}') + f' ± {std:.2f}'


def main_table(groups):
    order = sorted(groups, key=lambda g: -stats(groups[g], 'accuracy')[0])
    lines = ['| Experimento | Seeds | ' + ' | '.join(f'{t} (%)' for _, t in METRICS) + ' | Parâmetros | ms/janela |',
             '|---|---:|' + '---:|' * (len(METRICS) + 2)]
    for g in order:
        rs = groups[g]
        ms = np.mean([r['inference_ms_per_window'] for r in rs])
        lines.append(f'| {g} | {len(rs)} | ' + ' | '.join(fmt(stats(rs, k)) for k, _ in METRICS)
                     + f" | {rs[0]['n_params']:,}".replace(',', '.') + f' | {ms:.1f} |')
    return lines, order


def effects_table(groups):
    lines = ['| Efeito (B − A) | A | B | ' + ' | '.join(f'Δ {t} (p.p.)' for _, t in METRICS) + ' |',
             '|---|---|---|' + '---:|' * len(METRICS)]
    for label, a, b in EFFECTS:
        if a not in groups or b not in groups:
            continue
        by_seed_a = {r['seed']: r for r in groups[a]}
        by_seed_b = {r['seed']: r for r in groups[b]}
        seeds = sorted(set(by_seed_a) & set(by_seed_b))
        cells = []
        for key, _ in METRICS:
            diffs = [(by_seed_b[s][key] - by_seed_a[s][key]) * 100 for s in seeds
                     if key in by_seed_a[s] and key in by_seed_b[s]]
            cells.append(fmt((np.mean(diffs), np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0), signed=True)
                         if diffs else '—')
        lines.append(f'| {label} | {a} | {b} | ' + ' | '.join(cells) + ' |')
    return lines


def per_class_table(groups, order):
    classes = groups[order[0]][0]['classes']
    lines = ['| Experimento | ' + ' | '.join(classes) + ' |', '|---|' + '---:|' * len(classes)]
    for g in order:
        means = [np.mean([r['f1_per_class'][c] for r in groups[g]]) * 100 for c in classes]
        lines.append(f'| {g} | ' + ' | '.join(f'{v:.1f}' for v in means) + ' |')
    return lines


def load_confusion(run_name):
    path = os.path.join(RESULTS_DIR, run_name, 'confusion_matrix.csv')
    with open(path, encoding='utf-8') as f:
        rows = [l.strip().split(',') for l in f if l.strip()]
    labels = rows[0][1:]
    return labels, {(r[0], c): int(v) for r in rows[1:] for c, v in zip(labels, r[1:])}


def confused_pairs(labels, cm):
    """Erros somados nos dois sentidos (A->B + B->A) para cada par de acordes."""
    return {f'{a} × {b}': cm[(a, b)] + cm[(b, a)]
            for i, a in enumerate(labels) for b in labels[i + 1:]}


def mean_pairs(runs):
    all_pairs = [confused_pairs(*load_confusion(r['experiment'])) for r in runs]
    return {p: np.mean([ap[p] for ap in all_pairs]) for p in all_pairs[0]}


def confusion_section(groups, order, top=8):
    if BASELINE not in groups:
        return []
    pairs = {g: mean_pairs(groups[g]) for g in order}
    worst = sorted(pairs[BASELINE], key=lambda p: -pairs[BASELINE][p])[:top]
    cols = [BASELINE] + [g for g in order if g != BASELINE]
    lines = ['## Pares mais confundidos (média de erros A→B + B→A por seed, janelas de teste)', '',
             f'Os {top} pares mais confundidos pelo `{BASELINE}`. Valores menores = melhor.', '',
             '| Par | ' + ' | '.join(cols) + ' |', '|---|' + '---:|' * len(cols)]
    for p in worst:
        lines.append(f'| {p} | ' + ' | '.join(f'{pairs[g][p]:.1f}' for g in cols) + ' |')
    return lines + ['']


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # console do Windows (cp1252) nao tem '−'/'±'
    except Exception:
        pass
    groups = load_groups()
    if not groups:
        print(f'Nenhum metrics.json com "group" em {RESULTS_DIR}')
        return
    table, order = main_table(groups)
    seeds = sorted({r['seed'] for rs in groups.values() for r in rs})
    md = [
        '# Comparativo de experimentos — ChordAI',
        '',
        f'Média ± desvio padrão (amostral) sobre as seeds {", ".join(map(str, seeds))}. '
        'Teste: o mesmo do baseline (20% dos arquivos, split por arquivo, `random_state=42`, '
        '1545 janelas de 1,5 s), usado **só** na avaliação final. A validação dos callbacks '
        '(EarlyStopping/ReduceLROnPlateau) usa 15% dos arquivos de treino (estratificado, '
        '`random_state=42`). "Acc. degradado" = mesmo teste com ruído rosa a 15 dB + '
        'passa-banda 200 Hz–5 kHz. ms/janela = média de `model.predict` (batch 1), sem extração '
        'de features. Nomes: `_scalerfeat` = StandardScaler por feature (o do baseline normaliza '
        'por frame); `aug2` = 2 versões aumentadas por janela de treino; `_detune` = +pitch shift '
        'de ±40 cents.',
        '',
        *table,
        '',
        '## Efeitos isolados (diferença pareada por seed, em pontos percentuais)',
        '',
        *effects_table(groups),
        '',
        '## F1 por classe (%, média das seeds)',
        '',
        *per_class_table(groups, order),
        '',
        *confusion_section(groups, order),
        '---',
        '',
        '<sub>Split: `list_dataset_files()` agora ordena os arquivos (`sorted(os.listdir())`), então o '
        'teste é o mesmo em qualquer sistema operacional (SHA-256 da lista em `split_hash.txt`; '
        'conferir com `python -m experiments.split_hash --check`). O modelo '
        '`candidates/candidate_91_81` (o mesmo de `trained_model/`) não aparece na tabela: foi '
        'treinado com um split antigo, dependente da ordem do `os.listdir()`, e não é comparável. '
        'Rodadas anteriores: `experiments/results_v1_val_no_teste/` (teste usado como validação) e '
        '`experiments/results_v2_split_nao_ordenado/` (validação separada, mas split com a ordem '
        'do NTFS, que compartilha só 59 dos 297 arquivos de teste com o atual).</sub>',
        '',
    ]
    out = os.path.join(RESULTS_DIR, 'comparativo.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(table))
    print()
    print('\n'.join(effects_table(groups)))
    print(f'\nTabela salva em {out}')


if __name__ == '__main__':
    main()
