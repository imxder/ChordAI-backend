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
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    ('GAP+BN (A − feat_chroma_scalerfeat)', 'feat_chroma_scalerfeat', 'feat_chroma_scalerfeat_gapbn'),
    ('CNN 2D (B − feat_chroma_scalerfeat)', 'feat_chroma_scalerfeat', 'feat_chroma_scalerfeat_cnn2d'),
    ('CNN 2D − GAP+BN (B − A)', 'feat_chroma_scalerfeat_gapbn', 'feat_chroma_scalerfeat_cnn2d'),
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


def _lat(rs, key):
    vals = [r['latency'][key] for r in rs if 'latency' in r]
    return np.mean(vals) if vals else None


def main_table(groups):
    order = sorted(groups, key=lambda g: -stats(groups[g], 'accuracy')[0])
    lines = ['| Experimento | Seeds | ' + ' | '.join(f'{t} (%)' for _, t in METRICS)
             + ' | Parâmetros | TFLite (KB) | ms predict | ms compilado | ms TFLite |',
             '|---|---:|' + '---:|' * (len(METRICS) + 5)]
    for g in order:
        rs = groups[g]
        predict = _lat(rs, 'predict_ms') or np.mean([r['inference_ms_per_window'] for r in rs])
        extra = [_lat(rs, 'tflite_kb'), _lat(rs, 'compiled_ms'), _lat(rs, 'tflite_ms')]
        extra = ['—' if v is None else f'{v:.{p}f}' for v, p in zip(extra, (0, 2, 2))]
        lines.append(f'| {g} | {len(rs)} | ' + ' | '.join(fmt(stats(rs, k)) for k, _ in METRICS)
                     + f" | {rs[0]['n_params']:,}".replace(',', '.')
                     + f' | {extra[0]} | {predict:.1f} | {extra[1]} | {extra[2]} |')
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


def none_section(none_groups, groups):
    """Parte 5: modelos com classe _none, comparados nas MESMAS janelas de acorde."""
    if not none_groups:
        return []
    lines = ['## Parte 5 — classe "_none" (nenhum acorde)', '',
             'Accuracy dos acordes = nas mesmas 1545 janelas de acorde dos outros experimentos (prever '
             '"_none" conta como erro). Recall _none = % das janelas negativas do teste classificadas como '
             '"_none". Sem limiar de confiança (ver `threshold/threshold.md`).', '',
             '| Experimento | Seeds | Accuracy acordes (%) | Acordes → _none (%) | Recall _none (%) | '
             'Recall low_volume | Recall noise | Recall percussion | Acc. degradado (%) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g, rs in none_groups.items():
        kinds = sorted(rs[0]['none_recall_by_kind'])
        by_kind = [fmt((np.mean([r['none_recall_by_kind'][k] for r in rs]) * 100,
                        np.std([r['none_recall_by_kind'][k] for r in rs], ddof=1) * 100)) for k in kinds]
        lines.append(f'| {g} | {len(rs)} | {fmt(stats(rs, "accuracy_chords"))} | '
                     f'{fmt(stats(rs, "chords_predicted_as_none"))} | {fmt(stats(rs, "none_recall"))} | '
                     + ' | '.join(by_kind) + f' | {fmt(stats(rs, "accuracy_degraded"))} |')
    # efeito pareado: com _none - sem _none (mesma arquitetura)
    for g, rs in none_groups.items():
        base = g[:-len('_none')]
        if base in groups:
            a = {r['seed']: r for r in groups[base]}
            b = {r['seed']: r for r in rs}
            seeds = sorted(set(a) & set(b))
            d_acc = [(b[s]['accuracy_chords'] - a[s]['accuracy']) * 100 for s in seeds]
            d_deg = [(b[s]['accuracy_degraded'] - a[s]['accuracy_degraded']) * 100 for s in seeds]
            lines += ['', f'Efeito da classe _none (`{g}` − `{base}`, pareado por seed): '
                          f'Δ accuracy acordes = {fmt((np.mean(d_acc), np.std(d_acc, ddof=1)), True)} p.p.; '
                          f'Δ acc. degradado = {fmt((np.mean(d_deg), np.std(d_deg, ddof=1)), True)} p.p.']
    n = none_groups[next(iter(none_groups))][0]['none_counts']
    lines += ['', 'Negativos por split: ' + '; '.join(f'{sp}: ' + ', '.join(f'{k} {v}' for k, v in c.items())
                                                      for sp, c in n.items()), '']
    return lines


PROD_GROUP = 'feat_chroma_scalerfeat_gapbn_none'
# (rotulo, grupo, pasta do threshold.py) -- a primeira linha e a referencia (producao)
P7A_VARIANTS = [
    ('Produção: chroma_cqt (tuning automático)', PROD_GROUP, 'threshold'),
    ('chroma_stft, n_fft 2048', 'feat_chroma_stft_scalerfeat_gapbn_none', 'threshold_chroma_stft'),
    ('A: chroma_cqt, tuning=0.0', 'feat_chroma_cqt_t0_scalerfeat_gapbn_none', 'threshold_chroma_cqt_t0'),
    ('B: chroma_stft, n_fft 4096', 'feat_chroma_stft4096_scalerfeat_gapbn_none', 'threshold_chroma_stft4096'),
    ('C: chroma_stft, n_fft 8192', 'feat_chroma_stft8192_scalerfeat_gapbn_none', 'threshold_chroma_stft8192'),
]
# (rotulo, origem, chave, maior_e_melhor); origem 'm' = metrics.json (sem limiar),
# 't' = teste no limiar de cada seed (escolhido na validacao da propria seed)
P7A_METRICS = [
    ('Accuracy acordes', 'm', 'accuracy_chords', True),
    ('Macro-F1', 'm', 'macro_f1', True),
    ('Acc. degradado', 'm', 'accuracy_degraded', True),
    ('Falso aceite', 't', 'false_accept', False),
    ('Acurácia nos aceitos', 't', 'accuracy', True),
]
P7A_TOL_PP = 1.0  # "ate ~1 ponto da producao"


def _threshold_suggestion(dirname):
    path = os.path.join(RESULTS_DIR, dirname, 'suggestion.json')
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None


def _p7a_values(group, runs, thr_dir):
    """{seed: {chave: valor}} com as metricas sem limiar e as do limiar de cada seed."""
    sug = _threshold_suggestion(thr_dir)
    per_seed = {p['seed']: p for p in (sug or {}).get('limiar_por_seed', [])}
    out = {}
    for r in runs:
        v = {key: r[key] for _, src, key, _ in P7A_METRICS if src == 'm'}
        if r['seed'] in per_seed:
            v.update({key: per_seed[r['seed']]['teste'][key] for _, src, key, _ in P7A_METRICS if src == 't'})
            v['limiar'] = per_seed[r['seed']]['limiar']
        out[r['seed']] = v
    return out


def stft_section(all_groups):
    """Parte 7a: variantes de chroma na configuracao de producao (scaler por feature, GAP+BN,
    _none, sem augmentation), Δ pareado por seed contra a producao."""
    variants = [(label, g, d) for label, g, d in P7A_VARIANTS if g in all_groups]
    if len(variants) < 2 or variants[0][1] != PROD_GROUP:
        return []
    vals = {g: _p7a_values(g, all_groups[g], d) for _, g, d in variants}
    prod = vals[PROD_GROUP]

    def cell(g, key):
        xs = np.array([v[key] for v in vals[g].values() if key in v]) * 100
        return '—' if not len(xs) else f'{xs.mean():.2f} ± {xs.std(ddof=1):.2f}'

    def delta(g, key):
        seeds = [s for s in vals[g] if key in vals[g][s] and s in prod and key in prod[s]]
        d = np.array([vals[g][s][key] - prod[s][key] for s in seeds]) * 100
        return d if len(d) else None

    head = ' | '.join(f'{name} (%)' for name, *_ in P7A_METRICS)
    lines = ['## Parte 7a — chroma_stft × chroma_cqt (configuração de produção)', '',
             'Mesma configuração do modelo em produção (scaler por feature, GAP+BN, classe `_none`, validação '
             'separada, sem augmentation), mudando só a feature. Teste, seeds 42, 43, 44. Accuracy, macro-F1 e '
             'acc. degradado sem limiar (argmax); falso aceite (negativos do teste aceitos como acorde) e acurácia '
             'nos aceitos com o **limiar de cada seed**, escolhido na validação da própria seed (como o de '
             'produção). Todas as variantes saem com 65 frames.', '',
             f'| Experimento | {head} | Limiares (s42, s43, s44) |',
             '|---|' + '---:|' * len(P7A_METRICS) + '---|']
    for label, g, _ in variants:
        lims = ', '.join(f"{vals[g][s]['limiar']:.2f}" for s in sorted(vals[g]) if 'limiar' in vals[g][s]) or '—'
        lines.append(f'| {label} | ' + ' | '.join(cell(g, key) for _, _, key, _ in P7A_METRICS) + f' | {lims} |')

    lines += ['', f'Δ pareado por seed contra a produção (p.p., variante − produção). ✓ = no máximo '
                  f'{P7A_TOL_PP:.0f} p.p. pior que a produção (para falso aceite, pior = maior).', '',
              f'| Experimento | ' + ' | '.join(f'Δ {name}' for name, *_ in P7A_METRICS) + ' | Até ~1 p.p. em tudo? |',
              '|---|' + '---:|' * len(P7A_METRICS) + ':---:|']
    for label, g, _ in variants[1:]:
        cells, ok_all = [], True
        for _, _, key, higher in P7A_METRICS:
            d = delta(g, key)
            if d is None:
                cells.append('—')
                ok_all = False
                continue
            worse = -d.mean() if higher else d.mean()
            ok = worse <= P7A_TOL_PP
            ok_all &= ok
            cells.append(f'{d.mean():+.2f} ± {d.std(ddof=1):.2f}' + (' ✓' if ok else ''))
        lines.append(f'| {label} | ' + ' | '.join(cells) + f' | {"sim" if ok_all else "não"} |')
    return lines + ['']


def production_section():
    """Parte 6: o modelo em trained_model/ e o teste de equivalencia producao x experimento."""
    eq_path = os.path.join(RESULTS_DIR, 'parte6', 'equivalence.json')
    cfg_path = os.path.join(ROOT_DIR, 'trained_model', 'model_config.json')
    if not (os.path.exists(eq_path) and os.path.exists(cfg_path)):
        return []
    eq = json.load(open(eq_path, encoding='utf-8'))
    cfg = json.load(open(cfg_path, encoding='utf-8'))
    p, b = eq['production'], eq['production_breakdown']
    pct = lambda v: f'{v * 100:.2f}'
    return [
        '## Parte 6 — modelo em produção (`trained_model/`)',
        '',
        f"`{cfg['feature']}` + scaler `{cfg['scaler']}` + `{cfg['arch']}` + classe `_none`, pesos da seed "
        f"{cfg.get('seed')} (`{cfg.get('source_experiment', '')}`), **limiar de confiança "
        f"{cfg['confidence_threshold']:.2f}**. O limiar foi calibrado na validação para este modelo (o mesmo "
        f"valor que `train_model.py` calibra sozinho); o **0,45 da Parte 5 era a média das 3 seeds** "
        f"(`threshold/threshold.md`). Medido pelo caminho de produção (`load_trained_model` + "
        f"`recognize_chord`, áudio bruto) em `parte6/equivalence.json`; o caminho do experimento dá os mesmos "
        f"números (diferença 0,00 p.p.).",
        '',
        '| Janelas de acorde do teste | Janelas | % |',
        '|---|---:|---:|',
        f"| Certas | {b['correct']} | {pct(b['correct'] / b['total'])} |",
        f"| Erradas (outro acorde) | {b['wrong']} | {pct(b['wrong'] / b['total'])} |",
        f"| `unknown` (_none ou confiança < limiar) | {b['unknown']} | {pct(b['unknown'] / b['total'])} |",
        f"| `silence` (RMS < 0,003; o app trata à parte) | {b['silence']} | {pct(b['silence'] / b['total'])} |",
        f"| **Total** | {b['total']} | 100.00 |",
        '',
        f"Acurácia sem as janelas de silêncio: **{pct(b['accuracy_excluding_silence'])}%** "
        f"({b['correct']}/{b['total'] - b['silence']}); nos aceitos: {pct(p['accuracy_accepted'])}%; "
        f"cobertura: {pct(p['coverage'])}%. Negativos do teste ({p['n_negative_windows']}): "
        f"{pct(p['unknown_rate_negatives'])}% viram `unknown`, falso aceite {pct(p['false_accept'])}%.",
        '',
    ]


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # console do Windows (cp1252) nao tem '−'/'±'
    except Exception:
        pass
    all_groups = load_groups()
    none_groups = {g: rs for g, rs in all_groups.items() if rs[0].get('with_none')}
    groups = {g: rs for g, rs in all_groups.items() if g not in none_groups}
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
        'passa-banda 200 Hz–5 kHz. Latência por janela (batch 1, só o modelo, sem extração de '
        'features, CPU do PC, medida por `bench_latency.py` com todos os modelos nas mesmas '
        'condições): "ms predict" = `model.predict` (como `app.py` faz hoje, dominado pelo '
        'overhead do Keras); "ms compilado" = `tf.function(model)`; "ms TFLite" = interpretador '
        'TFLite com 1 thread (formato que iria para o celular). TFLite (KB) = modelo float32 sem '
        'quantização. Nomes: `_scalerfeat` = StandardScaler por feature (o do baseline normaliza '
        'por frame); `aug2` = 2 versões aumentadas por janela de treino; `_detune` = +pitch shift '
        'de ±40 cents; `_gapbn` = CNN 1D com BatchNorm + GlobalAveragePooling1D (variante A); '
        '`_cnn2d` = CNN 2D pequena (variante B).',
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
        *none_section(none_groups, groups),
        *stft_section(all_groups),
        *production_section(),
        '---',
        '',
        '<sub>Split: `list_dataset_files()` agora ordena os arquivos (`sorted(os.listdir())`), então o '
        'teste é o mesmo em qualquer sistema operacional (SHA-256 da lista em `split_hash.txt`; '
        'conferir com `python -m experiments.split_hash --check`). O modelo '
        '`candidates/candidate_91_81` (o que ficava em `trained_model/` antes da Parte 6) não aparece na tabela: foi '
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
