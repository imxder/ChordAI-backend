"""Le experiments/results/*/metrics.json, imprime a tabela comparativa e grava
experiments/results/comparativo.md.

Uso:  python -m experiments.compare
"""
import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BASELINE = 'baseline_mfcc'
COLUMNS = ['Experimento', 'Accuracy', 'Macro-F1', 'Acc. degradado', 'Parâmetros', 'ms/janela']


def load_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, '*', 'metrics.json'))):
        with open(path, encoding='utf-8') as f:
            m = json.load(f)
        rows.append(m | {'experiment': m.get('experiment') or os.path.basename(os.path.dirname(path))})
    return sorted(rows, key=lambda m: m['accuracy'], reverse=True)


def to_cells(m):
    return [
        m['experiment'],
        f"{m['accuracy'] * 100:.2f}%",
        f"{m['macro_f1'] * 100:.2f}%",
        f"{m['accuracy_degraded'] * 100:.2f}%" if 'accuracy_degraded' in m else '—',
        f"{m['n_params']:,}".replace(',', '.'),
        f"{m['inference_ms_per_window']:.2f}",
    ]


def markdown_table(rows):
    lines = ['| ' + ' | '.join(COLUMNS) + ' |',
             '|' + '|'.join(['---'] + ['---:'] * (len(COLUMNS) - 1)) + '|']
    lines += ['| ' + ' | '.join(to_cells(m)) + ' |' for m in rows]
    return '\n'.join(lines)


def per_class_table(rows):
    classes = rows[0]['classes']
    lines = ['| Experimento | ' + ' | '.join(classes) + ' |',
             '|---|' + '|'.join(['---:'] * len(classes)) + '|']
    for m in rows:
        f1 = m['f1_per_class']
        lines.append(f"| {m['experiment']} | " + ' | '.join(f'{f1.get(c, 0) * 100:.1f}' for c in classes) + ' |')
    return '\n'.join(lines)


def load_confusion(experiment):
    path = os.path.join(RESULTS_DIR, experiment, 'confusion_matrix.csv')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        lines = [l.strip().split(',') for l in f if l.strip()]
    labels = lines[0][1:]
    cm = {(r[0], c): int(v) for r in lines[1:] for c, v in zip(labels, r[1:])}
    return labels, cm


def confused_pairs(labels, cm):
    """Erros somados nos dois sentidos (A->B + B->A) para cada par de acordes."""
    pairs = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            pairs[f'{a} × {b}'] = cm[(a, b)] + cm[(b, a)]
    return pairs


def confusion_section(rows, reference=BASELINE, top=8):
    ref = load_confusion(reference)
    if ref is None:
        return []
    ref_pairs = confused_pairs(*ref)
    worst = [p for p, _ in sorted(ref_pairs.items(), key=lambda kv: -kv[1])[:top]]
    # ref_* sao modelos pre-treinados com outro split (ver Parte 1); ficam de fora
    others = [m['experiment'] for m in rows if m['experiment'] != reference
              and not m['experiment'].startswith('ref_')
              and load_confusion(m['experiment']) is not None]
    exp_pairs = {e: confused_pairs(*load_confusion(e)) for e in others}

    lines = [f'## Pares mais confundidos (erros A→B + B→A, janelas de teste)', '',
             f'Os {top} pares mais confundidos pelo `{reference}`. Valores menores = melhor.', '',
             '| Par | ' + ' | '.join([reference] + others) + ' |',
             '|---|' + '|'.join(['---:'] * (1 + len(others))) + '|']
    for p in worst:
        lines.append('| ' + ' | '.join([p, str(ref_pairs[p])] + [str(exp_pairs[e][p]) for e in others]) + ' |')
    return lines + ['']


def main():
    rows = load_all()
    if not rows:
        print(f'Nenhum metrics.json em {RESULTS_DIR}')
        return

    widths = [max(len(COLUMNS[i]), *(len(to_cells(m)[i]) for m in rows)) for i in range(len(COLUMNS))]
    print('  '.join(c.ljust(w) for c, w in zip(COLUMNS, widths)))
    print('  '.join('-' * w for w in widths))
    for m in rows:
        print('  '.join(c.ljust(w) for c, w in zip(to_cells(m), widths)))

    md = [
        '# Comparativo de experimentos — ChordAI',
        '',
        'Todos avaliados no mesmo conjunto de teste (split por arquivo, 80/20, '
        '`random_state=42`), em janelas de 1,5 s. Tempo = média de `model.predict` '
        'para uma janela (batch 1), sem extração de features. "Acc. degradado" = mesmo '
        'teste com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz (robustez a ruído/microfone).',
        '',
        markdown_table(rows),
        '',
        '## F1 por classe (%)',
        '',
        per_class_table(rows),
        '',
    ] + confusion_section(rows)
    out = os.path.join(RESULTS_DIR, 'comparativo.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f'\nTabela salva em {out}')


if __name__ == '__main__':
    main()
