"""Curva acuracia x cobertura variando o limiar de confianca (Parte 5).

Simula o que o app faria com cada janela de teste:
  1. RMS < SILENCE_THRESHOLD (0,003)        -> 'silence' (nem chega ao modelo)
  2. classe prevista == '_none'             -> 'unknown'
  3. confianca (prob. maxima) < limiar      -> 'unknown'
  4. senao                                  -> acorde aceito

Para cada limiar t em 0,30..0,95 (passo 0,05), medido sobre as janelas que
passam pelo filtro de silencio:
  - cobertura      = % das janelas de ACORDE que foram aceitas
  - acuracia       = % das janelas de acorde aceitas cujo acorde esta certo
  - falso aceite   = % das janelas NEGATIVAS (_none do teste) aceitas como acorde
  - util           = % das janelas de acorde aceitas E certas (= cobertura x acuracia)
Tambem no teste degradado (so acordes: ruido rosa 15 dB + passa-banda).

Compara o modelo com classe _none com o mesmo modelo SEM ela (Parte 4), para
ver se so o limiar ja bastaria. Media ± desvio sobre as seeds.

Saida em experiments/results/threshold/: threshold_curve.csv, threshold.png,
threshold.md e suggestion.json.

Uso:  python -m experiments.threshold
"""
import json
import os

import joblib
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from experiments import common
from experiments.common import tm
from experiments.data import N_JOBS, build, load_degraded_test
from experiments.negatives import NONE_LABEL, build_negatives

GROUP_NONE = 'feat_chroma_scalerfeat_gapbn_none'
GROUP_PLAIN = 'feat_chroma_scalerfeat_gapbn'
SEEDS = [42, 43, 44]
FEATURE = 'chroma'
THRESHOLDS = np.round(np.arange(0.30, 0.951, 0.05), 2)
OUT_DIR = os.path.join(common.RESULTS_DIR, 'threshold')
# criterio de sugestao (modelo com _none, teste limpo, media das seeds)
MAX_FALSE_ACCEPT = 0.05
MIN_ACCURACY = 0.98


def _rms_file(path):
    return [float(np.sqrt(np.mean(np.asarray(s, dtype=np.float64) ** 2))) for s in tm.make_windowed_segments(path)]


def test_chord_rms(test_files):
    """RMS de cada janela de acorde do teste, na mesma ordem de build(test_files)."""
    cache = os.path.join(common.CACHE_DIR, 'test_chord_rms.npy')
    if os.path.exists(cache):
        return np.load(cache)
    rms = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(_rms_file)(p) for p, _ in test_files))
    np.save(cache, rms)
    return rms


def curve(probs, classes, y_true):
    """Para cada limiar: (aceitas, corretas) por janela."""
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    pred = classes[pred_idx]
    is_chord_pred = pred != NONE_LABEL
    rows = []
    for t in THRESHOLDS:
        accepted = is_chord_pred & (conf >= t)
        rows.append((accepted, accepted & (pred == y_true)))
    return rows


def summarize(chord_rows, neg_rows, deg_rows):
    out = []
    for (acc_c, cor_c), (acc_n, _), (acc_d, cor_d) in zip(chord_rows, neg_rows, deg_rows):
        out.append({
            'coverage': acc_c.mean(),
            'accuracy': cor_c.sum() / max(1, acc_c.sum()),
            'useful': cor_c.mean(),
            'false_accept': acc_n.mean(),
            'coverage_degraded': acc_d.mean(),
            'accuracy_degraded': cor_d.sum() / max(1, acc_d.sum()),
        })
    return out


def evaluate_group(group, X_chords, y_chords, X_neg, X_deg, gate):
    per_seed = []
    for seed in SEEDS:
        run_dir = os.path.join(common.RESULTS_DIR, f'{group}_s{seed}')
        metrics = json.load(open(os.path.join(run_dir, 'metrics.json'), encoding='utf-8'))
        model = tf.keras.models.load_model(os.path.join(run_dir, 'model.h5'))
        scaler = joblib.load(os.path.join(run_dir, 'scaler.joblib'))
        encoder = joblib.load(os.path.join(run_dir, 'encoder.joblib'))
        classes = np.array(encoder.classes_)
        kind = metrics.get('scaler', 'baseline')

        def probs(X):
            return model.predict(common.apply_scaler(scaler, X, kind), verbose=0)

        chord_rows = curve(probs(X_chords[gate]), classes, y_chords[gate])
        neg_rows = curve(probs(X_neg), classes, np.full(len(X_neg), NONE_LABEL))
        deg_rows = curve(probs(X_deg[gate]), classes, y_chords[gate])
        per_seed.append(summarize(chord_rows, neg_rows, deg_rows))
    return per_seed


def mean_std(per_seed, i, key):
    v = np.array([s[i][key] for s in per_seed]) * 100
    return v.mean(), v.std(ddof=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _, test_files, _ = common.load_split()
    X_chords, y_chords, _, _ = build(test_files, FEATURE, 'clean')
    X_deg, y_deg = load_degraded_test(FEATURE)
    assert np.array_equal(y_chords, y_deg)
    X_neg, neg_kinds, neg_rms = build_negatives('test', test_files, FEATURE)
    rms = test_chord_rms(test_files)
    assert len(rms) == len(X_chords)
    gate = rms >= tm.SILENCE_THRESHOLD
    assert (neg_rms >= tm.SILENCE_THRESHOLD).all()
    print(f'teste: {len(X_chords)} janelas de acorde ({(~gate).sum()} abaixo do limiar de silencio, '
          f'descartadas como o app faz) + {len(X_neg)} negativas')

    results = {g: evaluate_group(g, X_chords, y_chords, X_neg, X_deg, gate) for g in (GROUP_NONE, GROUP_PLAIN)}

    # CSV com media e desvio
    keys = ['coverage', 'accuracy', 'useful', 'false_accept', 'coverage_degraded', 'accuracy_degraded']
    with open(os.path.join(OUT_DIR, 'threshold_curve.csv'), 'w', encoding='utf-8') as f:
        f.write('modelo,limiar,' + ','.join(f'{k}_mean,{k}_std' for k in keys) + '\n')
        for g, per_seed in results.items():
            for i, t in enumerate(THRESHOLDS):
                f.write(f'{g},{t:.2f},' + ','.join('%.3f,%.3f' % mean_std(per_seed, i, k) for k in keys) + '\n')

    # sugestao
    ps = results[GROUP_NONE]
    ok = [i for i in range(len(THRESHOLDS))
          if mean_std(ps, i, 'false_accept')[0] <= MAX_FALSE_ACCEPT * 100
          and mean_std(ps, i, 'accuracy')[0] >= MIN_ACCURACY * 100]
    best = max(ok, key=lambda i: mean_std(ps, i, 'coverage')[0]) if ok else None
    suggestion = {
        'criterio': f'maior cobertura com falso aceite <= {MAX_FALSE_ACCEPT:.0%} e acuracia >= {MIN_ACCURACY:.0%} '
                    f'(modelo {GROUP_NONE}, teste limpo, media das seeds)',
        'limiar': float(THRESHOLDS[best]) if best is not None else None,
        **({k: mean_std(ps, best, k)[0] / 100 for k in keys} if best is not None else {}),
        'n_chord_windows_gated': int(gate.sum()),
        'n_chord_windows_silence': int((~gate).sum()),
        'n_negative_windows': int(len(X_neg)),
        'negative_kinds': {k: int((neg_kinds == k).sum()) for k in sorted(set(neg_kinds))},
    }
    json.dump(suggestion, open(os.path.join(OUT_DIR, 'suggestion.json'), 'w', encoding='utf-8'), indent=2,
              ensure_ascii=False)

    # grafico
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [('coverage', 'Cobertura dos acordes (%)'), ('accuracy', 'Acurácia nos aceitos (%)'),
              ('false_accept', 'Falso aceite dos negativos (%)')]
    for ax, (key, title) in zip(axes, panels):
        for g, style in ((GROUP_NONE, '-o'), (GROUP_PLAIN, '--s')):
            m = np.array([mean_std(results[g], i, key) for i in range(len(THRESHOLDS))])
            label = 'com _none' if g == GROUP_NONE else 'sem _none (Parte 4)'
            ax.plot(THRESHOLDS, m[:, 0], style, label=label, markersize=4)
            ax.fill_between(THRESHOLDS, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], alpha=0.2)
        if best is not None:
            ax.axvline(THRESHOLDS[best], color='gray', linestyle=':', label=f'sugerido ({THRESHOLDS[best]:.2f})')
        ax.set_title(title)
        ax.set_xlabel('Limiar de confiança')
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'threshold.png'), dpi=150)
    plt.close(fig)

    # markdown
    def table(g):
        lines = ['| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite | Cobertura (degr.) | Acurácia (degr.) |',
                 '|---:|---:|---:|---:|---:|---:|---:|']
        for i, t in enumerate(THRESHOLDS):
            mark = ' **←**' if g == GROUP_NONE and i == best else ''
            lines.append(f'| {t:.2f}{mark} | ' + ' | '.join('%.1f ± %.1f' % mean_std(results[g], i, k) for k in keys) + ' |')
        return lines

    md = ['# Limiar de confiança — ChordAI (Parte 5)', '',
          f'Teste: {gate.sum()} janelas de acorde que passam pelo filtro de silêncio (de {len(X_chords)}; '
          f'{(~gate).sum()} têm RMS < {tm.SILENCE_THRESHOLD} e o app já devolve "silence") + {len(X_neg)} janelas '
          f'negativas ({", ".join(f"{k}: {v}" for k, v in suggestion["negative_kinds"].items())}). '
          f'Média ± desvio sobre as seeds {", ".join(map(str, SEEDS))}. Valores em %.', '',
          '"Útil" = janelas de acorde aceitas **e** certas (cobertura × acurácia). "Falso aceite" = negativas '
          'aceitas como algum acorde. Colunas "degr." = mesmo teste de acordes com ruído rosa a 15 dB + '
          'passa-banda 200 Hz–5 kHz.', '',
          f'**Sugestão:** {suggestion["criterio"]} → limiar **{suggestion["limiar"]}**.', '',
          '![curva](threshold.png)', '',
          f'## Com classe _none (`{GROUP_NONE}`)', '', *table(GROUP_NONE), '',
          f'## Sem classe _none (`{GROUP_PLAIN}`, Parte 4)', '', *table(GROUP_PLAIN), '']
    open(os.path.join(OUT_DIR, 'threshold.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print(json.dumps(suggestion, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
