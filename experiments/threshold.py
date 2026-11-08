"""Curva acuracia x cobertura variando o limiar de confianca (Parte 5).

O limiar e ESCOLHIDO na VALIDACAO (janelas de acorde dos 15% de arquivos de
validacao + negativos de 'val') e os numeros sao REPORTADOS no TESTE, que nao
participa da escolha.

Simula o que o app faria com cada janela:
  1. RMS < SILENCE_THRESHOLD (0,003)        -> 'silence' (nem chega ao modelo)
  2. classe prevista == '_none'             -> 'unknown'
  3. confianca (prob. maxima) < limiar      -> 'unknown'
  4. senao                                  -> acorde aceito

Para cada limiar t em 0,30..0,95 (passo 0,05), sobre as janelas que passam pelo
filtro de silencio:
  - cobertura      = % das janelas de ACORDE que foram aceitas
  - acuracia       = % das janelas de acorde aceitas cujo acorde esta certo
  - util           = % das janelas de acorde aceitas E certas (= cobertura x acuracia)
  - falso aceite   = % das janelas NEGATIVAS aceitas como algum acorde
No teste tambem: cobertura/acuracia no teste degradado (ruido rosa 15 dB + passa-banda).

Compara o modelo com classe _none com o mesmo modelo SEM ela (Parte 4), para
ver se so o limiar ja bastaria. Media ± desvio sobre as seeds.

Alem do limiar unico (media das seeds), calcula o limiar de CADA seed na
validacao com o mesmo criterio (pipeline.choose_threshold) -- e assim que o
modelo de producao e calibrado (Parte 6) -- e reporta esses limiares no teste.

Saida em experiments/results/threshold/ (chroma_cqt) ou threshold_<feature>/:
threshold_curve.csv, threshold.png, threshold.md e suggestion.json.

Uso:  python -m experiments.threshold [--feature chroma|chroma_stft]
"""
import argparse
import hashlib
import json
import os

import joblib
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from experiments import common
import pipeline as pl
from experiments.data import N_JOBS, build, load_degraded_test, split_train_val
from experiments.negatives import NONE_LABEL, build_negatives

# feature -> (modelo com _none, mesmo modelo sem _none ou None, pasta de saida)
FEATURE_GROUPS = {
    'chroma': ('feat_chroma_scalerfeat_gapbn_none', 'feat_chroma_scalerfeat_gapbn', 'threshold'),
    'chroma_stft': ('feat_chroma_stft_scalerfeat_gapbn_none', None, 'threshold_chroma_stft'),
    'chroma_cqt_t0': ('feat_chroma_cqt_t0_scalerfeat_gapbn_none', None, 'threshold_chroma_cqt_t0'),
    'chroma_stft4096': ('feat_chroma_stft4096_scalerfeat_gapbn_none', None, 'threshold_chroma_stft4096'),
    'chroma_stft8192': ('feat_chroma_stft8192_scalerfeat_gapbn_none', None, 'threshold_chroma_stft8192'),
}
SEEDS = [42, 43, 44]
FEATURE = 'chroma'
GROUP_NONE, GROUP_PLAIN, _out = FEATURE_GROUPS[FEATURE]
THRESHOLDS = pl.THRESHOLD_GRID
OUT_DIR = os.path.join(common.RESULTS_DIR, _out)
# criterio de escolha (modelo com _none, VALIDACAO, media das seeds)
MAX_FALSE_ACCEPT = 0.05
MIN_ACCURACY = 0.98
KEYS_VAL = ['coverage', 'accuracy', 'useful', 'false_accept']
KEYS_TEST = KEYS_VAL + ['coverage_degraded', 'accuracy_degraded']


def _rms_file(path):
    return [float(np.sqrt(np.mean(np.asarray(s, dtype=np.float64) ** 2))) for s in pl.make_windowed_segments(path)]


def window_rms(files):
    """RMS de cada janela de acorde, na mesma ordem de build(files)."""
    digest = hashlib.md5('\n'.join(os.path.relpath(p, common.ROOT_DIR) for p, _ in files).encode()).hexdigest()[:10]
    cache = os.path.join(common.CACHE_DIR, f'window_rms_{digest}.npy')
    if os.path.exists(cache):
        return np.load(cache)
    rms = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(_rms_file)(p) for p, _ in files))
    np.save(cache, rms)
    return rms


def load_eval_set(split, files, degraded=False):
    X, y, _, _ = build(files, FEATURE, 'clean')
    rms = window_rms(files)
    assert len(rms) == len(X)
    gate = rms >= pl.SILENCE_THRESHOLD
    X_neg, kinds, neg_rms = build_negatives(split, files, FEATURE)
    assert (neg_rms >= pl.SILENCE_THRESHOLD).all()
    s = dict(X=X[gate], y=y[gate], X_neg=X_neg, n_total=len(X), n_silence=int((~gate).sum()),
             neg_kinds={k: int((kinds == k).sum()) for k in sorted(set(kinds))})
    if degraded:
        X_deg, y_deg = load_degraded_test(FEATURE)
        assert np.array_equal(y_deg, y)
        s['X_deg'] = X_deg[gate]
    return s


def accepted_correct(probs, classes, y_true):
    """Para cada limiar: (aceitas, aceitas_e_certas) por janela."""
    conf = probs.max(axis=1)
    pred = classes[probs.argmax(axis=1)]
    rows = []
    for t in THRESHOLDS:
        acc = (pred != NONE_LABEL) & (conf >= t)
        rows.append((acc, acc & (pred == y_true)))
    return rows


def curve_for(model_fn, classes, s):
    chords = accepted_correct(model_fn(s['X']), classes, s['y'])
    neg = accepted_correct(model_fn(s['X_neg']), classes, np.full(len(s['X_neg']), NONE_LABEL))
    deg = accepted_correct(model_fn(s['X_deg']), classes, s['y']) if 'X_deg' in s else None
    out = []
    for i in range(len(THRESHOLDS)):
        (a, c), (an, _) = chords[i], neg[i]
        row = {'threshold': float(THRESHOLDS[i]), 'coverage': a.mean(), 'accuracy': c.sum() / max(1, a.sum()), 'useful': c.mean(),
               'false_accept': an.mean()}
        if deg is not None:
            ad, cd = deg[i]
            row.update(coverage_degraded=ad.mean(), accuracy_degraded=cd.sum() / max(1, ad.sum()))
        out.append(row)
    return out


def evaluate_group(group, sets):
    """{split: [curva por seed]}"""
    res = {split: [] for split in sets}
    for seed in SEEDS:
        run_dir = os.path.join(common.RESULTS_DIR, f'{group}_s{seed}')
        metrics = json.load(open(os.path.join(run_dir, 'metrics.json'), encoding='utf-8'))
        model = tf.keras.models.load_model(os.path.join(run_dir, 'model.h5'))
        scaler = joblib.load(os.path.join(run_dir, 'scaler.joblib'))
        classes = np.array(joblib.load(os.path.join(run_dir, 'encoder.joblib')).classes_)
        kind = metrics.get('scaler', 'baseline')

        def model_fn(X):
            return model.predict(common.apply_scaler(scaler, X, kind), verbose=0)

        for split, s in sets.items():
            res[split].append(curve_for(model_fn, classes, s))
    return res


def ms(per_seed, i, key):
    v = np.array([s[i][key] for s in per_seed]) * 100
    return float(v.mean()), float(v.std(ddof=1))


def choose(per_seed_val):
    ok = [i for i in range(len(THRESHOLDS))
          if ms(per_seed_val, i, 'false_accept')[0] <= MAX_FALSE_ACCEPT * 100
          and ms(per_seed_val, i, 'accuracy')[0] >= MIN_ACCURACY * 100]
    return max(ok, key=lambda i: ms(per_seed_val, i, 'coverage')[0]) if ok else None


def per_seed_thresholds(results):
    """Limiar de cada seed escolhido na validacao (criterio de pipeline.choose_threshold) e
    as metricas dessa seed nesse limiar, na validacao e no teste."""
    out = []
    for i_seed, seed in enumerate(SEEDS):
        val_rows = results['val'][i_seed]
        t = pl.choose_threshold(val_rows, MAX_FALSE_ACCEPT, MIN_ACCURACY)
        i = int(np.argmin(np.abs(THRESHOLDS - t)))
        clean = lambda row: {k: float(v) for k, v in row.items() if k != 'threshold'}
        out.append({'seed': seed, 'limiar': float(t), 'validacao': clean(val_rows[i]),
                    'teste': clean(results['test'][i_seed][i])})
    return out


def per_seed_summary(per_seed):
    return {k: {'mean': float(np.mean([p['teste'][k] for p in per_seed])),
                'std': float(np.std([p['teste'][k] for p in per_seed], ddof=1))} for k in KEYS_TEST}


def table(per_seed, keys, best):
    names = {'coverage': 'Cobertura', 'accuracy': 'Acurácia nos aceitos', 'useful': 'Útil',
             'false_accept': 'Falso aceite', 'coverage_degraded': 'Cobertura (degr.)',
             'accuracy_degraded': 'Acurácia (degr.)'}
    lines = ['| Limiar | ' + ' | '.join(names[k] for k in keys) + ' |', '|---:|' + '---:|' * len(keys)]
    for i, t in enumerate(THRESHOLDS):
        mark = ' **←**' if i == best else ''
        lines.append(f'| {t:.2f}{mark} | ' + ' | '.join('%.1f ± %.1f' % ms(per_seed, i, k) for k in keys) + ' |')
    return lines


def plot(results, best):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [('coverage', 'Cobertura dos acordes (%)'), ('accuracy', 'Acurácia nos aceitos (%)'),
              ('false_accept', 'Falso aceite dos negativos (%)')]
    styles = {(GROUP_NONE, 'val'): ('-o', 'com _none — validação'),
              (GROUP_NONE, 'test'): ('-s', 'com _none — teste')}
    if GROUP_PLAIN:
        styles[(GROUP_PLAIN, 'test')] = ('--^', 'sem _none (Parte 4) — teste')
    for ax, (key, title) in zip(axes, panels):
        for (g, split), (style, label) in styles.items():
            m = np.array([ms(results[g][split], i, key) for i in range(len(THRESHOLDS))])
            ax.plot(THRESHOLDS, m[:, 0], style, label=label, markersize=4)
            ax.fill_between(THRESHOLDS, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], alpha=0.15)
        if best is not None:
            ax.axvline(THRESHOLDS[best], color='gray', linestyle=':', label=f'escolhido na validação ({THRESHOLDS[best]:.2f})')
        ax.set_title(title)
        ax.set_xlabel('Limiar de confiança')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'threshold.png'), dpi=150)
    plt.close(fig)


def main():
    global FEATURE, GROUP_NONE, GROUP_PLAIN, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', default='chroma', choices=sorted(FEATURE_GROUPS))
    FEATURE = parser.parse_args().feature
    GROUP_NONE, GROUP_PLAIN, out = FEATURE_GROUPS[FEATURE]
    OUT_DIR = os.path.join(common.RESULTS_DIR, out)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_files, test_files, _ = common.load_split()
    _, val_files = split_train_val(train_files)
    sets = {'val': load_eval_set('val', val_files), 'test': load_eval_set('test', test_files, degraded=True)}
    for split, s in sets.items():
        print(f'{split}: {len(s["X"])} janelas de acorde (de {s["n_total"]}; {s["n_silence"]} silencio) '
              f'+ {len(s["X_neg"])} negativas {s["neg_kinds"]}')

    results = {g: evaluate_group(g, sets) for g in (GROUP_NONE, GROUP_PLAIN) if g}
    best = choose(results[GROUP_NONE]['val'])
    seed_thresholds = per_seed_thresholds(results[GROUP_NONE])
    t_best = float(THRESHOLDS[best]) if best is not None else None

    with open(os.path.join(OUT_DIR, 'threshold_curve.csv'), 'w', encoding='utf-8') as f:
        f.write('modelo,conjunto,limiar,' + ','.join(f'{k}_mean,{k}_std' for k in KEYS_TEST) + '\n')
        for g, by_split in results.items():
            for split, per_seed in by_split.items():
                for i, t in enumerate(THRESHOLDS):
                    vals = [('%.3f,%.3f' % ms(per_seed, i, k)) if k in per_seed[0][i] else ','
                            for k in KEYS_TEST]
                    f.write(f'{g},{split},{t:.2f},' + ','.join(vals) + '\n')

    def at(group, split, keys):
        return {k: {'mean': ms(results[group][split], best, k)[0] / 100,
                    'std': ms(results[group][split], best, k)[1] / 100} for k in keys}

    suggestion = {
        'limiar': t_best,
        'escolhido_em': 'validacao',
        'criterio': f'maior cobertura com falso aceite <= {MAX_FALSE_ACCEPT:.0%} e acuracia nos aceitos >= '
                    f'{MIN_ACCURACY:.0%} (modelo {GROUP_NONE}, media das seeds {SEEDS})',
        'modelo': GROUP_NONE,
        'validacao': at(GROUP_NONE, 'val', KEYS_VAL) if best is not None else None,
        'teste': at(GROUP_NONE, 'test', KEYS_TEST) if best is not None else None,
        'teste_sem_none_mesmo_limiar': at(GROUP_PLAIN, 'test', KEYS_TEST) if best is not None and GROUP_PLAIN else None,
        'feature': FEATURE,
        'limiar_por_seed': seed_thresholds,
        'teste_limiar_por_seed': per_seed_summary(seed_thresholds),
        'conjuntos': {split: {'janelas_acorde': int(len(s['X'])), 'janelas_silencio_descartadas': s['n_silence'],
                              'negativos': s['neg_kinds']} for split, s in sets.items()},
    }
    with open(os.path.join(OUT_DIR, 'suggestion.json'), 'w', encoding='utf-8') as f:
        json.dump(suggestion, f, indent=2, ensure_ascii=False)
    plot(results, best)

    def desc(split):
        s = sets[split]
        return (f'{len(s["X"])} janelas de acorde (de {s["n_total"]}; {s["n_silence"]} com RMS < '
                f'{pl.SILENCE_THRESHOLD}, para as quais o app já devolve "silence") + {len(s["X_neg"])} negativas '
                f'({", ".join(f"{k}: {v}" for k, v in s["neg_kinds"].items())})')

    v, te = suggestion['validacao'], suggestion['teste']
    ps = suggestion['teste_limiar_por_seed']
    md = [f'# Limiar de confiança — ChordAI (`{GROUP_NONE}`)', '',
          f'- **Validação** (usada para ESCOLHER o limiar): {desc("val")}.',
          f'- **Teste** (só REPORTADO, não participa da escolha): {desc("test")}.',
          f'- Média ± desvio sobre as seeds {", ".join(map(str, SEEDS))}. Valores em %.', '',
          '"Útil" = janelas de acorde aceitas **e** certas (cobertura × acurácia). "Falso aceite" = negativas '
          'aceitas como algum acorde. Colunas "degr." = mesmo teste de acordes com ruído rosa a 15 dB + '
          'passa-banda 200 Hz–5 kHz.', '',
          f'**Limiar escolhido na validação: {t_best}** — critério: {suggestion["criterio"]}.', '',
          '| No limiar escolhido | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |',
          '|---|---:|---:|---:|---:|',
          '| Validação (escolha) | ' + ' | '.join('%.1f ± %.1f' % (v[k]['mean'] * 100, v[k]['std'] * 100)
                                               for k in KEYS_VAL) + ' |',
          '| Teste (reportado) | ' + ' | '.join('%.1f ± %.1f' % (te[k]['mean'] * 100, te[k]['std'] * 100)
                                              for k in KEYS_VAL) + ' |',
          '', f'No teste degradado, com o mesmo limiar: cobertura '
              f'{te["coverage_degraded"]["mean"] * 100:.1f}%, acurácia nos aceitos '
              f'{te["accuracy_degraded"]["mean"] * 100:.1f}%.', '',
          '**Limiar de cada seed** (escolhido na validação da própria seed, mesmo critério; é assim que o '
          'modelo de produção é calibrado): ' + ', '.join(f"seed {p['seed']}: {p['limiar']:.2f}" for p in seed_thresholds)
          + '. No teste, cada seed no seu limiar: cobertura '
          + '%.1f ± %.1f, acurácia nos aceitos %.1f ± %.1f, falso aceite %.1f ± %.1f.' % (
              ps['coverage']['mean'] * 100, ps['coverage']['std'] * 100, ps['accuracy']['mean'] * 100,
              ps['accuracy']['std'] * 100, ps['false_accept']['mean'] * 100, ps['false_accept']['std'] * 100), '',
          '![curva](threshold.png)', '',
          f'## Com classe _none (`{GROUP_NONE}`) — validação (escolha)', '',
          *table(results[GROUP_NONE]['val'], KEYS_VAL, best), '',
          f'## Com classe _none (`{GROUP_NONE}`) — teste (reportado)', '',
          *table(results[GROUP_NONE]['test'], KEYS_TEST, best), '']
    if GROUP_PLAIN:
        md += [f'## Sem classe _none (`{GROUP_PLAIN}`, Parte 4) — teste', '',
               *table(results[GROUP_PLAIN]['test'], KEYS_TEST, None), '']
    with open(os.path.join(OUT_DIR, 'threshold.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(json.dumps({'limiar': t_best, 'validacao': v, 'teste': te}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
