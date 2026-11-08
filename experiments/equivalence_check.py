"""Teste de equivalencia producao x experimento (Parte 6, item 5).

PRODUCAO: train_model.load_trained_model(<candidato>) + recognize_chord() em cada
janela de teste, a partir do AUDIO BRUTO (sem cache de features), como o app faz:
  - as 1545 janelas de acorde do teste (make_windowed_segments dos arquivos de teste)
  - as 223 janelas negativas do teste (pipeline.make_negative_audio)

EXPERIMENTO: o mesmo modelo (seed 42) avaliado pelo caminho de threshold.py
(features do cache, scaler aplicado em lote, limiar do model_config.json).

As duas contagens tem que bater (diferenca <= 0,1 p.p.).

Uso:  python -m experiments.equivalence_check [pasta_do_candidato]
"""
import json
import os
import sys

import joblib
import numpy as np
import tensorflow as tf

import pipeline as pl
import train_model
from experiments import common
from experiments.threshold import load_eval_set

CANDIDATE = os.path.join(common.ROOT_DIR, 'candidates', 'candidate_chroma_gapbn_none')
SOURCE_RUN = 'feat_chroma_scalerfeat_gapbn_none_s42'
TOL_PP = 0.1


def production(model_dir, test_files):
    model, encoder, scaler = train_model.load_trained_model(model_dir)
    chords = []  # (status, chord, label)
    for i, (path, label) in enumerate(test_files):
        for seg in pl.make_windowed_segments(path):
            r = train_model.recognize_chord(seg, model, encoder, scaler)
            chords.append((r['status'], r['chord'], label))
        if (i + 1) % 50 == 0:
            print(f'  producao: {i + 1}/{len(test_files)} arquivos', flush=True)
    negs = []
    for kind, source in pl.negative_specs('test', test_files):
        r = train_model.recognize_chord(pl.make_negative_audio(kind, source), model, encoder, scaler)
        negs.append((r['status'], r['chord'], kind))
    return chords, negs, encoder.model_config_


def summarize_production(chords, negs):
    st = np.array([c[0] for c in chords])
    ok = st == 'ok'
    correct = ok & np.array([c[1] == c[2] for c in chords])
    gated = st != 'silence'
    nst = np.array([n[0] for n in negs])
    return {
        'n_chord_windows': len(chords),
        'n_silence': int((st == 'silence').sum()),
        'coverage': float(ok[gated].mean()),
        'unknown_rate_chords': float((st[gated] == 'unknown').mean()),
        'accuracy_accepted': float(correct.sum() / ok.sum()),
        'accuracy_chords_all': float(correct.sum() / len(chords)),
        'n_negative_windows': len(negs),
        'false_accept': float((nst == 'ok').mean()),
        'unknown_rate_negatives': float((nst == 'unknown').mean()),
        'n_negative_silence': int((nst == 'silence').sum()),
    }


def breakdown(chords):
    """As janelas de acorde separadas em certas / erradas / unknown / silence (o app trata
    silence a parte, entao a acuracia sem silencio e a que importa para o usuario)."""
    st = np.array([c[0] for c in chords])
    ok = st == 'ok'
    correct = ok & np.array([c[1] == c[2] for c in chords])
    n_sil = int((st == 'silence').sum())
    n = len(chords)
    return {
        'total': n, 'correct': int(correct.sum()), 'wrong': int((ok & ~correct).sum()),
        'unknown': int((st == 'unknown').sum()), 'silence': n_sil,
        'accuracy_excluding_silence': float(correct.sum() / (n - n_sil)),
        'accuracy_all_windows': float(correct.sum() / n),
    }


def experiment(threshold, test_files):
    s = load_eval_set('test', test_files)
    run_dir = os.path.join(common.RESULTS_DIR, SOURCE_RUN)
    metrics = json.load(open(os.path.join(run_dir, 'metrics.json'), encoding='utf-8'))
    model = tf.keras.models.load_model(os.path.join(run_dir, 'model.h5'))
    scaler = joblib.load(os.path.join(run_dir, 'scaler.joblib'))
    classes = np.array(joblib.load(os.path.join(run_dir, 'encoder.joblib')).classes_)
    kind = metrics['scaler']
    probs_c = model.predict(pl.apply_scaler(scaler, s['X'], kind), verbose=0)
    probs_n = model.predict(pl.apply_scaler(scaler, s['X_neg'], kind), verbose=0)
    row = pl.threshold_curve(probs_c, s['y'], probs_n, classes, thresholds=[threshold])[0]
    conf, pred = probs_c.max(axis=1), classes[probs_c.argmax(axis=1)]
    accepted = (pred != pl.NONE_LABEL) & (conf >= threshold)
    return {
        'n_chord_windows': s['n_total'],
        'n_silence': s['n_silence'],
        'coverage': row['coverage'],
        'unknown_rate_chords': 1 - row['coverage'],
        'accuracy_accepted': row['accuracy'],
        'accuracy_chords_all': float((accepted & (pred == s['y'])).sum() / s['n_total']),
        'n_negative_windows': len(s['X_neg']),
        'false_accept': row['false_accept'],
        'unknown_rate_negatives': 1 - row['false_accept'],
        'n_negative_silence': 0,
    }


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else CANDIDATE
    _, test_files, _ = common.load_split()
    print(f'producao: {model_dir}')
    chords, negs, config = production(model_dir, test_files)
    prod = summarize_production(chords, negs)
    bd = breakdown(chords)
    exp = experiment(config['confidence_threshold'], test_files)

    labels = {
        'n_chord_windows': 'janelas de acorde', 'n_silence': '  -> silence (RMS < 0,003)',
        'coverage': 'cobertura (aceitas / nao-silencio)', 'unknown_rate_chords': 'taxa de unknown nos acordes',
        'accuracy_accepted': 'acuracia nos aceitos', 'accuracy_chords_all': 'acuracia nos acordes (certas / 1545)',
        'n_negative_windows': 'janelas negativas', 'false_accept': 'falso aceite dos negativos',
        'unknown_rate_negatives': 'taxa de unknown nos negativos', 'n_negative_silence': '  -> silence',
    }
    rows, ok = [], True
    for k, name in labels.items():
        p, e = prod[k], exp[k]
        if isinstance(p, float):
            diff = (p - e) * 100
            good = abs(diff) <= TOL_PP
            rows.append((name, f'{p * 100:.2f}%', f'{e * 100:.2f}%', f'{diff:+.2f} p.p.', good))
        else:
            good = p == e
            rows.append((name, str(p), str(e), str(p - e), good))
        ok &= good
    w = max(len(r[0]) for r in rows)
    print(f"\n{'':{w}}  {'producao':>10}  {'experimento':>11}  {'diferenca':>11}")
    for name, p, e, d, good in rows:
        print(f"{name:{w}}  {p:>10}  {e:>11}  {d:>11}  {'OK' if good else 'DIFERENTE'}")
    print(f"\njanelas de acorde (producao): {bd['correct']} certas, {bd['wrong']} erradas, "
          f"{bd['unknown']} unknown, {bd['silence']} silence -> acuracia sem silence "
          f"{bd['accuracy_excluding_silence'] * 100:.2f}% ({bd['correct']}/{bd['total'] - bd['silence']})")
    print(f"\nlimiar: {config['confidence_threshold']}  |  {'EQUIVALENTE' if ok else 'NAO EQUIVALENTE'} "
          f"(tolerancia {TOL_PP} p.p.)")

    out = {'model_dir': os.path.relpath(model_dir, common.ROOT_DIR), 'threshold': config['confidence_threshold'],
           'production': prod, 'production_breakdown': bd, 'experiment_seed42': exp, 'tolerance_pp': TOL_PP, 'equivalent': bool(ok)}
    os.makedirs(os.path.join(common.RESULTS_DIR, 'parte6'), exist_ok=True)
    with open(os.path.join(common.RESULTS_DIR, 'parte6', 'equivalence.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
