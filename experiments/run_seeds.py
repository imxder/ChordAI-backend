"""Roda todas as configuracoes da revisao com as seeds 42, 43 e 44, uma por vez
(processo separado para cada, para o TF comecar limpo e o tempo medido nao ser
afetado por outra execucao). Pula o que ja tem metrics.json, entao pode ser
interrompido e retomado. No fim regenera o comparativo.

Uso:  python -m experiments.run_seeds [--only feat_mfcc feat_chroma ...]
"""
import argparse
import os
import subprocess
import sys

from experiments import common
from experiments.exp_features import group_name

SEEDS = [42, 43, 44]
# (feature, scaler, n_aug, detune, arch)
CONFIGS = [
    ('mfcc', 'baseline', 0, False, 'baseline'),
    ('mfcc', 'per_feature', 0, False, 'baseline'),
    ('chroma', 'baseline', 0, False, 'baseline'),
    ('chroma', 'per_feature', 0, False, 'baseline'),
    ('cqt', 'per_feature', 0, False, 'baseline'),
    ('chroma', 'baseline', 2, False, 'baseline'),
    ('chroma', 'baseline', 2, True, 'baseline'),
    # Parte 4: arquitetura
    ('chroma', 'per_feature', 0, False, 'gap_bn'),
    ('chroma', 'per_feature', 0, False, 'cnn2d'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', help='grupos a rodar (ex.: feat_mfcc aug2_chroma)')
    args = parser.parse_args()
    known = [group_name(*c) for c in CONFIGS]
    unknown = set(args.only or []) - set(known)
    if unknown:
        sys.exit(f'grupos desconhecidos: {sorted(unknown)}; opcoes: {known}')

    # o split tem que ser o registrado em results/split_hash.txt
    if subprocess.call([sys.executable, '-W', 'ignore', '-m', 'experiments.split_hash', '--check'],
                       cwd=common.ROOT_DIR) != 0:
        sys.exit('split diferente do registrado em results/split_hash.txt')

    for feature, scaler, n_aug, detune, arch in CONFIGS:
        group = group_name(feature, scaler, n_aug, detune, arch)
        if args.only and group not in args.only:
            continue
        for seed in SEEDS:
            name = f'{group}_s{seed}'
            if os.path.exists(os.path.join(common.RESULTS_DIR, name, 'metrics.json')):
                print(f'[pula] {name}', flush=True)
                continue
            cmd = [sys.executable, '-W', 'ignore', '-m', 'experiments.exp_features',
                   '--feature', feature, '--scaler', scaler, '--n-aug', str(n_aug),
                   '--arch', arch, '--seed', str(seed)]
            if detune:
                cmd.append('--detune')
            log = os.path.join(common.RESULTS_DIR, f'{name}.log')
            os.makedirs(common.RESULTS_DIR, exist_ok=True)
            with open(log, 'w', encoding='utf-8') as f:
                code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=common.ROOT_DIR)
            print(f'[{"ok" if code == 0 else f"ERRO {code}"}] {name}', flush=True)
            if code != 0:
                sys.exit(code)
            os.replace(log, os.path.join(common.RESULTS_DIR, name, 'train.log'))
    subprocess.call([sys.executable, '-m', 'experiments.compare'], cwd=common.ROOT_DIR)


if __name__ == '__main__':
    main()
