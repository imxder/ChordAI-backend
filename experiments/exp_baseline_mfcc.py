"""Baseline do TCC (40 MFCC + CNN 1D, hiperparametros e scaler de train_model.py)
com validacao separada. Equivale a:  python -m experiments.exp_features --feature mfcc

Uso:  python -m experiments.exp_baseline_mfcc [--seed 42]
Resultado em experiments/results/feat_mfcc_s<seed>/.

Diferenca em relacao a train_model.py: aqui a validacao (EarlyStopping /
ReduceLROnPlateau) usa 15% dos arquivos de TREINO; train_model.py usa o teste.
"""
import argparse

from experiments import common
from experiments.exp_features import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=common.DEFAULT_SEED)
    args = parser.parse_args()
    run('mfcc', 'baseline', seed=args.seed)


if __name__ == '__main__':
    main()
