"""Atalho para experimentos com data augmentation (Parte 3).
Equivale a exp_features com --n-aug; padrao: chroma, 2 versoes por janela.

Uso:  python -m experiments.exp_augment [--feature chroma] [--n-aug 1|2] [--detune] [--seed 42]

Pitch shift com troca de rotulo NAO foi implementado: com as classes do dataset
(Am, Bb, Bdim, C, Dm, Em, F, G), +-1 semitom nunca cai em outra classe existente,
entao todas as amostras transpostas seriam descartadas. No lugar dele existe
--detune (pitch shift de -40..+40 cents MANTENDO o rotulo).
"""
import argparse

from experiments import common
from experiments.exp_features import run
from experiments.features import EXTRACTORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', default='chroma', choices=sorted(EXTRACTORS))
    parser.add_argument('--n-aug', type=int, default=2, choices=[1, 2])
    parser.add_argument('--scaler', default='baseline', choices=['baseline', 'per_feature'])
    parser.add_argument('--detune', action='store_true')
    parser.add_argument('--seed', type=int, default=common.DEFAULT_SEED)
    args = parser.parse_args()
    run(args.feature, args.scaler, args.n_aug, args.detune, args.seed)


if __name__ == '__main__':
    main()
