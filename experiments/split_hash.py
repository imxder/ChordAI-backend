"""Gera um hash (SHA-256) da lista de arquivos de TESTE do split por arquivo e
grava em experiments/results/split_hash.txt. Tambem verifica que o split nao
depende da ordem devolvida por os.listdir() (simula ordens diferentes, como as
de NTFS / ext4 / APFS, e confere que o hash e sempre o mesmo).

Os caminhos entram no hash relativos a dataset/ e com '/' como separador, para
o resultado ser igual em Windows, Linux e macOS.

Uso:  python -m experiments.split_hash          (grava o arquivo)
      python -m experiments.split_hash --check  (so confere contra o arquivo salvo)
"""
import hashlib
import os
import random
import sys
from unittest import mock

from experiments import common
from experiments.common import tm

OUT_PATH = os.path.join(common.RESULTS_DIR, 'split_hash.txt')


def test_file_list():
    _, test_files, _ = common.load_split()
    return sorted(os.path.relpath(p, common.DATASET_PATH).replace(os.sep, '/') + '|' + l
                  for p, l in test_files)


def split_hash():
    return hashlib.sha256('\n'.join(test_file_list()).encode('utf-8')).hexdigest()


def hash_with_listdir_order(order_fn):
    real = os.listdir
    with mock.patch.object(tm.os, 'listdir', lambda p: order_fn(real(p))):
        return split_hash()


def main():
    reference = split_hash()
    orders = {
        'os.listdir (deste sistema)': lambda xs: xs,
        'invertida': lambda xs: list(reversed(xs)),
        'sem diferenciar maiusculas (NTFS)': lambda xs: sorted(xs, key=str.lower),
        'aleatoria (seed 1)': lambda xs: random.Random(1).sample(xs, len(xs)),
        'aleatoria (seed 2)': lambda xs: random.Random(2).sample(xs, len(xs)),
    }
    ok = True
    for name, fn in orders.items():
        h = hash_with_listdir_order(fn)
        same = h == reference
        ok &= same
        print(f'  {"OK  " if same else "FALHA"} ordem {name}: {h[:16]}...')
    if not ok:
        sys.exit('O split depende da ordem do os.listdir()!')

    n_files = len(test_file_list())
    if '--check' in sys.argv:
        saved = open(OUT_PATH, encoding='utf-8').read().split()[0]
        print(f'hash atual {reference} | salvo {saved} -> {"IGUAL" if saved == reference else "DIFERENTE"}')
        sys.exit(0 if saved == reference else 1)

    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(f'{reference}\n')
        f.write(f'# SHA-256 da lista ordenada dos {n_files} arquivos de teste (caminho relativo a dataset/ + "|" + classe),\n')
        f.write('# split_dataset_files(test_size=0.2, random_state=42) com sorted(os.listdir()).\n')
        f.write('# Conferir em outra maquina: python -m experiments.split_hash --check\n')
    print(f'{reference}  ({n_files} arquivos de teste) -> {OUT_PATH}')


if __name__ == '__main__':
    main()
