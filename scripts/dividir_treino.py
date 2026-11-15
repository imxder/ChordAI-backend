"""Divide o data/sources.csv em treino, validacao e testes para a comparacao
de modelos (so meu dataset, meu + v3, com e sem os acordes extras).

Uso (so mostra as contagens):
    python -m scripts.dividir_treino --fontes proprio kaggle_v3 --extras

Regras:
- Teste "v3": os arquivos de dataset_kaggle/Test (uso = teste). As pessoas e
  instrumentos do teste do v3 nao aparecem no treino dele.
- Teste "proprio": o take02 de cada acorde das minhas sessoes cortadas (s1 e
  g1); o take01 fica no treino. Mesmo dia, comodo e microfone do treino, entao
  o numero sai otimista: serve para comparar modelos, nao para prometer acerto.
- Sem acorde (_none) gravado: TAKES_NONE_TESTE ficam no teste, o resto no treino.
- As gravacoes antigas (sessao "antiga") so entram no treino.
- Validacao (para parar o treino e calibrar o limiar), sempre a mesma em todas
  as seeds: no v3, 15% das pessoas/instrumentos (nunca a mesma pessoa no
  treino e na validacao); nos meus cortes, 1 de cada 4 cortes de cada take.
- So entram os acordes pedidos (os 8 do app, ou eles mais os extras)."""
import argparse
import os
import re

import numpy as np

from scripts import fontes

TAKE_TESTE = 'take02'
# Cordas abafadas e afinando: ficam de fora do treino para medir se o modelo
# reconhece "sem acorde" de um tipo que ele nao viu.
TAKES_NONE_TESTE = ('take04', 'take06')
VAL_FRACAO_V3 = 0.15
VAL_CADA_N_CORTES = 4
SEED_DIVISAO = 42

_CORTE = re.compile(r'_(\d{3})\.wav$')


def pessoa_v3(sessao):
    """Nome da pessoa/instrumento do v3, sem os erros de digitacao do dataset
    ('AcusticPlug29 ' com espaco, 'AcustiicPlug10', 'Retro GB_JO')."""
    return re.sub(r'\s+', '', sessao).replace('Acustiic', 'Acustic')


def numero_do_corte(caminho):
    m = _CORTE.search(caminho)
    return int(m.group(1)) if m else None


def _take_curto(linha):
    return linha['take'].rsplit('_', 1)[-1]


def dividir(linhas, fontes_usadas, acordes, raiz=fontes.ROOT_DIR):
    """Devolve {'treino', 'val', 'teste_v3', 'teste_proprio'} com listas de
    (caminho absoluto, acorde) e {'none_treino', 'none_val', 'none_teste'} com
    listas de caminhos de sem acorde gravado."""
    acordes = set(acordes)
    partes = {k: [] for k in ('treino', 'val', 'teste_v3', 'teste_proprio', 'none_treino', 'none_val', 'none_teste')}
    v3_treino = []
    for l in linhas:
        if l['fonte'] not in fontes_usadas:
            continue
        caminho = os.path.join(raiz, l['caminho'])
        if l['acorde'] == fontes.NONE_LABEL:
            if l['fonte'] != 'proprio':
                continue
            if _take_curto(l) in TAKES_NONE_TESTE:
                partes['none_teste'].append(caminho)
            elif (numero_do_corte(caminho) or 1) % VAL_CADA_N_CORTES == 0:
                partes['none_val'].append(caminho)
            else:
                partes['none_treino'].append(caminho)
            continue
        if l['acorde'] not in acordes:
            continue
        par = (caminho, l['acorde'])
        if l['fonte'] == 'kaggle_v3':
            if l['uso'] == 'teste':
                partes['teste_v3'].append(par)
            else:
                v3_treino.append((par, pessoa_v3(l['sessao'])))
        elif l['fonte'] == 'proprio':
            corte = numero_do_corte(caminho)
            if l['sessao'] == 'antiga' or corte is None:
                partes['treino'].append(par)
            elif l['uso'] == 'teste' or _take_curto(l) == TAKE_TESTE:
                partes['teste_proprio'].append(par)
            elif corte % VAL_CADA_N_CORTES == 0:
                partes['val'].append(par)
            else:
                partes['treino'].append(par)
        else:
            raise ValueError(f'fonte desconhecida: {l["fonte"]}')

    if v3_treino:
        pessoas = sorted({p for _, p in v3_treino})
        rng = np.random.default_rng(SEED_DIVISAO)
        n_val = max(1, round(VAL_FRACAO_V3 * len(pessoas)))
        val_pessoas = set(rng.choice(pessoas, size=n_val, replace=False))
        for par, pessoa in v3_treino:
            partes['val' if pessoa in val_pessoas else 'treino'].append(par)
        partes['pessoas_val_v3'] = sorted(val_pessoas)

    for k in partes:
        partes[k] = sorted(partes[k])
    return partes


def resumo(partes):
    from collections import Counter
    linhas = []
    for k in ('treino', 'val', 'teste_v3', 'teste_proprio'):
        cont = Counter(a for _, a in partes[k])
        linhas.append(f'{k}: {len(partes[k])} arquivos ' + ' '.join(f'{a}={n}' for a, n in sorted(cont.items())))
    for k in ('none_treino', 'none_val', 'none_teste'):
        linhas.append(f'{k}: {len(partes[k])} arquivos')
    if partes.get('pessoas_val_v3'):
        linhas.append(f'pessoas do v3 na validacao: {", ".join(partes["pessoas_val_v3"])}')
    return '\n'.join(linhas)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mostra a divisao treino/validacao/teste do sources.csv.')
    parser.add_argument('--fontes', nargs='+', default=['proprio', 'kaggle_v3'])
    parser.add_argument('--extras', action='store_true', help='inclui os acordes extras (D, A, E)')
    args = parser.parse_args(argv)
    acordes = fontes.acordes_validos() if args.extras else fontes.ACORDES
    print(resumo(dividir(fontes.ler_sources(), set(args.fontes), acordes)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
