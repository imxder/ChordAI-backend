"""Monta as linhas do data/sources.csv que nao vem do script de corte.

Uso:
    python -m scripts.montar_sources

1. Copia as minhas 42 gravacoes antigas (*_Alex_*) de dataset/ para
   dataset_proprio/<acorde>/, como a sessao "antiga" (so treino). E copia, nao
   move: o dataset/ continua igual, porque o modelo atual e o split_hash dele
   dependem dele ate o modelo novo ser aprovado.
2. Lista todos os arquivos de dataset_kaggle/Training e dataset_kaggle/Test
   (o Guitar Chords V3) como fonte kaggle_v3.

As linhas dos takes cortados (fonte proprio, sessoes novas) ficam como estao.
Rodar de novo nao duplica nada."""
import glob
import os
import shutil
import sys

from scripts import fontes

_INSTRUMENTO_ANTIGO = {'Acoustic': 'violao', 'Electric': 'guitarra'}
_MICROFONE_ANTIGO = {'Mic': 'microfone', 'Plugged': 'plugado'}


def ler_nome_antigo(nome):
    """Am_Alex_Acoustic_Mic_2.wav -> ('Am', 'violao', 'microfone')."""
    partes = os.path.splitext(nome)[0].split('_')
    if len(partes) < 4 or partes[1] != 'Alex':
        raise fontes.ErroDeNome(f'{nome}: nao parece uma gravacao antiga minha (<acorde>_Alex_<instrumento>_<captacao>)')
    return partes[0], _INSTRUMENTO_ANTIGO.get(partes[2], partes[2].lower()), _MICROFONE_ANTIGO.get(partes[3], partes[3].lower())


def copiar_gravacoes_antigas(origem=fontes.DATASET_ANTIGO, destino=fontes.DATASET_PROPRIO):
    linhas = []
    for caminho in sorted(glob.glob(os.path.join(origem, '*', '*_Alex_*.wav'))):
        nome = os.path.basename(caminho)
        acorde, instrumento, microfone = ler_nome_antigo(nome)
        pasta = os.path.join(destino, fontes.rotulo(acorde))
        os.makedirs(pasta, exist_ok=True)
        saida = os.path.join(pasta, nome)
        if not os.path.exists(saida):
            shutil.copy2(caminho, saida)
        linhas.append({
            'caminho': fontes.caminho_relativo(saida), 'acorde': fontes.rotulo(acorde), 'fonte': 'proprio',
            'instrumento': instrumento, 'sessao': 'antiga', 'take': os.path.splitext(nome)[0],
            'microfone': microfone, 'uso': 'treino',
        })
    return linhas


def listar_kaggle(pasta=fontes.DATASET_KAGGLE):
    """No V3 cada arquivo e uma gravacao: o 'take' e o proprio arquivo e a
    'sessao' e o violao (ex.: AcousticGuitar_RodrigoMercador), que e o que
    agrupa gravacoes parecidas."""
    linhas = []
    for subpasta, uso in (('Training', 'treino'), ('Test', 'teste')):
        for caminho in sorted(glob.glob(os.path.join(pasta, subpasta, '*', '*.wav'))):
            partes = os.path.splitext(os.path.basename(caminho))[0].split('_')
            violao = '_'.join(partes[1:-1]) or partes[0]
            linhas.append({
                'caminho': fontes.caminho_relativo(caminho), 'acorde': os.path.basename(os.path.dirname(caminho)),
                'fonte': 'kaggle_v3', 'instrumento': violao, 'sessao': violao,
                'take': os.path.splitext(os.path.basename(caminho))[0], 'microfone': 'desconhecido', 'uso': uso,
            })
    return linhas


def montar(origem=fontes.DATASET_ANTIGO, destino=fontes.DATASET_PROPRIO, kaggle=fontes.DATASET_KAGGLE,
           sources=fontes.SOURCES_CSV):
    mantidas = [l for l in fontes.ler_sources(sources)
                if not (l['fonte'] == 'kaggle_v3' or (l['fonte'] == 'proprio' and l['sessao'] == 'antiga'))]
    antigas = copiar_gravacoes_antigas(origem, destino) if os.path.isdir(origem) else []
    do_kaggle = listar_kaggle(kaggle)
    fontes.salvar_sources(mantidas + antigas + do_kaggle, sources)
    return len(antigas), len(do_kaggle), len(mantidas)


def main():
    antigas, do_kaggle, mantidas = montar()
    print(f'{antigas} gravacoes antigas em dataset_proprio/ (sessao antiga)')
    if do_kaggle:
        print(f'{do_kaggle} arquivos do Guitar Chords V3 em dataset_kaggle/')
    else:
        print('dataset_kaggle/ vazio: rode python baixar_dataset.py antes')
    print(f'{mantidas} linhas de takes cortados mantidas')
    print(f'sources.csv salvo em {fontes.caminho_relativo(fontes.SOURCES_CSV)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
