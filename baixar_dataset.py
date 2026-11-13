"""Baixa os datasets do Kaggle para treinar o modelo.

Uso:
    python baixar_dataset.py              # Guitar Chords V3 -> dataset_kaggle/
    python baixar_dataset.py --proprio    # meu dataset -> dataset_proprio/ (quando eu publicar)
    python baixar_dataset.py --antigo     # zip antigo do Google Drive -> dataset/ (o do modelo atual)

O Guitar Chords V3 e publico: o Kaggle entrega o zip sem login nem token.
Ele e da Fabiana Vinci (CC BY-SA 4.0), que autorizou o uso no ChordAI com
credito e link: https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3

O --antigo existe so para reproduzir o modelo que esta no app hoje, que foi
treinado com a pasta dataset/ (Training do V3 + 42 gravacoes minhas). Ele sai
quando o modelo treinado com as duas fontes for aprovado."""
import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

KAGGLE_DOWNLOAD = 'https://www.kaggle.com/api/v1/datasets/download/{slug}'
GUITAR_CHORDS_V3 = 'fabianavinci/guitar-chords-v3'
DATASET_PROPRIO = None  # ex.: 'imxder/<nome>' depois que eu publicar no Kaggle
DRIVE_ANTIGO = '15d92Uf9keWqYeFZtpAfYQAoEeSjSQwCs'


def baixar_zip(url, destino_zip):
    print(f'baixando {url}')
    with urllib.request.urlopen(url, timeout=120) as resposta, open(destino_zip, 'wb') as arquivo:
        shutil.copyfileobj(resposta, arquivo)


def extrair(zip_path, destino):
    with zipfile.ZipFile(zip_path) as z:
        nomes = [n for n in z.namelist() if not n.startswith('__MACOSX/')]
        z.extractall(destino, members=nomes)
    print(f'{len([n for n in nomes if n.lower().endswith(".wav")])} arquivos .wav em {destino}')


def baixar_do_kaggle(slug, destino):
    if os.path.isdir(destino) and os.listdir(destino):
        print(f'{destino} ja existe e nao esta vazia: apague a pasta para baixar de novo')
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, 'dataset.zip')
        baixar_zip(KAGGLE_DOWNLOAD.format(slug=slug), zip_path)
        extrair(zip_path, destino)
    return 0


def baixar_antigo():
    import gdown
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, 'dataset.zip')
        gdown.download(f'https://drive.google.com/uc?id={DRIVE_ANTIGO}', zip_path, quiet=False)
        extrair(zip_path, ROOT_DIR)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Baixa os datasets do ChordAI.')
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument('--proprio', action='store_true', help='baixa o meu dataset (depois de publicado)')
    grupo.add_argument('--antigo', action='store_true', help='baixa o zip antigo do Drive para dataset/')
    args = parser.parse_args(argv)

    if args.antigo:
        return baixar_antigo()
    if args.proprio:
        if DATASET_PROPRIO is None:
            print('meu dataset ainda nao foi publicado no Kaggle; quando publicar, preencha DATASET_PROPRIO aqui')
            return 1
        return baixar_do_kaggle(DATASET_PROPRIO, os.path.join(ROOT_DIR, 'dataset_proprio'))
    return baixar_do_kaggle(GUITAR_CHORDS_V3, os.path.join(ROOT_DIR, 'dataset_kaggle'))


if __name__ == '__main__':
    sys.exit(main())
