"""Coisas que o script de corte e o montar_sources dividem: a lista de acordes,
as pastas das duas fontes e o formato do data/sources.csv.

Nao importa o pipeline.py de proposito: ele carrega o TensorFlow, e cortar um
take nao precisa disso. Os testes conferem que NONE_LABEL e SILENCE_RMS batem
com os do pipeline."""
import csv
import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Para aumentar a lista de acordes, e so mexer aqui.
ACORDES = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']
ACORDES_EXTRAS = ['D', 'A', 'E']
NONE_NO_NOME = 'none'
NONE_LABEL = '_none'

SILENCE_RMS = 0.003

DATASET_ANTIGO = os.path.join(ROOT_DIR, 'dataset')
DATASET_PROPRIO = os.path.join(ROOT_DIR, 'dataset_proprio')
DATASET_KAGGLE = os.path.join(ROOT_DIR, 'dataset_kaggle')
SOURCES_CSV = os.path.join(ROOT_DIR, 'data', 'sources.csv')
SESSOES_CSV = os.path.join(ROOT_DIR, 'data', 'sessoes.csv')

COLUNAS = ['caminho', 'acorde', 'fonte', 'instrumento', 'sessao', 'take', 'microfone', 'uso']
USOS = ('treino', 'teste')

_TAKE = re.compile(r'^(?P<acorde>[A-Za-z#]+)_(?P<instrumento>[a-z0-9]+)_(?P<sessao>[a-z0-9]+)_take(?P<n>\d{2,})$')


class ErroDeNome(ValueError):
    pass


def acordes_validos():
    return ACORDES + ACORDES_EXTRAS


def rotulo(acorde_no_nome):
    """O 'none' do nome do arquivo vira o rotulo '_none' que o pipeline usa."""
    return NONE_LABEL if acorde_no_nome == NONE_NO_NOME else acorde_no_nome


def ler_nome_do_take(caminho):
    """Am_violao_s1_take01.wav -> {'acorde': 'Am', 'instrumento': 'violao', 'sessao': 's1', 'take': 'Am_violao_s1_take01'}."""
    nome = os.path.basename(caminho)
    raiz, ext = os.path.splitext(nome)
    if ext.lower() != '.wav':
        raise ErroDeNome(f'{nome}: o take precisa ser .wav')
    m = _TAKE.match(raiz)
    if not m:
        raise ErroDeNome(
            f'{nome}: o nome precisa ser <acorde>_<instrumento>_<sessao>_take<NN>.wav, '
            'ex.: Am_violao_s1_take01.wav (instrumento e sessao em minusculas, sem acento)'
        )
    acorde = m['acorde']
    if acorde != NONE_NO_NOME and acorde not in acordes_validos():
        raise ErroDeNome(
            f'{nome}: o acorde "{acorde}" nao esta na lista ({", ".join(acordes_validos())} ou {NONE_NO_NOME}). '
            'Para aceitar um acorde novo, adicione em ACORDES_EXTRAS no scripts/fontes.py'
        )
    return {'acorde': acorde, 'instrumento': m['instrumento'], 'sessao': m['sessao'], 'take': raiz}


def caminho_relativo(caminho):
    return os.path.relpath(os.path.abspath(caminho), ROOT_DIR).replace(os.sep, '/')


def ler_sessoes(caminho=SESSOES_CSV):
    with open(caminho, newline='', encoding='utf-8') as f:
        return {linha['sessao']: linha for linha in csv.DictReader(f)}


def ler_sources(caminho=SOURCES_CSV):
    if not os.path.exists(caminho):
        return []
    with open(caminho, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def salvar_sources(linhas, caminho=SOURCES_CSV):
    """Sempre na mesma ordem (por caminho), para o diff do Git mostrar so o que mudou."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    linhas = sorted(linhas, key=lambda l: l['caminho'])
    with open(caminho, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS, lineterminator='\n')
        escritor.writeheader()
        for linha in linhas:
            escritor.writerow({c: linha.get(c, '') for c in COLUNAS})
