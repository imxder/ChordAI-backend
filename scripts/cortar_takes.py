"""Corta takes longos gravados com metronomo em um arquivo por batida.

Uso:
    python -m scripts.cortar_takes gravacoes/Am_violao_s1_take01.wav --conferir
    python -m scripts.cortar_takes gravacoes/                       # todos os .wav da pasta
    python -m scripts.cortar_takes gravacoes/Am_violao_s1_take01.wav --refazer

Como funciona:
- O acorde vem do nome do take (<acorde>_<instrumento>_<sessao>_take<NN>.wav):
  cada take tem um acorde so, entao o script nao tenta adivinhar.
- Acha o inicio de cada batida (onset) e corta CORTE_SEGUNDOS a partir dele.
- Onsets muito perto do primeiro (JUNTAR_SEGUNDOS) contam como a mesma
  batida: o ataque de um rasgueado ou de um arpejo rapido.
- Descarta o corte se outra batida comecar antes de CORTE_MINIMO segundos
  ("duas batidas"), se ficar abaixo do limiar de silencio do app ("silencio")
  ou se tiver menos de FRACO_RELATIVO do volume mediano das batidas do take
  ("fraco", que costuma ser um estalo no fim do som).
- Takes "sem acorde" (none_...) nao tem batida para seguir: sao cortados em
  pedacos seguidos de CORTE_SEGUNDOS, sem os que forem silencio.
- Nunca altera o take original. Os cortes vao para dataset_proprio/<acorde>/
  como <take>_<NNN>.wav e as linhas entram no data/sources.csv.
- Rodar de novo no mesmo take para e avisa; com --refazer, apaga so os cortes
  antigos daquele take e corta de novo, entao nada fica duplicado."""
import argparse
import csv
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf

from scripts import fontes

CORTE_SEGUNDOS = 2.5
CORTE_MINIMO = 2.0
PRE_ATAQUE = 0.05
JUNTAR_SEGUNDOS = 0.35
FORCA_MINIMA = 0.1
FRACO_RELATIVO = 0.15
HOP = 512


def detectar_batidas(y, sr, juntar=JUNTAR_SEGUNDOS, forca_minima=FORCA_MINIMA):
    """Inicio de cada batida, em segundos.

    Fica so com onsets que tenham pelo menos `forca_minima` da forca do maior
    onset do take (tira ruido de corda e a cauda do som) e junta os que vem
    a menos de `juntar` segundos do anterior."""
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    if envelope.max() <= 0:
        return []
    picos = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=HOP, backtrack=False)
    picos = np.array([p for p in picos if envelope[p] >= forca_minima * envelope.max()], dtype=int)
    if len(picos) == 0:
        return []
    inicios = librosa.onset.onset_backtrack(picos, envelope)
    t_picos = librosa.frames_to_time(picos, sr=sr, hop_length=HOP)
    t_inicios = librosa.frames_to_time(inicios, sr=sr, hop_length=HOP)

    # A distancia entre batidas e medida pelos picos; o recuo ate o comeco do
    # som (backtrack) so serve para o corte nao cortar o ataque.
    batidas, ultimo_pico = [], None
    for pico, inicio in zip(t_picos, t_inicios):
        if ultimo_pico is None or pico - ultimo_pico >= juntar:
            batidas.append(float(max(inicio, batidas[-1] if batidas else 0.0)))
            ultimo_pico = pico
    return batidas


def rms(trecho):
    return float(np.sqrt(np.mean(np.square(trecho)))) if len(trecho) else 0.0


def planejar_cortes(y, sr, sem_acorde=False):
    """Lista de cortes com inicio, fim, duracao e status ('ok' ou o motivo do descarte)."""
    total = len(y) / sr
    cortes = []
    if sem_acorde:
        inicio = 0.0
        while inicio + CORTE_MINIMO <= total:
            fim = min(inicio + CORTE_SEGUNDOS, total)
            cortes.append({'inicio': inicio, 'fim': fim})
            inicio = fim
    else:
        batidas = detectar_batidas(y, sr)
        for i, batida in enumerate(batidas):
            inicio = max(0.0, batida - PRE_ATAQUE)
            proxima = batidas[i + 1] if i + 1 < len(batidas) else None
            fim = min(inicio + CORTE_SEGUNDOS, total)
            if proxima is not None:
                fim = min(fim, proxima - PRE_ATAQUE)
            corte = {'inicio': inicio, 'fim': fim}
            if fim - inicio < CORTE_MINIMO:
                corte['status'] = 'duas batidas' if proxima is not None and proxima - PRE_ATAQUE < inicio + CORTE_MINIMO else 'curto'
            cortes.append(corte)

    for corte in cortes:
        corte['duracao'] = corte['fim'] - corte['inicio']
        corte['rms'] = rms(y[int(corte['inicio'] * sr):int(corte['fim'] * sr)])
        if 'status' not in corte:
            corte['status'] = 'silencio' if corte['rms'] < fontes.SILENCE_RMS else 'ok'

    # Um estalo no fim do som (cadeira, mao na corda) vira onset e passa do
    # limiar de silencio, mas e muito mais baixo que uma batida de verdade.
    if not sem_acorde:
        volumes = [c['rms'] for c in cortes if c['status'] == 'ok']
        if volumes:
            referencia = float(np.median(volumes))
            for corte in cortes:
                if corte['status'] == 'ok' and corte['rms'] < FRACO_RELATIVO * referencia:
                    corte['status'] = 'fraco'
    return cortes


def cortes_existentes(take, pasta):
    return sorted(glob.glob(os.path.join(pasta, f'{glob.escape(take)}_[0-9][0-9][0-9].wav')))


def cortar_take(caminho, conferir=False, refazer=False, destino=None, sources=None, sessoes=None):
    """Corta um take. Devolve (info do take, lista de cortes planejados).

    Em --conferir nao grava nada: so planeja, para o relatorio."""
    info = fontes.ler_nome_do_take(caminho)
    sessoes = sessoes if sessoes is not None else fontes.ler_sessoes()
    sessao = sessoes.get(info['sessao'])
    if sessao is None:
        raise fontes.ErroDeNome(f'{os.path.basename(caminho)}: a sessao "{info["sessao"]}" nao esta em data/sessoes.csv')
    if not sessao.get('microfone'):
        raise fontes.ErroDeNome(f'preencha o microfone da sessao "{info["sessao"]}" em data/sessoes.csv antes de cortar')
    if sessao.get('uso') not in fontes.USOS:
        raise fontes.ErroDeNome(f'o uso da sessao "{info["sessao"]}" em data/sessoes.csv precisa ser treino ou teste')

    y, sr = librosa.load(caminho, sr=None, mono=True)
    sem_acorde = info['acorde'] == fontes.NONE_NO_NOME
    cortes = planejar_cortes(y, sr, sem_acorde=sem_acorde)
    if conferir:
        return info, cortes

    rotulo = fontes.rotulo(info['acorde'])
    pasta = os.path.join(destino or fontes.DATASET_PROPRIO, rotulo)
    antigos = cortes_existentes(info['take'], pasta)
    if antigos and not refazer:
        raise FileExistsError(
            f'{info["take"]} ja foi cortado ({len(antigos)} arquivos em {fontes.caminho_relativo(pasta)}). '
            'Use --refazer para cortar de novo.'
        )
    for antigo in antigos:
        os.remove(antigo)

    sources_csv = sources or fontes.SOURCES_CSV
    linhas = [l for l in fontes.ler_sources(sources_csv) if not (l['fonte'] == 'proprio' and l['take'] == info['take'])]
    os.makedirs(pasta, exist_ok=True)
    subtipo = sf.info(caminho).subtype
    numero = 0
    for corte in cortes:
        if corte['status'] != 'ok':
            continue
        numero += 1
        saida = os.path.join(pasta, f'{info["take"]}_{numero:03d}.wav')
        sf.write(saida, y[int(corte['inicio'] * sr):int(corte['fim'] * sr)], sr, subtype=subtipo)
        corte['arquivo'] = saida
        linhas.append({
            'caminho': fontes.caminho_relativo(saida),
            'acorde': rotulo,
            'fonte': 'proprio',
            'instrumento': info['instrumento'],
            'sessao': info['sessao'],
            'take': info['take'],
            'microfone': sessao['microfone'],
            'uso': sessao['uso'],
        })
    fontes.salvar_sources(linhas, sources_csv)
    return info, cortes


def listar_takes(entradas):
    takes = []
    for entrada in entradas:
        if os.path.isdir(entrada):
            takes += sorted(glob.glob(os.path.join(entrada, '*.wav')))
        else:
            takes.append(entrada)
    return takes


def imprimir_relatorio(resultados, saida_csv=None):
    linhas = []
    for caminho, info, cortes in resultados:
        ok = [c for c in cortes if c['status'] == 'ok']
        descartes = {}
        for c in cortes:
            if c['status'] != 'ok':
                descartes[c['status']] = descartes.get(c['status'], 0) + 1
        duracoes = [c['duracao'] for c in ok]
        faixa = f'{min(duracoes):.2f} a {max(duracoes):.2f} s' if duracoes else '-'
        resumo = ', '.join(f'{n} {motivo}' for motivo, n in sorted(descartes.items())) or 'nenhum'
        print(f'{os.path.basename(caminho)}: {len(ok)} cortes ({faixa}); descartados: {resumo}')
        for i, c in enumerate(cortes, 1):
            linhas.append({
                'take': info['take'], 'n': i, 'inicio_s': f'{c["inicio"]:.2f}', 'fim_s': f'{c["fim"]:.2f}',
                'duracao_s': f'{c["duracao"]:.2f}', 'rms': f'{c["rms"]:.4f}', 'status': c['status'],
            })
    if saida_csv:
        with open(saida_csv, 'w', newline='', encoding='utf-8') as f:
            escritor = csv.DictWriter(f, fieldnames=list(linhas[0]) if linhas else ['take'], lineterminator='\n')
            escritor.writeheader()
            escritor.writerows(linhas)
        print(f'relatorio com cada corte em {saida_csv}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Corta takes longos em um arquivo por batida.')
    parser.add_argument('takes', nargs='+', help='arquivos .wav ou pastas com takes')
    parser.add_argument('--conferir', action='store_true', help='so mostra o relatorio, sem gravar nada')
    parser.add_argument('--refazer', action='store_true', help='apaga os cortes antigos do take e corta de novo')
    parser.add_argument('--relatorio', help='salva um CSV com cada corte (inicio, fim, duracao, status)')
    args = parser.parse_args(argv)

    takes = listar_takes(args.takes)
    if not takes:
        print('nenhum .wav encontrado')
        return 1
    try:
        for take in takes:
            fontes.ler_nome_do_take(take)
        resultados = []
        for take in takes:
            info, cortes = cortar_take(take, conferir=args.conferir, refazer=args.refazer)
            resultados.append((take, info, cortes))
    except (fontes.ErroDeNome, FileExistsError) as erro:
        print(f'parei: {erro}')
        return 1
    imprimir_relatorio(resultados, args.relatorio)
    if args.conferir:
        print('modo conferencia: nada foi gravado')
    return 0


if __name__ == '__main__':
    sys.exit(main())
