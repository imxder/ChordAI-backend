"""Corta takes longos gravados com metronomo em um arquivo por batida.

Uso:
    python -m scripts.cortar_takes gravacoes/Am_violao_s1_take01.wav --conferir
    python -m scripts.cortar_takes ~/Downloads/Am_violao_s1_take01.m4a   # direto do iPhone
    python -m scripts.cortar_takes gravacoes/                       # todos os .wav e .m4a da pasta
    python -m scripts.cortar_takes gravacoes/Am_violao_s1_take01.wav --refazer

Como funciona:
- O acorde vem do nome do take (<acorde>_<instrumento>_<sessao>_take<NN>.wav):
  cada take tem um acorde so, entao o script nao tenta adivinhar.
- Acha o inicio de cada batida (onset) e corta CORTE_SEGUNDOS a partir dele.
- Onsets a menos de JUNTAR_SEGUNDOS do onset anterior contam como a mesma
  batida: as cordas de um rasgueado ou de um arpejo vem em sequencia, e o
  corte comeca no primeiro onset da sequencia.
- Ignora onsets em que o volume nao sobe pelo menos SUBIDA_MINIMA vezes
  logo depois: sao a cauda do acorde ou ruido, nao uma batida nova. A guitarra
  limpa sustenta mais que o violao e a batida sobe menos, entao o limite
  depende do instrumento (SUBIDA_MINIMA_POR_INSTRUMENTO).
- Descarta o corte se outra batida comecar antes de CORTE_MINIMO segundos
  ("duas batidas"), se ficar abaixo do limiar de silencio do app ("silencio")
  ou se tiver menos de FRACO_RELATIVO do volume mediano das batidas do take
  ("fraco", que costuma ser um estalo no fim do som).
- Takes "sem acorde" (none_...) nao tem batida para seguir: sao cortados em
  pedacos seguidos de CORTE_SEGUNDOS, sem os que forem silencio.
- Take em .m4a (Gravador do iPhone) e convertido com o afconvert do macOS
  para gravacoes/<take>.wav, mantendo a taxa de amostragem, e o corte sai
  desse .wav. Se o .wav ja existir, para e avisa (com --refazer, converte de
  novo). Em --conferir a conversao vai para uma pasta temporaria.
- Nunca altera o take original. Os cortes vao para dataset_proprio/<acorde>/
  como <take>_<NNN>.wav e as linhas entram no data/sources.csv.
- Rodar de novo no mesmo take para e avisa; com --refazer, apaga so os cortes
  antigos daquele take e corta de novo, entao nada fica duplicado."""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile

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
SUBIDA_MINIMA = 3.5
# Faixa que deu 8 cortes no lugar certo em todos os takes, com SUBIDA_ATE 0,25:
# violao s1 (iPhone): so 3,5 (3,0 e 4,0 ja erram); Jazzmaster g1 (GT-1): 1,6 a 2,5.
SUBIDA_MINIMA_POR_INSTRUMENTO = {'jazzmaster': 2.0}
SUBIDA_JANELA = 0.2
SUBIDA_ATE = 0.25
HOP = 512


def detectar_batidas(y, sr, juntar=JUNTAR_SEGUNDOS, forca_minima=FORCA_MINIMA, subida_minima=None):
    """Inicio de cada batida, em segundos.

    Fica so com onsets que tenham pelo menos `forca_minima` da forca do maior
    onset do take (tira ruido de corda e a cauda do som) e junta os que vem
    a menos de `juntar` segundos do onset anterior, em cadeia: um rasgueado
    espalhado tem varios onsets seguidos e vira uma batida so, que comeca
    no primeiro."""
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

    # O filtro de subida vem antes de juntar: a cauda de um acorde pode ter
    # onsets fracos a cada 0,3 s, e se eles entrassem na cadeia ela iria ate a
    # batida seguinte e a batida inteira seria descartada junto com a cauda.
    onsets = [(pico, inicio) for pico, inicio in zip(t_picos, t_inicios)
              if subida(y, sr, float(inicio)) >= (SUBIDA_MINIMA if subida_minima is None else subida_minima)]

    # A distancia entre batidas e medida pelos picos; o recuo ate o comeco do
    # som (backtrack) so serve para o corte nao cortar o ataque.
    batidas, ultimo_pico = [], None
    for pico, inicio in onsets:
        if ultimo_pico is None or pico - ultimo_pico >= juntar:
            batidas.append(float(max(inicio, batidas[-1] if batidas else 0.0)))
        ultimo_pico = pico
    return batidas


def subida(y, sr, t):
    """Quanto o volume sobe logo depois de `t`: o maior RMS numa janela de
    SUBIDA_JANELA comecando ate SUBIDA_ATE - SUBIDA_JANELA segundos depois,
    dividido pelo RMS da janela logo antes. Nos 22 takes de acorde da s1
    (iPhone), com SUBIDA_MINIMA em 3,5, saem as 8 batidas de cada take e os
    onsets na cauda do som (zumbido de corda, ruido) ficam de fora. A janela
    e curta de proposito: olhando mais longe, um onset de cauda ate 1 s antes
    da batida seguinte "via" a subida dela e o corte comecava cedo demais.
    """
    def r(a, z):
        return rms(y[max(0, int(a * sr)):max(0, int(z * sr))])
    passo = 0.05
    depois = max(r(t + k * passo, t + k * passo + SUBIDA_JANELA)
                 for k in range(int(round((SUBIDA_ATE - SUBIDA_JANELA) / passo)) + 1))
    return depois / max(r(t - SUBIDA_JANELA, t), 1e-6)


def rms(trecho):
    return float(np.sqrt(np.mean(np.square(trecho)))) if len(trecho) else 0.0


def planejar_cortes(y, sr, sem_acorde=False, subida_minima=None):
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
        batidas = detectar_batidas(y, sr, subida_minima=subida_minima)
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


class ErroDeConversao(RuntimeError):
    pass


def converter_para_wav(caminho, destino):
    """Converte um .m4a para .wav PCM 24 bits, mesma taxa e mesmos canais."""
    if shutil.which('afconvert') is None:
        raise ErroDeConversao('nao achei o afconvert (vem com o macOS) para converter o .m4a')
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    temporario = destino + '.convertendo'
    resultado = subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI24', caminho, temporario],
                               capture_output=True, text=True)
    if resultado.returncode != 0:
        if os.path.exists(temporario):
            os.remove(temporario)
        raise ErroDeConversao(f'{os.path.basename(caminho)}: o afconvert falhou: {resultado.stderr.strip()}')
    os.replace(temporario, destino)
    return destino


def preparar_take(caminho, conferir=False, refazer=False, pasta_wav=None, pasta_temporaria=None):
    """Devolve o .wav que vai ser cortado. Um .wav passa direto; um .m4a e
    convertido para <pasta_wav>/<take>.wav (ou para a pasta temporaria em --conferir)."""
    if os.path.splitext(caminho)[1].lower() == '.wav':
        return caminho
    take = fontes.ler_nome_do_take(caminho)['take']
    if conferir:
        return converter_para_wav(caminho, os.path.join(pasta_temporaria or tempfile.mkdtemp(), take + '.wav'))
    destino = os.path.join(pasta_wav or fontes.GRAVACOES, take + '.wav')
    if os.path.exists(destino) and not refazer:
        raise FileExistsError(
            f'{fontes.caminho_relativo(destino)} ja existe. Use --refazer para converter o .m4a de novo '
            'por cima dele, ou corte o .wav direto.'
        )
    return converter_para_wav(caminho, destino)


def cortes_existentes(take, pasta):
    return sorted(glob.glob(os.path.join(pasta, f'{glob.escape(take)}_[0-9][0-9][0-9].wav')))


def cortar_take(caminho, conferir=False, refazer=False, destino=None, sources=None, sessoes=None,
                pasta_wav=None, pasta_temporaria=None):
    """Corta um take (.wav ou .m4a). Devolve (info do take, lista de cortes planejados).

    Em --conferir nao grava nada no projeto: so planeja, para o relatorio."""
    info = fontes.ler_nome_do_take(caminho)
    sessoes = sessoes if sessoes is not None else fontes.ler_sessoes()
    sessao = sessoes.get(info['sessao'])
    if sessao is None:
        raise fontes.ErroDeNome(f'{os.path.basename(caminho)}: a sessao "{info["sessao"]}" nao esta em data/sessoes.csv')
    if not sessao.get('microfone'):
        raise fontes.ErroDeNome(f'preencha o microfone da sessao "{info["sessao"]}" em data/sessoes.csv antes de cortar')
    if sessao.get('uso') not in fontes.USOS:
        raise fontes.ErroDeNome(f'o uso da sessao "{info["sessao"]}" em data/sessoes.csv precisa ser treino ou teste')

    caminho = preparar_take(caminho, conferir=conferir, refazer=refazer,
                            pasta_wav=pasta_wav, pasta_temporaria=pasta_temporaria)
    y, sr = librosa.load(caminho, sr=None, mono=True)
    sem_acorde = info['acorde'] == fontes.NONE_NO_NOME
    cortes = planejar_cortes(y, sr, sem_acorde=sem_acorde,
                             subida_minima=SUBIDA_MINIMA_POR_INSTRUMENTO.get(info['instrumento']))
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
            # Depois de converter, o .m4a e o .wav do mesmo take ficam lado a
            # lado na pasta: o .wav e o convertido, entao o .m4a fica de fora.
            arquivos = sorted(c for ext in fontes.EXTENSOES_TAKE for c in glob.glob(os.path.join(entrada, '*' + ext)))
            wavs = {os.path.splitext(c)[0] for c in arquivos if c.lower().endswith('.wav')}
            takes += [c for c in arquivos if c.lower().endswith('.wav') or os.path.splitext(c)[0] not in wavs]
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
    parser.add_argument('takes', nargs='+', help='arquivos .wav ou .m4a, ou pastas com takes')
    parser.add_argument('--conferir', action='store_true', help='so mostra o relatorio, sem gravar nada')
    parser.add_argument('--refazer', action='store_true',
                        help='apaga os cortes antigos do take e corta de novo (e converte o .m4a de novo)')
    parser.add_argument('--relatorio', help='salva um CSV com cada corte (inicio, fim, duracao, status)')
    args = parser.parse_args(argv)

    takes = listar_takes(args.takes)
    if not takes:
        print('nenhum .wav encontrado')
        return 1
    try:
        vistos = {}
        for take in takes:
            nome = fontes.ler_nome_do_take(take)['take']
            if nome in vistos:
                raise fontes.ErroDeNome(f'o take {nome} apareceu duas vezes ({vistos[nome]} e {take}); passe so um deles')
            vistos[nome] = take
        resultados = []
        with tempfile.TemporaryDirectory() as temporaria:
            for take in takes:
                info, cortes = cortar_take(take, conferir=args.conferir, refazer=args.refazer,
                                           pasta_temporaria=temporaria)
                resultados.append((take, info, cortes))
    except (fontes.ErroDeNome, FileExistsError, ErroDeConversao) as erro:
        print(f'parei: {erro}')
        return 1
    imprimir_relatorio(resultados, args.relatorio)
    if args.conferir:
        print('modo conferencia: nada foi gravado')
    return 0


if __name__ == '__main__':
    sys.exit(main())
