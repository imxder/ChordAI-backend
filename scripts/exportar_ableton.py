"""Exporta os takes de um projeto do Ableton, um .wav por take, sem abrir o Ableton.

Uso:
    python -m scripts.exportar_ableton "/Users/alex/Documents/dataset Project/dataset.als" --conferir
    python -m scripts.exportar_ableton "/Users/alex/Documents/dataset Project/dataset.als"
    python -m scripts.cortar_takes gravacoes/

Como funciona:
- O .als e um XML compactado. Cada clipe de audio aponta para um arquivo em
  Samples/ e diz qual trecho dele toca. O script le isso e recorta o trecho
  direto do arquivo original: nada de render, nada de warp no meio.
- So entram os clipes cujo nome segue o padrao dos takes
  (<acorde>_<instrumento>_<sessao>_take<NN>, como C_jazzmaster_g1_take01).
  Os outros (tentativas, "C 2", "Am 3") ficam de fora.
- Um take pode estar dividido em mais de um clipe no arranjo, desde que os
  pedacos sejam seguidos no tempo e no arquivo: eles sao emendados.
- O ganho do clipe e aplicado. Um arquivo que o Ableton consolidou vem
  normalizado, e o ganho do clipe e o que traz ele de volta ao volume gravado.
- Para e avisa se o warp estiver esticando o audio (marcadores fora de 1:1
  com o tempo do projeto), se o pitch do clipe foi mexido, se o clipe esta em
  loop ou se o mesmo nome aparece em dois trechos diferentes do arquivo.
- Sai em mono, mesma taxa de amostragem, PCM 24 bits, em gravacoes/<take>.wav.
  Se o .wav ja existir, para e avisa (com --refazer, grava por cima)."""
import argparse
import gzip
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf

from scripts import fontes

TOLERANCIA = 1e-3  # segundos


class ErroDoProjeto(ValueError):
    pass


def _valor(elemento, caminho, padrao=None):
    achado = elemento.find(caminho)
    return achado.get('Value') if achado is not None else padrao


def ler_projeto(caminho_als):
    with gzip.open(caminho_als) as f:
        return ET.parse(f).getroot()


def tempo_do_projeto(raiz):
    for tempo in raiz.iter('Tempo'):
        manual = tempo.find('Manual')
        if manual is not None:
            return float(manual.get('Value'))
    raise ErroDoProjeto('nao achei o tempo (BPM) do projeto no .als')


def arquivo_do_clipe(clipe, pasta_projeto):
    ref = clipe.find('SampleRef/FileRef')
    relativo = _valor(ref, 'RelativePath')
    if relativo:
        candidato = os.path.join(pasta_projeto, relativo)
        if os.path.exists(candidato):
            return candidato
    absoluto = _valor(ref, 'Path')
    if absoluto and os.path.exists(absoluto):
        return absoluto
    raise ErroDoProjeto(f'clipe "{_valor(clipe, "Name")}": nao achei o arquivo {relativo or absoluto}')


def segundos_por_batida_do_warp(clipe):
    """Segundos de arquivo por batida, pelos marcadores de warp. Com o warp
    desligado o Ableton toca o arquivo na velocidade original."""
    marcadores = [(float(m.get('SecTime')), float(m.get('BeatTime'))) for m in clipe.findall('WarpMarkers/WarpMarker')]
    if _valor(clipe, 'IsWarped') != 'true' or len(marcadores) < 2:
        return None
    razoes = {round((s2 - s1) / (b2 - b1), 6) for (s1, b1), (s2, b2) in zip(marcadores, marcadores[1:]) if b2 != b1}
    if len(razoes) != 1:
        raise ErroDoProjeto(f'clipe "{_valor(clipe, "Name")}": o warp tem mais de um marcador mexido; '
                            'desligue o warp ou exporte pelo Ableton')
    return razoes.pop()


def trechos_dos_takes(raiz, pasta_projeto):
    """{take: [trechos]}, cada trecho com arquivo, inicio e fim em segundos do
    arquivo, posicao no arranjo (em segundos) e ganho."""
    seg_por_batida = 60.0 / tempo_do_projeto(raiz)
    takes = {}
    vistos = set()
    for clipe in raiz.iter('AudioClip'):
        nome = _valor(clipe, 'Name', '')
        try:
            fontes.ler_nome_do_take(nome + '.wav')
        except fontes.ErroDeNome:
            continue
        if _valor(clipe, 'Disabled') == 'true':
            continue
        if _valor(clipe, 'Loop/LoopOn') == 'true':
            raise ErroDoProjeto(f'clipe "{nome}" esta em loop; desligue o loop do clipe')
        if float(_valor(clipe, 'PitchCoarse', 0)) != 0 or float(_valor(clipe, 'PitchFine', 0)) != 0:
            raise ErroDoProjeto(f'clipe "{nome}" tem o pitch mexido; volte o Transpose/Detune para 0')

        warp = segundos_por_batida_do_warp(clipe)
        if warp is not None and abs(warp - seg_por_batida) > 1e-4:
            raise ErroDoProjeto(
                f'clipe "{nome}": o warp esta esticando o audio ({warp:.4f} s por batida no arquivo, '
                f'{seg_por_batida:.4f} no projeto). Desligue o warp do clipe ou volte os marcadores.'
            )
        inicio_batidas = float(_valor(clipe, 'Loop/LoopStart')) + float(_valor(clipe, 'Loop/StartRelative', 0))
        duracao_batidas = float(_valor(clipe, 'CurrentEnd')) - float(_valor(clipe, 'CurrentStart'))
        # Com 1:1 (ou warp desligado), uma batida do clipe e seg_por_batida segundos do arquivo.
        arquivo = arquivo_do_clipe(clipe, pasta_projeto)
        trecho = {
            'arquivo': arquivo,
            'inicio': inicio_batidas * seg_por_batida,
            'fim': (inicio_batidas + duracao_batidas) * seg_por_batida,
            'arranjo': float(_valor(clipe, 'CurrentStart')) * seg_por_batida,
            'ganho': float(_valor(clipe, 'SampleVolume', 1)),
        }
        # O mesmo clipe aparece de novo quando a faixa foi duplicada.
        chave = (nome, arquivo, round(trecho['inicio'], 4), round(trecho['fim'], 4), round(trecho['arranjo'], 4))
        if chave in vistos:
            continue
        vistos.add(chave)
        takes.setdefault(nome, []).append(trecho)

    # Pedacos do mesmo take (clipe dividido, ou uma sobra numa faixa duplicada)
    # viram um trecho so, desde que apontem para o mesmo lugar do mesmo arquivo
    # e se encostem ou se sobreponham.
    juntos = {}
    for nome, trechos in takes.items():
        primeiro = trechos[0]
        deslocamento = primeiro['arranjo'] - primeiro['inicio']
        for t in trechos:
            if (t['arquivo'] != primeiro['arquivo'] or abs(t['ganho'] - primeiro['ganho']) > 1e-6
                    or abs((t['arranjo'] - t['inicio']) - deslocamento) > TOLERANCIA):
                raise ErroDoProjeto(f'o take "{nome}" aparece em clipes que tocam trechos diferentes; '
                                    'deixe so um clipe com esse nome')
        trechos.sort(key=lambda t: t['inicio'])
        inicio, fim = trechos[0]['inicio'], trechos[0]['fim']
        for t in trechos[1:]:
            if t['inicio'] > fim + TOLERANCIA:
                raise ErroDoProjeto(f'o take "{nome}" tem um buraco entre os clipes; deixe so um clipe com esse nome')
            fim = max(fim, t['fim'])
        juntos[nome] = [dict(primeiro, inicio=inicio, fim=fim, arranjo=inicio + deslocamento, pedacos=len(trechos))]
    return juntos


def montar_audio(trechos):
    pedacos, taxa = [], None
    for t in trechos:
        info = sf.info(t['arquivo'])
        if taxa is not None and info.samplerate != taxa:
            raise ErroDoProjeto('pedacos do mesmo take com taxas de amostragem diferentes')
        taxa = info.samplerate
        a = int(round(t['inicio'] * taxa))
        b = min(int(round(t['fim'] * taxa)), info.frames)
        y, _ = sf.read(t['arquivo'], start=a, stop=b, always_2d=True)
        pedacos.append(y.mean(axis=1) * t['ganho'])
    return np.concatenate(pedacos), taxa


def exportar(caminho_als, pasta_saida=None, conferir=False, refazer=False):
    raiz = ler_projeto(caminho_als)
    takes = trechos_dos_takes(raiz, os.path.dirname(os.path.abspath(caminho_als)))
    pasta_saida = pasta_saida or fontes.GRAVACOES
    resultados = []
    if not refazer and not conferir:
        existentes = [n for n in takes if os.path.exists(os.path.join(pasta_saida, n + '.wav'))]
        if existentes:
            raise FileExistsError(f'{len(existentes)} takes ja existem em {fontes.caminho_relativo(pasta_saida)} '
                                  f'({", ".join(sorted(existentes)[:3])}...). Use --refazer para gravar por cima.')
    for nome in sorted(takes):
        y, taxa = montar_audio(takes[nome])
        pico = float(np.abs(y).max()) if len(y) else 0.0
        if pico > 1.0:
            raise ErroDoProjeto(f'o take "{nome}" passa de 0 dB com o ganho do clipe (pico {pico:.2f})')
        saida = os.path.join(pasta_saida, nome + '.wav')
        if not conferir:
            os.makedirs(pasta_saida, exist_ok=True)
            sf.write(saida, y.astype(np.float32), taxa, subtype='PCM_24')
        resultados.append({'take': nome, 'duracao': len(y) / taxa, 'pico': pico, 'pedacos': takes[nome][0]['pedacos'],
                           'arquivo': saida})
    return resultados


def main(argv=None):
    parser = argparse.ArgumentParser(description='Exporta os takes de um projeto do Ableton, um .wav por take.')
    parser.add_argument('projeto', help='arquivo .als do projeto')
    parser.add_argument('--saida', help='pasta de saida (padrao: gravacoes/)')
    parser.add_argument('--conferir', action='store_true', help='so lista os takes, sem gravar nada')
    parser.add_argument('--refazer', action='store_true', help='grava por cima dos .wav que ja existem')
    args = parser.parse_args(argv)
    try:
        resultados = exportar(args.projeto, args.saida, conferir=args.conferir, refazer=args.refazer)
    except (ErroDoProjeto, FileExistsError, fontes.ErroDeNome) as erro:
        print(f'parei: {erro}')
        return 1
    for r in resultados:
        extra = f', {r["pedacos"]} pedacos emendados' if r['pedacos'] > 1 else ''
        print(f'{r["take"]}: {r["duracao"]:.2f} s, pico {20 * np.log10(max(r["pico"], 1e-9)):.1f} dB{extra}')
    print(f'{len(resultados)} takes' + (' (modo conferencia: nada foi gravado)' if args.conferir else ' exportados'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
