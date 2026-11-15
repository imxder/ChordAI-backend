"""Exportacao dos takes de um projeto do Ableton (scripts/exportar_ableton.py), com um .als minimo."""
import gzip
import os

import numpy as np
import pytest
import soundfile as sf

from scripts import exportar_ableton

SR = 48000


def clipe(nome, arquivo, inicio=0.0, fim=4.0, posicao=None, ganho=1.0, seg_por_batida_warp=1.0, pitch=0, loop=False):
    posicao = inicio if posicao is None else posicao
    return f'''
    <AudioClip Time="{posicao}">
      <Name Value="{nome}"/><Disabled Value="false"/>
      <CurrentStart Value="{posicao}"/><CurrentEnd Value="{posicao + fim - inicio}"/>
      <Loop><LoopStart Value="{inicio}"/><LoopEnd Value="{fim}"/><StartRelative Value="0"/>
            <LoopOn Value="{str(loop).lower()}"/></Loop>
      <IsWarped Value="true"/>
      <WarpMarkers><WarpMarker SecTime="0" BeatTime="0"/>
                   <WarpMarker SecTime="{0.03125 * seg_por_batida_warp}" BeatTime="0.03125"/></WarpMarkers>
      <SampleRef><FileRef><RelativePath Value="Samples/{arquivo}"/><Path Value=""/></FileRef></SampleRef>
      <SampleVolume Value="{ganho}"/><PitchCoarse Value="{pitch}"/><PitchFine Value="0"/>
    </AudioClip>'''


@pytest.fixture
def projeto(tmp_path):
    pasta = tmp_path / 'proj'
    (pasta / 'Samples').mkdir(parents=True)
    # Arquivo estereo de 6 s: cada amostra vale o proprio tempo em segundos / 10,
    # entao da para saber exatamente qual trecho foi recortado.
    t = np.arange(6 * SR) / SR
    sf.write(str(pasta / 'Samples' / 'gravado.wav'), np.stack([t / 10, t / 10], axis=1), SR, subtype='PCM_24')

    def montar(*clipes, bpm=60):
        xml = (f'<Ableton><LiveSet><MainTrack><DeviceChain><Mixer><Tempo><Manual Value="{bpm}"/></Tempo>'
               f'</Mixer></DeviceChain></MainTrack><Tracks><AudioTrack>{"".join(clipes)}</AudioTrack></Tracks>'
               '</LiveSet></Ableton>')
        caminho = pasta / 'projeto.als'
        with gzip.open(caminho, 'wt') as f:
            f.write(xml)
        return str(caminho)
    return montar, str(tmp_path / 'gravacoes')


def test_exporta_so_os_takes_com_nome_recortando_e_aplicando_o_ganho(projeto):
    montar, saida = projeto
    als = montar(
        clipe('C_jazzmaster_g1_take01', 'gravado.wav', inicio=1.0, fim=5.0, ganho=0.5),
        clipe('C_jazzmaster_g1_take01', 'gravado.wav', inicio=1.0, fim=5.0, ganho=0.5),  # faixa duplicada
        clipe('C 2', 'gravado.wav'),
    )
    resultados = exportar_ableton.exportar(als, saida)
    assert [r['take'] for r in resultados] == ['C_jazzmaster_g1_take01']
    assert os.listdir(saida) == ['C_jazzmaster_g1_take01.wav']

    y, sr = sf.read(os.path.join(saida, 'C_jazzmaster_g1_take01.wav'))
    info = sf.info(os.path.join(saida, 'C_jazzmaster_g1_take01.wav'))
    assert sr == SR and info.channels == 1 and info.subtype == 'PCM_24'
    assert len(y) == 4 * SR
    assert y[0] == pytest.approx(0.5 * 1.0 / 10, abs=1e-5)
    assert y[-1] == pytest.approx(0.5 * 5.0 / 10, abs=1e-4)


def test_pedaco_que_sobrou_dentro_do_take_e_emendado(projeto):
    montar, saida = projeto
    als = montar(
        clipe('Am_jazzmaster_g1_take01', 'gravado.wav', inicio=0.0, fim=5.0),
        clipe('Am_jazzmaster_g1_take01', 'gravado.wav', inicio=4.5, fim=5.5),
    )
    [r] = exportar_ableton.exportar(als, saida)
    assert r['pedacos'] == 2 and r['duracao'] == pytest.approx(5.5)


@pytest.mark.parametrize('problema, mensagem', [
    (dict(seg_por_batida_warp=2.0), 'esticando'),
    (dict(pitch=-12), 'pitch'),
    (dict(loop=True), 'loop'),
])
def test_para_quando_o_clipe_nao_toca_o_arquivo_como_foi_gravado(projeto, problema, mensagem):
    montar, saida = projeto
    als = montar(clipe('C_jazzmaster_g1_take01', 'gravado.wav', **problema))
    with pytest.raises(exportar_ableton.ErroDoProjeto, match=mensagem):
        exportar_ableton.exportar(als, saida)
    assert not os.path.exists(saida)


def test_mesmo_nome_em_trechos_diferentes_para(projeto):
    montar, saida = projeto
    als = montar(clipe('G_jazzmaster_g1_take01', 'gravado.wav', inicio=0.0, fim=2.0),
                 clipe('G_jazzmaster_g1_take01', 'gravado.wav', inicio=3.0, fim=5.0, posicao=10.0))
    with pytest.raises(exportar_ableton.ErroDoProjeto, match='trechos diferentes'):
        exportar_ableton.exportar(als, saida)


def test_nao_grava_por_cima_sem_refazer(projeto):
    montar, saida = projeto
    als = montar(clipe('C_jazzmaster_g1_take01', 'gravado.wav'))
    exportar_ableton.exportar(als, saida)
    with pytest.raises(FileExistsError, match='--refazer'):
        exportar_ableton.exportar(als, saida)
    exportar_ableton.exportar(als, saida, refazer=True)
    assert exportar_ableton.exportar(als, saida, conferir=True)[0]['duracao'] == pytest.approx(4.0)
