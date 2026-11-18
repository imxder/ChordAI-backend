"""Listagem do dataset, divisão treino/validação/teste por ARQUIVO e janelas de 1,5 s."""
import math
import os

import numpy as np
import soundfile as sf

import pipeline
from tests.sintetico import gravar_dataset

SR = pipeline.SAMPLE_RATE
JANELA = int(SR * pipeline.WINDOW_SECONDS)


def test_lista_so_wav_validos_em_ordem(tmp_path):
    gravar_dataset(str(tmp_path), acordes=('G', 'C'), por_acorde=3)
    (tmp_path / 'C' / 'leia-me.txt').write_text('não é áudio')
    sf.write(str(tmp_path / 'C' / 'curto.wav'), np.zeros(int(SR * 0.01), dtype=np.float32), SR)
    (tmp_path / 'G' / 'quebrado.wav').write_bytes(b'isso nao e um wav')

    arquivos, rotulos, classes = pipeline.list_dataset_files(str(tmp_path))

    assert classes == ['C', 'G']  # pastas em ordem alfabética
    nomes = [os.path.basename(p) for p in arquivos]
    assert nomes == ['C_00.wav', 'C_01.wav', 'C_02.wav', 'G_00.wav', 'G_01.wav', 'G_02.wav']
    assert rotulos == ['C'] * 3 + ['G'] * 3


def test_split_por_arquivo_sem_vazamento_e_estratificado(tmp_path):
    gravar_dataset(str(tmp_path), por_acorde=10)

    treino, teste, classes = pipeline.split_dataset_files(str(tmp_path))

    caminhos_treino = {p for p, _ in treino}
    caminhos_teste = {p for p, _ in teste}
    assert not caminhos_treino & caminhos_teste  # nenhum arquivo nos dois
    assert len(treino) + len(teste) == 40
    assert len(teste) == 8  # 20%
    for c in classes:  # estratificado: 2 de cada acorde no teste
        assert sum(1 for _, l in teste if l == c) == 2


def test_split_e_sempre_igual(tmp_path):
    gravar_dataset(str(tmp_path), por_acorde=10)
    primeira = pipeline.split_dataset_files(str(tmp_path))
    segunda = pipeline.split_dataset_files(str(tmp_path))
    assert primeira == segunda  # mesma divisão em toda rodada (random_state fixo)


def test_validacao_sai_do_treino_sem_repetir(tmp_path):
    gravar_dataset(str(tmp_path), por_acorde=10)
    treino, teste, _ = pipeline.split_dataset_files(str(tmp_path))

    sub, val = pipeline.split_train_val(treino)

    assert len(val) == math.ceil(0.15 * len(treino))  # 15%, o sklearn arredonda pra cima
    assert not {p for p, _ in sub} & {p for p, _ in val}
    assert sorted(sub + val) == sorted(treino)
    assert not {p for p, _ in val} & {p for p, _ in teste}


def test_janelas_tem_1_5_s_e_no_maximo_8(tmp_path):
    longo = tmp_path / 'longo.wav'
    sf.write(str(longo), np.ones(SR * 20, dtype=np.float32) * 0.1, SR)

    janelas = pipeline.make_windowed_segments(str(longo))

    assert len(janelas) == pipeline.MAX_CROPS_PER_FILE
    assert all(len(j) == JANELA for j in janelas)


def test_janelas_com_50_por_cento_de_sobreposicao(tmp_path):
    arq = tmp_path / 'tres.wav'
    rampa = np.linspace(0, 0.9, SR * 3, dtype=np.float32)  # cada amostra tem um valor diferente
    sf.write(str(arq), rampa, SR, subtype='FLOAT')
    # 3 s com janela de 1,5 s e passo de 0,75 s: janelas começando em 0; 0,75 e 1,5 s
    janelas = pipeline.make_windowed_segments(str(arq))
    assert len(janelas) == 3
    passo = int(SR * pipeline.CROP_HOP_SECONDS)
    for i, j in enumerate(janelas):
        np.testing.assert_allclose(j, rampa[i * passo:i * passo + JANELA], atol=1e-6)


def test_audio_curto_ganha_zeros_no_fim(tmp_path):
    curto = tmp_path / 'curto.wav'
    sf.write(str(curto), np.ones(SR // 2, dtype=np.float32) * 0.5, SR)  # 0,5 s

    janelas = pipeline.make_windowed_segments(str(curto))

    assert len(janelas) == 1
    assert len(janelas[0]) == JANELA
    assert np.all(janelas[0][SR // 2 + 10:] == 0)  # o que passou de 0,5 s é zero
    assert np.allclose(janelas[0][:SR // 2 - 10], 0.5, atol=1e-3)


def test_load_split_e_o_mesmo_split_do_treino(tmp_path):
    gravar_dataset(str(tmp_path), por_acorde=10)
    assert pipeline.load_split(str(tmp_path)) == pipeline.split_dataset_files(str(tmp_path))
