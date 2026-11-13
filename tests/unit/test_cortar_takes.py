"""Script de corte do dataset proprio (scripts/cortar_takes.py), com takes sinteticos."""
import hashlib
import os

import numpy as np
import pytest
import soundfile as sf

import pipeline
from scripts import cortar_takes, fontes

SR = 22050


def batida(segundos=2.8, f0=110.0, ganho=0.4):
    """Um 'acorde' sintetico: harmonicos que decaem, com um estalo curto de
    ruido no comeco, como o da palheta (sem ele o detector nao ve o ataque)."""
    t = np.arange(int(segundos * SR)) / SR
    som = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * np.exp(-1.5 * t)
    estalo = np.random.default_rng(int(f0)).standard_normal(len(t)) * np.exp(-t / 0.01)
    return (ganho * (som + 0.5 * estalo)).astype(np.float32)


def montar_take(inicios, total):
    y = np.zeros(int(total * SR), dtype=np.float32)
    for inicio in inicios:
        b = batida()
        a = int(inicio * SR)
        y[a:a + len(b)] += b[:len(y) - a]
    return y


@pytest.fixture
def ambiente(tmp_path):
    sessoes = {
        's1': {'sessao': 's1', 'microfone': 'interface', 'uso': 'treino'},
        's3': {'sessao': 's3', 'microfone': 'celular', 'uso': 'teste'},
        'sx': {'sessao': 'sx', 'microfone': '', 'uso': 'treino'},
    }
    return {
        'pasta': tmp_path,
        'destino': str(tmp_path / 'dataset_proprio'),
        'sources': str(tmp_path / 'sources.csv'),
        'sessoes': sessoes,
    }


def gravar(ambiente, nome, y):
    caminho = str(ambiente['pasta'] / nome)
    sf.write(caminho, y, SR, subtype='PCM_16')
    return caminho


def cortar(ambiente, caminho, **kw):
    return cortar_takes.cortar_take(caminho, destino=ambiente['destino'], sources=ambiente['sources'],
                                    sessoes=ambiente['sessoes'], **kw)


def test_constantes_batem_com_o_pipeline():
    assert fontes.NONE_LABEL == pipeline.NONE_LABEL
    assert fontes.SILENCE_RMS == pipeline.SILENCE_THRESHOLD
    assert fontes.ACORDES == ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']


@pytest.mark.parametrize('nome', ['Am_violao_s1.wav', 'am-violao-s1-take01.wav', 'Am_Violao_s1_take01.wav',
                                  'Xm_violao_s1_take01.wav', 'Am_violao_s1_take01.mp3'])
def test_nome_fora_do_padrao_para_e_avisa(nome):
    with pytest.raises(fontes.ErroDeNome):
        fontes.ler_nome_do_take(nome)


def test_nome_valido_inclui_extras_e_none():
    assert fontes.ler_nome_do_take('Am_violao_s1_take01.wav')['acorde'] == 'Am'
    assert fontes.ler_nome_do_take('D_violao_s1_take02.wav')['acorde'] == 'D'
    info = fontes.ler_nome_do_take('none_violao_s3_take01.wav')
    assert info['acorde'] == 'none' and fontes.rotulo(info['acorde']) == '_none'


def test_corta_uma_vez_por_batida_e_descarta_batida_dupla(ambiente):
    inicios = [0.5, 4.5, 8.5, 12.5, 12.9, 16.5]
    caminho = gravar(ambiente, 'Am_violao_s1_take01.wav', montar_take(inicios, 20.0))
    _, cortes = cortar(ambiente, caminho, conferir=True)

    ok = [c for c in cortes if c['status'] == 'ok']
    assert len(ok) == 5
    assert [c['status'] for c in cortes].count('duas batidas') == 1
    for c, inicio in zip(ok, [0.5, 4.5, 8.5, 12.9, 16.5]):
        assert inicio - 0.25 <= c['inicio'] <= inicio
        assert cortar_takes.CORTE_MINIMO <= c['duracao'] <= 3.0


def test_batida_fraca_fica_e_estalo_no_fim_do_som_sai(ambiente):
    y = montar_take([0.5, 4.5, 8.5, 12.5], 18.0)
    fraca = batida(ganho=0.1)
    a = int(4.5 * SR)
    y[a:a + len(fraca)] = fraca
    estalo = 0.15 * np.random.default_rng(1).standard_normal(int(0.02 * SR)).astype(np.float32)
    b = int(15.5 * SR)
    y[b:b + len(estalo)] += estalo
    caminho = gravar(ambiente, 'F_violao_s1_take01.wav', y)

    _, cortes = cortar(ambiente, caminho, conferir=True)
    assert [c['status'] for c in cortes] == ['ok', 'ok', 'ok', 'ok', 'fraco']


def test_conferir_nao_grava_nada(ambiente):
    caminho = gravar(ambiente, 'Am_violao_s1_take01.wav', montar_take([0.5, 4.5], 8.0))
    cortar(ambiente, caminho, conferir=True)
    assert not os.path.exists(ambiente['destino'])
    assert not os.path.exists(ambiente['sources'])


def test_salva_cortes_e_linhas_do_sources(ambiente):
    caminho = gravar(ambiente, 'Am_violao_s3_take02.wav', montar_take([0.5, 4.5, 8.5], 12.0))
    cortar(ambiente, caminho)

    arquivos = sorted(os.listdir(os.path.join(ambiente['destino'], 'Am')))
    assert arquivos == ['Am_violao_s3_take02_001.wav', 'Am_violao_s3_take02_002.wav', 'Am_violao_s3_take02_003.wav']
    linhas = fontes.ler_sources(ambiente['sources'])
    assert len(linhas) == 3
    assert {(l['acorde'], l['fonte'], l['instrumento'], l['sessao'], l['take'], l['microfone'], l['uso']) for l in linhas} == {
        ('Am', 'proprio', 'violao', 's3', 'Am_violao_s3_take02', 'celular', 'teste')
    }


def test_nunca_altera_o_take_e_nao_duplica(ambiente):
    caminho = gravar(ambiente, 'C_violao_s1_take01.wav', montar_take([0.5, 4.5, 8.5], 12.0))
    antes = hashlib.sha256(open(caminho, 'rb').read()).hexdigest()

    cortar(ambiente, caminho)
    with pytest.raises(FileExistsError):
        cortar(ambiente, caminho)
    cortar(ambiente, caminho, refazer=True)

    assert hashlib.sha256(open(caminho, 'rb').read()).hexdigest() == antes
    assert len(os.listdir(os.path.join(ambiente['destino'], 'C'))) == 3
    assert len(fontes.ler_sources(ambiente['sources'])) == 3


def test_refazer_com_menos_cortes_nao_deixa_sobra(ambiente):
    caminho = gravar(ambiente, 'G_violao_s1_take01.wav', montar_take([0.5, 4.5, 8.5], 12.0))
    cortar(ambiente, caminho)
    gravar(ambiente, 'G_violao_s1_take01.wav', montar_take([0.5, 4.5], 8.0))
    cortar(ambiente, caminho, refazer=True)
    assert len(os.listdir(os.path.join(ambiente['destino'], 'G'))) == 2


def test_sem_acorde_corta_em_pedacos_e_pula_silencio(ambiente):
    rng = np.random.default_rng(0)
    y = np.concatenate([0.05 * rng.standard_normal(int(5 * SR)), np.zeros(int(5 * SR)),
                        0.05 * rng.standard_normal(int(5 * SR))]).astype(np.float32)
    caminho = gravar(ambiente, 'none_violao_s3_take01.wav', y)
    cortar(ambiente, caminho)

    _, cortes = cortar(ambiente, caminho, conferir=True)
    assert [c['status'] for c in cortes] == ['ok', 'ok', 'silencio', 'silencio', 'ok', 'ok']
    assert len(os.listdir(os.path.join(ambiente['destino'], '_none'))) == 4
    assert {l['acorde'] for l in fontes.ler_sources(ambiente['sources'])} == {'_none'}


def test_sessao_sem_microfone_ou_fora_da_lista_para(ambiente):
    y = montar_take([0.5], 4.0)
    with pytest.raises(fontes.ErroDeNome, match='microfone'):
        cortar(ambiente, gravar(ambiente, 'Am_violao_sx_take01.wav', y))
    with pytest.raises(fontes.ErroDeNome, match='sessoes.csv'):
        cortar(ambiente, gravar(ambiente, 'Am_violao_s9_take01.wav', y))
