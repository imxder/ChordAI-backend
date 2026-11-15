"""Script de corte do dataset proprio (scripts/cortar_takes.py), com takes sinteticos."""
import hashlib
import os
import shutil
import subprocess

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
    assert fontes.ler_nome_do_take('C_violao_s1_take01.m4a')['take'] == 'C_violao_s1_take01'


def test_corta_uma_vez_por_batida_e_descarta_batida_dupla(ambiente):
    inicios = [0.5, 4.5, 8.5, 12.5, 14.0, 16.5]
    caminho = gravar(ambiente, 'Am_violao_s1_take01.wav', montar_take(inicios, 20.0))
    _, cortes = cortar(ambiente, caminho, conferir=True)

    ok = [c for c in cortes if c['status'] == 'ok']
    assert len(ok) == 5
    assert [c['status'] for c in cortes].count('duas batidas') == 1
    for c, inicio in zip(ok, [0.5, 4.5, 8.5, 14.0, 16.5]):
        assert inicio - 0.25 <= c['inicio'] <= inicio
        assert cortar_takes.CORTE_MINIMO <= c['duracao'] <= 3.0


def test_rasgueado_espalhado_vira_uma_batida_que_comeca_no_primeiro_onset(ambiente):
    # Cordas a 0,25 s uma da outra: cada uma perto da anterior, mas o grupo
    # todo passa de JUNTAR_SEGUNDOS. Antes o corte comecava na ultima corda.
    y = montar_take([0.5, 8.5], 12.0)
    for k, inicio in enumerate([4.5, 4.75, 5.0, 5.25]):
        corda = batida(f0=110.0 * (k + 1), ganho=0.3)
        a = int(inicio * SR)
        y[a:a + len(corda)] += corda[:len(y) - a]
    caminho = gravar(ambiente, 'C_violao_s1_take01.wav', y)

    _, cortes = cortar(ambiente, caminho, conferir=True)
    assert [c['status'] for c in cortes] == ['ok', 'ok', 'ok']
    assert 4.25 <= cortes[1]['inicio'] <= 4.5


def test_ruido_na_cauda_e_batida_logo_em_cima_nao_viram_batida_nova(ambiente):
    # Com o acorde ainda soando alto, o volume quase nao sobe: um estalo na
    # cauda ou uma segunda batida a 0,4 s ficam dentro do corte da primeira.
    y = montar_take([0.5, 4.5, 4.9, 8.5], 12.0)
    estalo = 0.05 * np.random.default_rng(2).standard_normal(int(0.02 * SR)).astype(np.float32)
    a = int(1.8 * SR)
    y[a:a + len(estalo)] += estalo
    caminho = gravar(ambiente, 'G_violao_s1_take01.wav', y)

    _, cortes = cortar(ambiente, caminho, conferir=True)
    assert [c['status'] for c in cortes] == ['ok', 'ok', 'ok']
    assert [round(c['inicio']) for c in cortes] == [0, 4, 8]


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


precisa_afconvert = pytest.mark.skipif(shutil.which('afconvert') is None, reason='afconvert so existe no macOS')


def gravar_m4a(ambiente, nome, y):
    """Um take como o Gravador do iPhone exporta: AAC mono em .m4a."""
    wav = gravar(ambiente, 'origem.wav', y)
    caminho = str(ambiente['pasta'] / nome)
    subprocess.run(['afconvert', '-f', 'm4af', '-d', 'aac', wav, caminho], check=True, capture_output=True)
    os.remove(wav)
    return caminho


@precisa_afconvert
def test_m4a_do_iphone_e_convertido_e_cortado(ambiente):
    caminho = gravar_m4a(ambiente, 'C_violao_s1_take01.m4a', montar_take([0.5, 4.5, 8.5], 12.0))
    pasta_wav = str(ambiente['pasta'] / 'gravacoes')
    antes = hashlib.sha1(open(caminho, 'rb').read()).hexdigest()

    _, cortes = cortar(ambiente, caminho, conferir=True, pasta_wav=pasta_wav,
                       pasta_temporaria=str(ambiente['pasta'] / 'tmp'))
    assert [c['status'] for c in cortes] == ['ok', 'ok', 'ok']
    assert not os.path.exists(pasta_wav) and not os.path.exists(ambiente['destino'])

    cortar(ambiente, caminho, pasta_wav=pasta_wav)
    wav = os.path.join(pasta_wav, 'C_violao_s1_take01.wav')
    info = sf.info(wav)
    assert info.samplerate == SR and info.subtype == 'PCM_24' and abs(info.duration - 12.0) < 0.1
    assert len(os.listdir(os.path.join(ambiente['destino'], 'C'))) == 3
    assert os.listdir(pasta_wav) == ['C_violao_s1_take01.wav']
    assert hashlib.sha1(open(caminho, 'rb').read()).hexdigest() == antes

    with pytest.raises(FileExistsError, match='--refazer'):
        cortar(ambiente, caminho, pasta_wav=pasta_wav)
    cortar(ambiente, caminho, pasta_wav=pasta_wav, refazer=True)
    assert len(os.listdir(os.path.join(ambiente['destino'], 'C'))) == 3


def test_na_pasta_o_wav_convertido_ganha_do_m4a_mas_os_dois_juntos_param(tmp_path, capsys):
    for ext in ('.wav', '.m4a'):
        (tmp_path / f'C_violao_s1_take01{ext}').write_bytes(b'')
    (tmp_path / 'G_violao_s1_take01.m4a').write_bytes(b'')
    assert [os.path.basename(t) for t in cortar_takes.listar_takes([str(tmp_path)])] == [
        'C_violao_s1_take01.wav', 'G_violao_s1_take01.m4a']

    assert cortar_takes.main([str(tmp_path / 'C_violao_s1_take01.wav'), str(tmp_path / 'C_violao_s1_take01.m4a'),
                              '--conferir']) == 1
    assert 'duas vezes' in capsys.readouterr().out


def test_onsets_de_cauda_em_sequencia_nao_engolem_a_batida_seguinte(ambiente):
    # Como no G_violao_s1_take02: o acorde ainda soa, tem ruido de fundo e a
    # cauda tem estalinhos a cada 0,25 s ate a batida seguinte. Eles nao podem
    # virar batida, nem puxar o comeco do corte de 4,5 s para antes do ataque.
    y = 0.01 * np.random.default_rng(0).standard_normal(int(12.0 * SR)).astype(np.float32)
    for inicio in [0.5, 4.5, 8.5]:
        b = batida(segundos=4.0)
        a = int(inicio * SR)
        y[a:a + len(b)] += b[:len(y) - a]
    for k in range(10):
        a = int((2.2 + 0.25 * k) * SR)
        y[a:a + int(0.01 * SR)] += 0.08 * np.random.default_rng(10 + k).standard_normal(int(0.01 * SR)).astype(np.float32)
    caminho = gravar(ambiente, 'G_violao_s1_take02.wav', y)

    _, cortes = cortar(ambiente, caminho, conferir=True)
    ok = [c for c in cortes if c['status'] == 'ok']
    assert [round(c['inicio']) for c in ok] == [0, 4, 8]
    assert 4.25 <= ok[1]['inicio'] <= 4.5


def take_que_sustenta(total=14.0):
    """Guitarra limpa: o acorde quase nao decai, entao a batida seguinte soma
    em cima de um som ainda alto e o volume sobe so umas 2 a 3 vezes."""
    y = np.zeros(int(total * SR), dtype=np.float32)
    for inicio in [0.5, 4.5, 8.5]:
        a = int(inicio * SR)
        b = batida(segundos=total - inicio)
        b = b * np.exp((1.5 - 0.15) * np.arange(len(b)) / SR)
        y[a:] += b[:len(y) - a]
    return 0.3 * y


def test_jazzmaster_usa_limite_de_subida_menor_que_o_violao(ambiente):
    y = take_que_sustenta()
    _, violao = cortar(ambiente, gravar(ambiente, 'C_violao_s1_take01.wav', y), conferir=True)
    _, guitarra = cortar(ambiente, gravar(ambiente, 'C_jazzmaster_s1_take01.wav', y), conferir=True)
    assert [c['status'] for c in violao] == ['ok']
    assert [c['status'] for c in guitarra] == ['ok', 'ok', 'ok']
