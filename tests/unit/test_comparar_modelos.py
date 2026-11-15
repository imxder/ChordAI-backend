"""Metricas e montagem dos testes da comparacao de modelos (scripts/comparar_modelos.py)."""
import os

import numpy as np
import pytest

import pipeline as pl
from scripts import comparar_modelos as cm
from scripts import fontes

FORTE = 0.1  # rms acima do filtro de silencio
FRACO = pl.SILENCE_THRESHOLD / 2


def test_aceito_segue_o_app_silencio_none_e_limiar():
    pred = np.array(['C', 'C', '_none', 'C'])
    conf = np.array([0.9, 0.9, 0.9, 0.4])
    rms = np.array([FORTE, FRACO, FORTE, FORTE])
    assert cm.aceito(pred, conf, rms, 0.5).tolist() == [True, False, False, False]


def test_acuracia_so_conta_os_acordes_pedidos():
    pred = np.array(['C', 'G', 'D', 'Dm'])
    y = np.array(['C', 'C', 'D', 'D'])
    assert cm.acuracia(pred, y, ['C']) == (0.5, 2)
    assert cm.acuracia(pred, y, ['D']) == (0.5, 2)
    assert cm.acuracia(pred, y, ['A']) == (None, 0)


def test_aceitos_cobertura_e_acerto_no_que_o_app_mostra():
    pred = np.array(['C', 'G', '_none', 'C'])
    conf = np.array([0.9, 0.9, 0.9, 0.3])
    rms = np.full(4, FORTE)
    y = np.array(['C', 'C', 'C', 'C'])
    r = cm.aceitos(pred, conf, rms, y, 0.5, ['C'])
    assert r['cobertura'] == 0.5 and r['acuracia_aceitos'] == 0.5


def test_falso_aceite_separa_gravado_e_sintetico():
    pred = np.array(['C', '_none', 'G', '_none', '_none', 'Am'])
    conf = np.full(6, 0.9)
    rms = np.full(6, FORTE)
    tipos = np.array(['gravado_abafadas', 'gravado_abafadas', 'gravado_afinando', 'gravado_afinando',
                      'noise', 'percussion'])
    r = cm.falso_aceite(pred, conf, rms, tipos, 0.5)
    assert r['gravado'] == 0.5 and r['sintetico'] == 0.5
    assert r['por_tipo']['gravado_abafadas'] == {'falso_aceite': 0.5, 'n': 2}
    assert r['por_tipo']['percussion']['falso_aceite'] == 1.0


def test_confusao_e_pares_que_diferem_por_uma_nota():
    pred = np.array(['A', 'Am', 'Am', 'Am', 'E'])
    y = np.array(['A', 'A', 'Am', 'Am', 'E'])
    c = cm.confusao(pred, y)
    assert c['A'] == {'A': 1, 'Am': 1}
    pares = cm.confusao_pares(c)
    assert pares['A->Am'] == {'n': 1, 'de': 2}
    assert pares['Am->A'] == {'n': 0, 'de': 2}
    assert pares['E->Em'] == {'n': 0, 'de': 1}
    assert 'D->Dm' not in pares  # sem janelas de D nem de Dm


def test_peso_por_classe_compensa_a_classe_pequena():
    y = np.array(['C'] * 90 + ['D'] * 10)
    pesos = cm.pesos_por_classe(y)
    assert pesos['D'] == pytest.approx(5.0) and pesos['C'] == pytest.approx(100 / 180)
    # com os pesos, as duas classes pesam o mesmo na loss
    assert 90 * pesos['C'] == pytest.approx(10 * pesos['D'])


def test_validacao_sorteada_mantem_o_teste_fora_e_nao_perde_arquivos():
    linhas = fontes.ler_sources()
    if not linhas:
        pytest.skip('sem data/sources.csv')
    p = cm.dividir_treino.dividir(linhas, {'proprio', 'kaggle_v3'}, fontes.ACORDES)
    q = cm.validacao_sorteada(p)
    assert sorted(q['treino'] + q['val']) == sorted(p['treino'] + p['val'])
    assert not set(q['treino']) & set(q['val'])
    assert q['teste_v3'] == p['teste_v3'] and q['teste_proprio'] == p['teste_proprio']
    assert abs(len(q['val']) / (len(q['treino']) + len(q['val'])) - 0.15) < 0.01


def test_audio_de_afinacao_e_determinista_e_valido():
    a, b = cm.audio_afinando(3), cm.audio_afinando(3)
    assert np.array_equal(a, b) and not np.array_equal(a, cm.audio_afinando(4))
    assert len(a) == int(pl.SAMPLE_RATE * pl.WINDOW_SECONDS) and np.isfinite(a).all()
    assert np.abs(a).max() <= 1.0 and 0.009 <= pl.rms(a) <= 0.101  # passa do filtro de silencio


def test_negativos_de_afinacao_seguem_o_split_dos_sinteticos():
    X, rms = cm.negativos_afinacao('val')
    assert len(X) == len(pl.synth_negative_ids('val', n=cm.N_AFINACAO)) == len(rms)
    assert X.shape[1:] == (12, pl.MAX_PAD_LEN) and (rms >= pl.SILENCE_THRESHOLD).all()
    # nenhum id vai pro treino e pro teste ao mesmo tempo
    assert not set(pl.synth_negative_ids('train', n=cm.N_AFINACAO)) & set(pl.synth_negative_ids('test', n=cm.N_AFINACAO))


def test_sufixo_so_aparece_fora_do_padrao():
    assert cm.sufixo('chroma', 0) == ''
    assert cm.sufixo('chroma_stft8192', 0) == '_stft8192'
    assert cm.sufixo('chroma_stft8192', 2) == '_stft8192_aug2'
    assert cm.sufixo('chroma', 2) == '_aug2'


def test_metricas_de_saidas_funciona_com_probabilidades_prontas():
    classes = ['C', 'D', '_none']
    um = lambda i: np.eye(3)[i] * 0.9 + 0.1 / 3
    forte = np.full(2, FORTE)
    saidas = {
        'teste_v3': (np.array([um(0), um(1)]), np.array(['C', 'C']), forte),
        'take02': (np.array([um(0), um(1)]), np.array(['C', 'D']), forte),
        'sem_acorde': (np.array([um(2), um(0)]), np.array(['noise', 'gravado_afinando']), forte),
    }
    m = cm.metricas_de_saidas(saidas, classes, 0.5)
    assert m['teste_v3']['acuracia'] == 0.5 and m['teste_v3']['n_janelas'] == 2
    assert m['take02']['acuracia'] == 1.0 and m['take02']['acuracia_extras'] == 1.0
    assert m['sem_acorde']['gravado'] == 1.0 and m['sem_acorde']['sintetico'] == 0.0


def _seno(cents, segundos=2.0):
    t = np.arange(int(pl.SAMPLE_RATE * segundos)) / pl.SAMPLE_RATE
    f = 440.0 * 2 ** (cents / 1200)
    return (0.3 * sum(np.sin(2 * np.pi * h * f * t) / h for h in range(1, 5))).astype(np.float32)


def test_copia_afinada_leva_a_nota_para_o_la_440_sem_mexer_no_original(tmp_path):
    import soundfile as sf
    original = tmp_path / 'A_violao_s1_take01_001.wav'
    sf.write(original, _seno(-12), pl.SAMPLE_RATE)
    antes = original.read_bytes()
    assert cm.afinacao_do_take([str(original)]) == pytest.approx(-12, abs=3)
    copia = cm.copia_afinada(str(original), 12.0, destino=str(tmp_path / 'afinados'))
    assert original.read_bytes() == antes and copia != str(original)
    assert cm.afinacao_do_take([copia]) == pytest.approx(0, abs=3)


def test_corrigir_afinacao_so_mexe_nas_sessoes_novas(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'afinacao_do_take', lambda cs: -10.0)
    monkeypatch.setattr(cm, 'copia_afinada', lambda c, cents: c.replace('dataset_proprio', f'afinado{cents:+.0f}'))
    raiz = fontes.ROOT_DIR
    linhas = [
        {'caminho': 'dataset_proprio/C/C_violao_s1_take01_001.wav', 'acorde': 'C', 'fonte': 'proprio',
         'sessao': 's1', 'take': 'C_violao_s1_take01'},
        {'caminho': 'dataset_proprio/C/C_Alex_Acoustic_Mic.wav', 'acorde': 'C', 'fonte': 'proprio',
         'sessao': 'antiga', 'take': 'C_Alex_Acoustic_Mic'},
        {'caminho': 'dataset_kaggle/Training/C/C_x_1.wav', 'acorde': 'C', 'fonte': 'kaggle_v3',
         'sessao': 'x', 'take': 'C_x_1'},
    ]
    pares = [(os.path.join(raiz, l['caminho']), 'C') for l in linhas]
    partes, correcao = cm.corrigir_afinacao({'treino': pares, 'val': []}, linhas)
    assert correcao == {'C_violao_s1_take01': 10.0}
    caminhos = [c for c, _ in partes['treino']]
    assert any('afinado+10' in c for c in caminhos)
    assert any(c.endswith('C_Alex_Acoustic_Mic.wav') and 'dataset_proprio' in c for c in caminhos)
    assert any('dataset_kaggle' in c for c in caminhos)


def test_tipo_do_sem_acorde_gravado_vem_do_take():
    assert cm._tipo_gravado('/r/dataset_proprio/_none/none_violao_s1_take04_003.wav') == 'gravado_abafadas'
    assert cm._tipo_gravado('/r/dataset_proprio/_none/none_violao_s1_take06_012.wav') == 'gravado_afinando'


def test_testes_sao_os_mesmos_para_todos_os_modelos():
    """Os testes saem do sources.csv de verdade com as duas fontes e todos os
    acordes; nenhum arquivo de teste aparece no treino ou na validacao de
    nenhum modelo."""
    linhas = fontes.ler_sources()
    if not linhas:
        pytest.skip('sem data/sources.csv')
    testes = cm.partes_de_teste()
    arquivos_teste = {p for p, _ in testes['teste_v3'] + testes['teste_proprio']} | set(testes['none_teste'])
    assert {a for _, a in testes['teste_v3']} == set(fontes.ACORDES)
    for nome, (fontes_usadas, acordes, _) in cm.MODELOS.items():
        p = cm.dividir_treino.dividir(linhas, fontes_usadas, acordes)
        vistos = {c for c, _ in p['treino'] + p['val']} | set(p['none_treino']) | set(p['none_val'])
        assert not (vistos & arquivos_teste), nome


def test_resumo_mostra_media_e_extremos_das_seeds():
    def m(seed, v3):
        return {'modelo': 'B8', 'seed': seed, 'limiar': 0.5, 'classes': fontes.ACORDES + ['_none'],
                'teste_v3': {'acuracia': v3, 'n_janelas': 10},
                'take02': {'acuracia': 0.9, 'acuracia_extras': 0.0, 'n_janelas': 4, 'n_janelas_extras': 2,
                           'pares': {}},
                'sem_acorde': {'gravado': 0.1, 'sintetico': 0.0, 'por_tipo': {'noise': {'falso_aceite': 0.0, 'n': 3}}}}
    texto = cm.resumo({'B8': [m(42, 0.90), m(43, 0.94)]})
    assert '| B8 | 42, 43 | 92.0% (90.0–94.0) | 90.0% (90.0–90.0) | - |' in texto
