"""O fluxo do scripts/comparar_modelos.py sem o dataset: cache das features, montagem
do treino e dos testes, treino de um modelo pequeno, ensemble, avaliação do modelo da
API e do .tflite do celular, e o main() que junta tudo no comparativo.md."""
import json
import os
import shutil

import numpy as np
import pytest

import pipeline as pl
from scripts import comparar_modelos as cm
from scripts import fontes
from tests.sintetico import gravar_dataset

FRAMES = pl.MAX_PAD_LEN
ACORDES = fontes.ACORDES  # os 8 do app


def _janelas(rotulos, seed=0):
    """Features falsas (12 x 65) em que cada acorde tem um bin mais forte:
    fácil de separar, pro treino de teste acabar em segundos."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(rotulos), 12, FRAMES)) * 0.3
    todos = ACORDES + [pl.NONE_LABEL] + fontes.ACORDES_EXTRAS
    for i, r in enumerate(rotulos):
        X[i, todos.index(r) % 12] += 3.0
    return X


def _testes(seed=1):
    y_v3 = np.array(ACORDES * 3)
    y_take = np.array(ACORDES * 2)
    tipos = np.array(['gravado_abafadas', 'gravado_afinando', 'noise', 'percussion'] * 3)
    return {
        'teste_v3': {'X': _janelas(y_v3, seed), 'y': y_v3, 'rms': np.full(len(y_v3), 0.1)},
        'take02': {'X': _janelas(y_take, seed + 1), 'y': y_take, 'rms': np.full(len(y_take), 0.1)},
        'sem_acorde': {'X': _janelas([pl.NONE_LABEL] * len(tipos), seed + 2), 'tipo': tipos,
                       'rms': np.full(len(tipos), 0.1)},
    }


@pytest.fixture
def resultados(tmp_path, monkeypatch):
    pasta = tmp_path / 'results_duas_fontes'
    monkeypatch.setattr(cm, 'RESULTADOS', str(pasta))
    monkeypatch.setattr(cm, 'FEATURE', 'chroma')
    return pasta


# --- cache das features ----------------------------------------------------

def test_features_vao_pro_cache_e_nao_sao_extraidas_de_novo(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(cm, 'FEATURE', 'chroma')
    chamadas = []

    def extrair(pares, feature):
        chamadas.append(len(pares))
        n = len(pares)
        return _janelas(['C'] * n), np.array(['C'] * n), None, np.arange(n), np.full(n, 0.1)

    monkeypatch.setattr(pl, 'build_features', extrair)
    pares = [(os.path.join(fontes.ROOT_DIR, 'dataset_kaggle', 'C', f'c{i}.wav'), 'C') for i in range(3)]

    X1, y1, rms1, fidx1 = cm._features(pares)
    X2, y2, _, _ = cm._features(pares)  # a segunda vem do .npz

    assert chamadas == [3]
    np.testing.assert_array_equal(X1, X2)
    assert list(y2) == ['C'] * 3 and list(fidx1) == [0, 1, 2]
    cm._features(pares[:2])  # outra lista de arquivos: extrai de novo
    assert chamadas == [3, 2]


# --- montagem do treino e dos testes ---------------------------------------

def _fakes_de_extracao(monkeypatch):
    """Troca a extração por funções que devolvem uma janela por arquivo."""
    def features(pares):
        rot = [a for _, a in pares]
        return _janelas(rot), np.array(rot), np.full(len(rot), 0.1), np.arange(len(rot))

    vistos = {}

    def sinteticos(split, pares):
        vistos[split] = list(pares)
        return _janelas([pl.NONE_LABEL] * 2), np.array(['noise', 'percussion']), np.full(2, 0.05)

    monkeypatch.setattr(cm, '_features', features)
    monkeypatch.setattr(cm, '_negativos_sinteticos', sinteticos)
    monkeypatch.setattr(cm, 'negativos_afinacao', lambda split: (_janelas([pl.NONE_LABEL]), np.array([0.05])))
    return vistos


def _arquivo(pasta, nome):
    return os.path.join(fontes.ROOT_DIR, pasta, nome)


def test_montar_treino_junta_acordes_sinteticos_e_gravados(monkeypatch):
    _fakes_de_extracao(monkeypatch)
    partes = {
        'treino': [(_arquivo('dataset_proprio/C', 'C_violao_s1_take01_001.wav'), 'C'),
                   (_arquivo('dataset_kaggle/Training/G', 'G_x_1.wav'), 'G')],
        'val': [(_arquivo('dataset_kaggle/Training/C', 'C_x_2.wav'), 'C')],
        'none_treino': [_arquivo('dataset_proprio/_none', 'none_violao_s1_take04_001.wav')],
        'none_val': [],
    }

    d = cm.montar_treino(partes, afinacao=True)

    # treino: 2 acordes + 2 sintéticos + 1 de afinação + 1 gravado
    assert list(d['y_treino']) == ['C', 'G'] + [pl.NONE_LABEL] * 4
    assert (d['n_acordes_treino'], d['n_sinteticos_treino'], d['n_gravados_treino']) == (2, 3, 1)
    assert d['X_treino'].shape == (6, 12, FRAMES) and len(d['rms_treino']) == 6
    # só o corte da s1 é sessão nova
    assert d['novos_treino'].tolist() == [True, False, False, False, False, False]
    # validação sem sem-acorde gravado
    assert list(d['y_val']) == ['C'] + [pl.NONE_LABEL] * 3
    assert d['n_gravados_val'] == 0


def test_montar_testes_tipos_e_origem_do_volume_baixo(monkeypatch):
    vistos = _fakes_de_extracao(monkeypatch)
    v3 = [(_arquivo('dataset_kaggle/Test/C', 'C_1.wav'), 'C'), (_arquivo('dataset_kaggle/Test/G', 'G_1.wav'), 'G')]
    proprio = [(_arquivo('dataset_proprio/C', 'C_violao_s1_take02_001.wav'), 'C'),
               (_arquivo('dataset_proprio/A', 'A_violao_s1_take02_001.wav'), 'A')]  # A é extra
    partes = {'teste_v3': v3, 'teste_proprio': proprio,
              'none_teste': [_arquivo('dataset_proprio/_none', 'none_violao_s1_take04_001.wav'),
                             _arquivo('dataset_proprio/_none', 'none_violao_s1_take06_001.wav')]}

    t = cm.montar_testes(partes)

    assert list(t['teste_v3']['y']) == ['C', 'G'] and list(t['take02']['y']) == ['C', 'A']
    assert list(t['sem_acorde']['tipo']) == ['gravado_abafadas', 'gravado_afinando', 'noise', 'percussion']
    # o volume baixo sai dos acordes de teste das duas fontes, só dos 8 do app (o A fica de fora)
    assert vistos['test'] == v3 + proprio[:1]

    cm.montar_testes(partes, low_volume_so_v3=True)
    assert vistos['test'] == v3


# --- treino, ensemble e avaliação -------------------------------------------

@pytest.fixture(scope='module')
def _treinos(tmp_path_factory):
    """Treina o modelo A (8 acordes + _none) com as seeds 42 e 43 em dados falsos,
    uma vez só pro arquivo inteiro (cada treino leva uns 20 s)."""
    pasta = tmp_path_factory.mktemp('treinos') / 'results_duas_fontes'
    mp = pytest.MonkeyPatch()
    mp.setattr(cm, 'RESULTADOS', str(pasta))
    mp.setattr(cm, 'FEATURE', 'chroma')
    rot_treino = (ACORDES + [pl.NONE_LABEL]) * 12
    n_val = len(ACORDES) * 4
    d = {'X_treino': _janelas(rot_treino, 3), 'y_treino': np.array(rot_treino),
         'X_val': np.concatenate([_janelas(ACORDES * 4, 4), _janelas([pl.NONE_LABEL] * 4, 5)]),
         'y_val': np.array(ACORDES * 4 + [pl.NONE_LABEL] * 4),
         'rms_val': np.full(n_val + 4, 0.1),
         'n_acordes_treino': len(ACORDES) * 12, 'n_sinteticos_treino': 12, 'n_gravados_treino': 0,
         'n_acordes_val': n_val, 'n_sinteticos_val': 4, 'n_gravados_val': 0}
    partes = {'treino': [('x', 'C')] * 96, 'val': [('y', 'C')] * 32, 'none_treino': [], 'none_val': []}
    testes = _testes()
    cache = {'A': (partes, d)}
    try:
        for seed in (42, 43):
            cm.treinar('A', seed, testes, cache)
    finally:
        mp.undo()
    return pasta, testes


@pytest.fixture
def dois_treinos(_treinos, monkeypatch):
    pasta, testes = _treinos
    monkeypatch.setattr(cm, 'RESULTADOS', str(pasta))
    monkeypatch.setattr(cm, 'FEATURE', 'chroma')
    return pasta, testes


def test_treinar_salva_modelo_e_metricas(dois_treinos):
    resultados, _ = dois_treinos
    pasta = resultados / 'A_s42'
    for nome in ('model.h5', 'encoder.joblib', 'scaler.joblib', 'model_config.json', 'metrics.json'):
        assert (pasta / nome).exists(), nome
    m = json.loads((pasta / 'metrics.json').read_text(encoding='utf-8'))
    assert m['modelo'] == 'A' and m['seed'] == 42
    assert m['classes'] == sorted(ACORDES + [pl.NONE_LABEL])
    assert m['limiar'] in pl.THRESHOLD_GRID
    # dados fáceis: o modelo tem que acertar quase tudo do teste
    assert m['teste_v3']['acuracia'] > 0.9
    assert m['janelas']['n_acordes_val'] == len(ACORDES) * 4
    config = json.loads((pasta / 'model_config.json').read_text(encoding='utf-8'))
    assert config['trained_by'] == 'scripts/comparar_modelos.py'


def test_ensemble_tira_a_media_das_seeds(dois_treinos):
    resultados, testes = dois_treinos
    cm.avaliar_ensemble('A', testes)
    m = json.loads((resultados / 'A_ens' / 'metrics.json').read_text(encoding='utf-8'))
    assert m['seeds'] == [42, 43] and m['modelo'] == 'A_ens'
    assert 'A_s42' in m['origem'] and 'A_s43' in m['origem']


def test_ensemble_sem_seed_treinada_para_com_erro(resultados):
    resultados.mkdir()
    with pytest.raises(SystemExit, match='nenhuma seed'):
        cm.avaliar_ensemble('B8', _testes())


def test_curva_do_limiar_tem_uma_linha_por_limiar(dois_treinos):
    resultados, testes = dois_treinos
    model, encoder, scaler, _ = pl.load_model_bundle(str(resultados / 'A_s42'))
    linhas = cm.curva_limiar(model, scaler, encoder.classes_, testes)
    assert [l['limiar'] for l in linhas] == [float(t) for t in pl.THRESHOLD_GRID]
    cobertura = [l['cobertura_v3'] for l in linhas]
    assert cobertura == sorted(cobertura, reverse=True)


def test_avalia_o_modelo_da_api_sem_treinar(resultados):
    cm.avaliar_modelo_atual(_testes())
    m = json.loads((resultados / '0' / 'metrics.json').read_text(encoding='utf-8'))
    assert m['modelo'] == '0' and m['origem'] == 'trained_model/ (so avaliado)'
    assert 0 <= m['teste_v3']['acuracia'] <= 1


def test_modelo_da_api_com_outra_feature_para(resultados, monkeypatch):
    monkeypatch.setattr(cm, 'FEATURE', 'chroma_stft8192')
    with pytest.raises(SystemExit, match='o script espera'):
        cm.avaliar_modelo_atual(_testes())


def test_avalia_o_tflite_do_celular_com_audio(resultados, tmp_path):
    dataset = gravar_dataset(str(tmp_path / 'ds'), acordes=('C', 'G'), por_acorde=2, segundos=1.5)
    none = tmp_path / 'ds' / 'none_violao_s1_take04_001.wav'
    shutil.copy(os.path.join(dataset, 'C', 'C_00.wav'), none)
    partes = {'teste_v3': [(os.path.join(dataset, 'C', 'C_00.wav'), 'C'), (os.path.join(dataset, 'G', 'G_00.wav'), 'G')],
              'teste_proprio': [(os.path.join(dataset, 'C', 'C_01.wav'), 'C')],
              'none_teste': [str(none)]}

    cm.avaliar_tflite(partes)

    m = json.loads((resultados / '0c' / 'metrics.json').read_text(encoding='utf-8'))
    assert m['modelo'] == '0c' and 'chord_model.tflite' in m['origem']
    assert m['teste_v3']['n_janelas'] == 2
    assert 'gravado_abafadas' in m['sem_acorde']['por_tipo']


# --- resumo e main ----------------------------------------------------------

def _metricas_falsas(modelo, seed, v3=0.9):
    return {'modelo': modelo, 'seed': seed, 'limiar': 0.5, 'classes': ACORDES + [pl.NONE_LABEL],
            'teste_v3': {'acuracia': v3, 'n_janelas': 10},
            'take02': {'acuracia': 0.8, 'acuracia_extras': None, 'n_janelas': 4, 'n_janelas_extras': 0,
                       'pares': {'A->Am': {'n': 1, 'de': 3}}},
            'sem_acorde': {'gravado': 0.2, 'sintetico': 0.0, 'por_tipo': {'noise': {'falso_aceite': 0.0, 'n': 3}}}}


def _gravar(resultados, pasta, m):
    (resultados / pasta).mkdir(parents=True, exist_ok=True)
    (resultados / pasta / 'metrics.json').write_text(json.dumps(m), encoding='utf-8')


def test_main_so_resumo_gera_o_comparativo(resultados):
    _gravar(resultados, 'B11_s42', _metricas_falsas('B11', 42, 0.90))
    _gravar(resultados, 'B11_s43', _metricas_falsas('B11', 43, 0.94))
    _gravar(resultados, '0', dict(_metricas_falsas('0', None), seed=None))

    assert cm.main(['--so-resumo']) == 0

    texto = (resultados / 'comparativo.md').read_text(encoding='utf-8')
    assert '| 0 | - |' in texto
    assert '| B11 | 42, 43 | 92.0% (90.0–94.0) |' in texto
    assert '## B11: confusão' in texto and '1 de 3' in texto  # a tabela dos pares do B11
    assert 'Teste v3: 10 janelas' in texto


def test_resumo_sem_metricas_ainda_tem_o_como_ler(resultados):
    assert cm._ler_metricas() == {}
    assert '## Como ler' in cm.resumo({})


def test_celula_vazia_e_com_um_valor():
    assert cm._celula([None]) == '-'
    assert cm._celula([0.875]) == '87.5%'


def test_main_pula_o_que_ja_tem_metricas(resultados, monkeypatch):
    _gravar(resultados, 'A_s42', _metricas_falsas('A', 42))
    treinados, testes_montados, avaliados = [], [], []
    monkeypatch.setattr(cm, 'partes_de_teste', lambda: {})
    monkeypatch.setattr(cm, 'montar_testes', lambda partes, low_volume_so_v3=False:
                        testes_montados.append(low_volume_so_v3) or {'final': low_volume_so_v3})
    monkeypatch.setattr(cm, 'treinar', lambda nome, seed, testes, cache, n_aug=0:
                        treinados.append((nome, seed, testes['final'])))
    monkeypatch.setattr(cm, 'avaliar_modelo_atual', lambda testes: avaliados.append('0'))
    monkeypatch.setattr(cm, 'avaliar_tflite', lambda partes: avaliados.append('0c'))
    monkeypatch.setattr(cm, 'avaliar_ensemble', lambda nome, testes: avaliados.append(nome))

    cm.main(['--modelos', '0', '0c', 'A', 'F', '--seeds', '42', '43', '--ensemble', 'B11vs'])

    # A_s42 já existia; o F treina com os take02, então usa os testes "final"
    assert treinados == [('A', 43, False), ('F', 42, True), ('F', 43, True)]
    assert avaliados == ['B11vs', '0', '0c']
    assert testes_montados == [False, True]
    assert (resultados / 'comparativo.md').exists()
