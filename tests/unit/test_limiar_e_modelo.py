"""Limiar de confiança (calibrado na validação), construtores das CNNs e salvar/carregar o modelo."""
import json

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pipeline

CLASSES = ['C', 'G', '_none']


def _curva_pequena():
    # 3 janelas de acorde: C certo com 0,9; C errado (era G) com 0,6; _none com 0,7
    probs_acordes = np.array([[0.9, 0.05, 0.05], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    y_acordes = np.array(['C', 'G', 'C'])
    # 2 negativos: um confundido com C (0,5) e um reconhecido como _none
    probs_neg = np.array([[0.5, 0.2, 0.3], [0.1, 0.1, 0.8]])
    return pipeline.threshold_curve(probs_acordes, y_acordes, probs_neg, CLASSES, thresholds=[0.5, 0.7])


def test_curva_do_limiar_conta_certo():
    """Conta feita à mão: aceito = previu um acorde (não _none) com confiança >= limiar."""
    com_05, com_07 = _curva_pequena()

    # limiar 0,5: aceita as janelas 1 e 2 (a 3 previu _none) -> 2/3; acerta 1 de 2; o negativo 1 passa
    assert com_05 == pytest.approx({'threshold': 0.5, 'coverage': 2 / 3, 'accuracy': 0.5, 'false_accept': 0.5})
    # limiar 0,7: só a janela 1 passa, e está certa; nenhum negativo passa
    assert com_07 == pytest.approx({'threshold': 0.7, 'coverage': 1 / 3, 'accuracy': 1.0, 'false_accept': 0.0})


def test_limiar_maior_nunca_aumenta_cobertura_nem_falso_aceite():
    rng = np.random.default_rng(0)
    probs_acordes = rng.dirichlet(np.ones(3), size=200)
    probs_neg = rng.dirichlet(np.ones(3), size=100)
    y = np.array(CLASSES)[rng.integers(0, 2, 200)]

    linhas = pipeline.threshold_curve(probs_acordes, y, probs_neg, CLASSES)

    cobertura = [l['coverage'] for l in linhas]
    falso = [l['false_accept'] for l in linhas]
    assert cobertura == sorted(cobertura, reverse=True)
    assert falso == sorted(falso, reverse=True)


def _linha(t, cobertura, acuracia, falso):
    return {'threshold': t, 'coverage': cobertura, 'accuracy': acuracia, 'false_accept': falso}


def test_escolhe_maior_cobertura_dentro_das_regras():
    linhas = [
        _linha(0.3, 0.99, 0.97, 0.20),  # falso aceite alto demais
        _linha(0.5, 0.90, 0.985, 0.04),  # passa: maior cobertura entre os que passam
        _linha(0.7, 0.80, 0.99, 0.01),  # passa, mas cobre menos
    ]
    assert pipeline.choose_threshold(linhas) == pytest.approx(0.5)


def test_sem_nenhum_valido_escolhe_menor_falso_aceite():
    linhas = [_linha(0.3, 0.99, 0.90, 0.20), _linha(0.5, 0.90, 0.92, 0.10), _linha(0.7, 0.80, 0.95, 0.08)]
    assert pipeline.choose_threshold(linhas) == pytest.approx(0.7)


@pytest.mark.parametrize('arch', ['flatten', 'gap_bn', 'cnn2d', 'baseline'])
def test_cnn_devolve_uma_probabilidade_por_classe(arch):
    modelo = pipeline.build_model(arch, (pipeline.MAX_PAD_LEN, 12), n_classes=9)
    x = np.random.default_rng(0).standard_normal((2, pipeline.MAX_PAD_LEN, 12)).astype(np.float32)

    probs = modelo.predict(x, verbose=0)

    assert probs.shape == (2, 9)
    np.testing.assert_allclose(probs.sum(axis=1), 1, rtol=1e-5)  # softmax


def test_callbacks_olham_a_validacao():
    paradas = pipeline.training_callbacks()
    assert [type(c).__name__ for c in paradas] == ['EarlyStopping', 'ReduceLROnPlateau']
    assert all(c.monitor == 'val_loss' for c in paradas)


def test_set_seeds_deixa_o_sorteio_igual():
    pipeline.set_seeds(7)
    a = np.random.rand(3)
    pipeline.set_seeds(7)
    assert np.array_equal(a, np.random.rand(3))


@pytest.fixture
def pacote(tmp_path):
    """Modelo pequeno salvo com encoder, scaler e model_config.json."""
    modelo = pipeline.build_model('gap_bn', (pipeline.MAX_PAD_LEN, 12), len(CLASSES))
    encoder = LabelEncoder().fit(CLASSES)
    scaler = StandardScaler().fit(np.random.default_rng(0).standard_normal((50, 12)))
    config = pipeline.make_model_config('chroma_stft8192', 'feature', 'gap_bn', encoder.classes_, True, 0.5)
    pipeline.save_model_bundle(modelo, encoder, scaler, config, str(tmp_path), metrics={'accuracy': 0.9})
    return modelo, tmp_path


def test_salvar_e_carregar_da_as_mesmas_previsoes(pacote):
    modelo, pasta = pacote
    assert pipeline.model_bundle_exists(str(pasta))
    assert json.loads((pasta / 'metrics.json').read_text()) == {'accuracy': 0.9}

    carregado, encoder, _, config = pipeline.load_model_bundle(str(pasta))

    x = np.random.default_rng(1).standard_normal((3, pipeline.MAX_PAD_LEN, 12)).astype(np.float32)
    np.testing.assert_allclose(carregado.predict(x, verbose=0), modelo.predict(x, verbose=0), rtol=1e-5)
    assert config['confidence_threshold'] == pytest.approx(0.5) and list(encoder.classes_) == CLASSES


def test_classes_do_config_diferentes_do_encoder_da_erro(pacote):
    _, pasta = pacote
    caminho = pasta / pipeline.CONFIG_FILENAME
    config = json.loads(caminho.read_text())
    config['classes'] = ['C', 'Dm', '_none']
    caminho.write_text(json.dumps(config))

    with pytest.raises(ValueError, match='classes'):
        pipeline.load_model_bundle(str(pasta))


def test_pasta_sem_o_modelo_nao_existe(tmp_path):
    assert not pipeline.model_bundle_exists(str(tmp_path))


def test_config_com_feature_desconhecida_da_erro():
    config = pipeline.make_model_config('chroma', 'feature', 'gap_bn', CLASSES, True, 0.5)
    config['feature'] = 'espectrograma_inventado'
    with pytest.raises(ValueError, match='feature desconhecida'):
        pipeline.validate_config(config)
