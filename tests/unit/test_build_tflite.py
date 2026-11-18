"""export/build_tflite.py sem os modelos dos experimentos: o grafo com a chroma_stft
dentro dá a mesma feature do librosa, a conversão só usa ops nativas do TFLite, o
int8 não mexe na janela nem na matriz de chroma, e os arquivos do app saem certos.
Usa uma CNN pequena com pesos aleatórios: o que se testa aqui é a exportação, não a
acurácia."""
import hashlib
import json
import os

import numpy as np
import pytest
import soundfile as sf
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pipeline as pl
from export import build_tflite as bt
from tests.sintetico import acorde

FEATURE = 'chroma_stft8192'
CLASSES = ['C', 'G', '_none']


@pytest.fixture(scope='module')
def exportacao():
    """Modelo, scaler, o grafo de ponta a ponta e as duas versões convertidas."""
    pl.set_seeds(0)
    janelas = np.stack([acorde(n, seed=s) for n in ('C', 'G') for s in range(4)])
    feats = np.stack([pl.EXTRACTORS[FEATURE](a) for a in janelas])
    _, scaler = pl.fit_scaler(feats, 'feature')
    model = pl.build_model('gap_bn', (pl.MAX_PAD_LEN, 12), len(CLASSES))
    e2e = bt.EndToEnd(model, scaler, FEATURE)
    return model, scaler, e2e, {'float32': bt.convert(e2e, False), 'int8': bt.convert(e2e, True)}


def _dados(n=6):
    """No formato do load_test_audio: janelas de acorde, degradadas e negativas."""
    clean = np.stack([acorde('C' if i % 2 == 0 else 'G', seed=i) for i in range(n)]).astype(np.float32)
    negs = np.stack([pl.make_negative_audio('noise', str(i)) for i in range(3)])
    return {'clean': clean, 'degraded': clean * 0.5, 'labels': np.array(['C', 'G'] * (n // 2)),
            'rms_clean': np.array([pl.rms(x) for x in clean]), 'rms_degraded': np.array([pl.rms(x) for x in clean * 0.5]),
            'negatives': negs, 'neg_kinds': np.array(['noise'] * 3), 'rms_negatives': np.array([pl.rms(x) for x in negs])}


def test_chroma_do_grafo_e_a_mesma_do_librosa(exportacao, monkeypatch):
    model, scaler, e2e, _ = exportacao
    monkeypatch.setattr(bt, 'N_EQUIV_WINDOWS', 6)
    r = bt.equivalence(e2e, model, scaler, FEATURE, _dados())
    assert r['ok'] and r['class_agreement'] == pytest.approx(1.0)
    assert r['feature_abs_err_max'] < 1e-4  # no export de verdade deu ~8e-7
    assert r['prob_abs_err_max'] < 1e-4


def test_grafo_espera_os_65_frames_de_1_5_s(exportacao):
    _, _, e2e, _ = exportacao
    assert (e2e.n_fft, e2e.hop, e2e.n_frames, e2e.pad) == (8192, 512, 65, 4096)
    assert bt.WINDOW_SAMPLES == 33075


def test_conversao_so_usa_ops_nativas(exportacao):
    _, _, _, blobs = exportacao
    for blob in blobs.values():
        ops = bt.op_names(blob)
        assert 'RFFT2D' in ops  # a STFT virou a FFT nativa do TFLite
        assert not [op for op in ops if op.startswith('FLEX') or 'CUSTOM' in op]


def test_int8_nao_quantiza_a_janela_nem_a_matriz_de_chroma(exportacao):
    _, _, e2e, blobs = exportacao
    formas = [s for _, s in bt.quantized_constants(blobs['int8'])]
    assert formas  # os pesos da CNN viraram int8
    n_bins = e2e.n_fft // 2 + 1
    for proibida in ((12, n_bins), (n_bins, 12), (e2e.n_fft,)):
        assert proibida not in formas
    assert bt.quantized_constants(blobs['float32']) == []


def test_tflite_da_as_mesmas_probabilidades_do_grafo(exportacao):
    _, _, e2e, blobs = exportacao
    audio = _dados(4)['clean']
    esperado = e2e.batch_probs(audio).numpy()
    np.testing.assert_allclose(bt.tflite_probs(blobs['float32'], audio), esperado, atol=1e-5)
    np.testing.assert_allclose(bt.tflite_probs(blobs['int8'], audio), esperado, atol=0.05)
    assert bt.tflite_latency_ms(blobs['float32'], audio[0], n=3) > 0


def test_metricas_contadas_a_mao():
    dados = {'labels': np.array(['C', 'C', 'G', 'G']), 'rms_clean': np.array([0.1, 0.1, 0.1, 0.001]),
             'rms_degraded': np.full(4, 0.1), 'rms_negatives': np.array([0.1, 0.1])}
    p = lambda i, c: np.eye(3)[i] * c + (1 - c) / 3
    clean = np.array([p(0, 0.9), p(1, 0.9), p(1, 0.15), p(1, 0.9)])  # C certo, C errado, G fraco, G em silêncio
    neg = np.array([p(2, 0.9), p(0, 0.9)])  # um _none certo, um aceito como C
    m = bt.metrics_from_probs(clean, clean, neg, dados, CLASSES, threshold=0.5)
    assert m['accuracy_chords'] == pytest.approx(0.75)  # argmax, sem limiar: errou só o segundo
    assert m['coverage'] == pytest.approx(2 / 3)  # das 3 fora do silêncio, mostrou 2
    assert m['accuracy_accepted'] == pytest.approx(0.5)
    assert m['none_recall'] == pytest.approx(0.5) and m['false_accept'] == pytest.approx(0.5)


def test_pcm16_como_o_app_le_o_wav():
    x = np.array([0.0, 0.5, -0.5, 1.2, -1.2], dtype=np.float32)
    y = bt.pcm16_roundtrip(x)
    assert np.abs(y[:3] - x[:3]).max() <= 1 / 32768
    assert y[3] == pytest.approx(32767 / 32768) and y[4] == pytest.approx(-1.0)  # satura


def test_arquivos_do_app(exportacao, tmp_path, monkeypatch):
    _, _, _, blobs = exportacao
    monkeypatch.setattr(bt, 'EXPORT_DIR', str(tmp_path))
    dados = _dados()
    cfg = pl.make_model_config(FEATURE, 'feature', 'gap_bn', CLASSES, True, 0.5,
                               extra={'seed': 43, 'model_version': 'teste-s43', 'source_experiment': 'x'})
    # a janela 2 é um C "acertado" com confiança no meio (0,6 a 0,95), a que vira o example.wav
    probs = np.tile([0.1, 0.8, 0.1], (len(dados['clean']), 1))
    probs[2] = [0.8, 0.1, 0.1]

    esperado = bt.save_app_files(blobs['float32'], blobs['int8'], cfg, dados, {'validacao': {'coverage': 0.9}}, probs)

    info = json.loads((tmp_path / 'model_info.json').read_text(encoding='utf-8'))
    for nome, arquivo in info['files'].items():
        assert info['sha256'][nome] == hashlib.sha256((tmp_path / arquivo).read_bytes()).hexdigest()
    assert info['input']['shape'] == [1, 33075] and info['confidence_threshold'] == pytest.approx(0.5)
    assert info['feature_inside_model']['norm'] == 'inf'
    assert json.loads((tmp_path / 'labels.json').read_text(encoding='utf-8')) == CLASSES
    audio, sr = sf.read(str(tmp_path / 'example.wav'), dtype='int16')
    assert sr == pl.SAMPLE_RATE and len(audio) == 33075
    np.testing.assert_allclose(audio / 32768.0, bt.pcm16_roundtrip(dados['clean'][2]), atol=1 / 32768)
    exemplo = json.loads((tmp_path / 'example_expected.json').read_text(encoding='utf-8'))
    assert exemplo['true_label'] == 'C' and set(exemplo['float32']['probabilities']) == set(CLASSES)
    assert esperado['float32']['top_class'] == exemplo['float32']['top_class']


def _salvar_semente(pasta, seed, acc, loss, model, scaler):
    run = pasta / f'GRUPO_s{seed}'
    cfg = pl.make_model_config(FEATURE, 'feature', 'gap_bn', CLASSES, True, 0.55, extra={'seed': seed})
    pl.save_model_bundle(model, LabelEncoder().fit(CLASSES), scaler, cfg, str(run), metrics={
        'val_acuracia_acordes': acc, 'best_val_loss': loss, 'limiar': 0.55,
        'validacao_no_limiar': {'coverage': 0.9}, 'sem_acorde': {'sintetico': 0.01}})


def test_seed_escolhida_pela_validacao_nas_duas_fontes(exportacao, tmp_path, monkeypatch):
    model, scaler, _, _ = exportacao
    monkeypatch.setattr(bt, 'DUAS_FONTES_DIR', str(tmp_path))
    monkeypatch.setattr(bt, 'DUAS_FONTES', 'GRUPO')
    _salvar_semente(tmp_path, 42, 0.90, 0.30, model, scaler)
    _salvar_semente(tmp_path, 43, 0.92, 0.40, model, scaler)
    _salvar_semente(tmp_path, 44, 0.92, 0.35, model, scaler)  # empata com a 43, mas tem val_loss menor

    seed, escolha = bt.choose_seed()
    assert seed == 44 and escolha['chosen_seed'] == 44
    assert 'O teste nao participa' in escolha['criterion']

    _, encoder, _, cfg, per_seed = bt.load_source(44)
    assert cfg['model_version'] == 'grupo-s44' and cfg['confidence_threshold'] == pytest.approx(0.55)
    assert list(encoder.classes_) == CLASSES and per_seed['teste']['false_accept'] == pytest.approx(0.01)


def test_audio_de_teste_no_formato_que_o_app_recebe(tmp_path, monkeypatch):
    arquivos = []
    for nome in ('C', 'G'):
        caminho = tmp_path / f'{nome}.wav'
        sf.write(str(caminho), acorde(nome, segundos=3.0), pl.SAMPLE_RATE)
        arquivos.append((str(caminho), nome))
    monkeypatch.setattr(bt, '_test_files', lambda: arquivos)

    d = bt.load_test_audio()

    assert d['clean'].shape == (6, 33075) and d['clean'].dtype == np.float32  # 2 arquivos x 3 janelas
    assert d['degraded'].shape == d['clean'].shape and not np.array_equal(d['degraded'], d['clean'])
    assert list(d['labels']) == ['C'] * 3 + ['G'] * 3
    assert len(d['negatives']) == len(d['neg_kinds']) == len(d['rms_negatives'])


def test_main_exporta_e_grava_o_relatorio(exportacao, tmp_path, monkeypatch):
    model, scaler, _, _ = exportacao
    cfg = pl.make_model_config(FEATURE, 'feature', 'gap_bn', CLASSES, True, 0.5, extra={'seed': 42})
    monkeypatch.setattr(bt, 'EXPORT_DIR', str(tmp_path))
    monkeypatch.setattr(bt, 'N_EQUIV_WINDOWS', 4)
    monkeypatch.setattr(bt, 'choose_seed', lambda: (42, {'chosen_seed': 42}))
    monkeypatch.setattr(bt, 'load_source', lambda seed: (model, LabelEncoder().fit(CLASSES), scaler, cfg,
                                                         {'teste': {'false_accept': 0.01}}))
    monkeypatch.setattr(bt, 'load_test_audio', lambda: _dados(4))
    monkeypatch.setattr(bt, 'save_app_files', lambda *a: {'float32': 'ok'})

    bt.main([])

    r = json.loads((tmp_path / 'export_report.json').read_text(encoding='utf-8'))
    assert r['equivalence']['ok'] and r['int8_touched_window_or_chroma'] is False
    assert set(r['metrics']) == {'keras', 'float32', 'int8'}
    assert r['agreement_with_keras']['float32'] == pytest.approx(1.0)
    assert r['source']['seed'] == 42 and 'tensorflow' in r['source']['versions']


def test_main_para_se_o_falso_aceite_da_seed_passa_do_limite(exportacao, monkeypatch):
    model, scaler, _, _ = exportacao
    cfg = pl.make_model_config(FEATURE, 'feature', 'gap_bn', CLASSES, True, 0.5, extra={'seed': 42})
    monkeypatch.setattr(bt, 'choose_seed', lambda: (42, {}))
    monkeypatch.setattr(bt, 'load_source', lambda seed: (model, None, scaler, cfg, {'teste': {'false_accept': 0.10}}))
    with pytest.raises(SystemExit, match='falso aceite'):
        bt.main([])
