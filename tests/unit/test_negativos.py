"""Negativos da classe "_none": áudios que NÃO são acorde (volume baixo, ruído, batidas)."""
import numpy as np
import pytest

import pipeline
from tests.sintetico import gravar_dataset


@pytest.fixture
def arquivo_de_acorde(tmp_path):
    gravar_dataset(str(tmp_path), acordes=('C',), por_acorde=1)
    return str(tmp_path / 'C' / 'C_00.wav')


@pytest.mark.parametrize('tipo,origem', [('noise', '3'), ('percussion', '7')])
def test_negativo_e_sempre_o_mesmo(tipo, origem):
    """A mesma origem gera o mesmo áudio: o treino é reproduzível em qualquer máquina."""
    a = pipeline.make_negative_audio(tipo, origem)
    b = pipeline.make_negative_audio(tipo, origem)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, pipeline.make_negative_audio(tipo, str(int(origem) + 1)))


@pytest.mark.parametrize('tipo', ['noise', 'percussion'])
def test_negativo_tem_1_5_s_e_fica_entre_menos_1_e_1(tipo):
    for origem in range(20):
        x = pipeline.make_negative_audio(tipo, str(origem))
        assert len(x) == int(pipeline.SAMPLE_RATE * pipeline.WINDOW_SECONDS)
        assert x.dtype == np.float32
        assert np.max(np.abs(x)) <= 1.0


def test_ruido_passa_do_limiar_de_silencio():
    """Se o ruído ficasse abaixo do limiar, a API devolveria "silence" e o modelo nunca
    aprenderia a dizer "_none" pra ele."""
    for origem in range(30):
        assert pipeline.rms(pipeline.make_negative_audio('noise', str(origem))) >= pipeline.SILENCE_THRESHOLD


def test_volume_baixo_passa_do_silencio_mas_fica_baixo(arquivo_de_acorde):
    x = pipeline.make_negative_audio('low_volume', arquivo_de_acorde)
    r = pipeline.rms(x)
    assert pipeline.SILENCE_THRESHOLD < r <= 0.012
    assert r < pipeline.rms(pipeline.make_windowed_segments(arquivo_de_acorde)[0])  # mais baixo que o original


def test_percussao_tem_picos():
    """Batidas são curtas e fortes: o pico fica bem acima da média (fator de crista alto)."""
    x = pipeline.make_negative_audio('percussion', '1')
    assert np.max(np.abs(x)) / pipeline.rms(x) > 4


def test_ids_sinteticos_divididos_sem_sobrepor():
    treino = pipeline.synth_negative_ids('train')
    val = pipeline.synth_negative_ids('val')
    teste = pipeline.synth_negative_ids('test')

    assert len(teste) == 74  # 20% de 370
    assert len(val) == 44    # 15% dos 296 restantes
    assert len(treino) == 370 - 74 - 44
    assert not set(treino) & set(val) and not set(treino) & set(teste) and not set(val) & set(teste)
    assert sorted(treino + val + teste) == list(range(370))


def test_specs_tem_um_volume_baixo_a_cada_4_arquivos():
    arquivos = [(f'dataset/C/c{i}.wav', 'C') for i in range(10)]
    specs = pipeline.negative_specs('test', arquivos)
    baixos = [s for k, s in specs if k == 'low_volume']
    assert baixos == ['dataset/C/c0.wav', 'dataset/C/c4.wav', 'dataset/C/c8.wav']
    assert sum(1 for k, _ in specs if k == 'noise') == 74
    assert sum(1 for k, _ in specs if k == 'percussion') == 74


def test_chave_de_origem_igual_no_windows_e_no_linux():
    """O volume baixo usa o caminho do arquivo como semente; com '\\' fixo, o Windows e o
    Linux geram os mesmos negativos (e o digest bate)."""
    caminho = pipeline.ROOT_DIR + '/dataset/C/c0.wav'
    assert pipeline._source_key('low_volume', caminho) == 'dataset\\C\\c0.wav'
    assert pipeline._source_key('noise', '5') == '5'
    d = pipeline.negative_specs_digest('val', [(caminho, 'C')])
    assert d == pipeline.negative_specs_digest('val', [(caminho, 'C')]) and len(d) == 10


def test_build_negatives_extrai_features(arquivo_de_acorde, monkeypatch):
    original = pipeline.synth_negative_ids
    monkeypatch.setattr(pipeline, 'synth_negative_ids', lambda split: original(split, n=10))

    X, tipos, rms = pipeline.build_negatives('test', [(arquivo_de_acorde, 'C')], 'chroma_stft8192', n_jobs=1)

    assert X.shape == (5, 12, pipeline.MAX_PAD_LEN)  # 1 volume baixo + 2 ruídos + 2 batidas
    assert list(tipos) == ['low_volume', 'noise', 'noise', 'percussion', 'percussion']
    assert np.all(rms > 0)


def test_load_dataset_com_none(tmp_path, monkeypatch):
    gravar_dataset(str(tmp_path), por_acorde=10)
    original = pipeline.synth_negative_ids
    monkeypatch.setattr(pipeline, 'synth_negative_ids', lambda split: original(split, n=20))  # menos, pra ser rápido

    d = pipeline.load_dataset('chroma_stft8192', with_none=True, dataset_path=str(tmp_path), verbose=False)

    assert d['labels'] == ['Am', 'C', 'Em', 'G', '_none']
    for split in ('train', 'val', 'test'):
        assert len(d[f'X_{split}']) == len(d[f'y_{split}'])
        assert '_none' in d[f'y_{split}']
        n_acordes = int((d[f'y_{split}'] != '_none').sum())
        assert len(d[f'rms_{split}']) == n_acordes  # RMS só das janelas de acorde
    assert d['n_train_files'] + d['n_val_files'] + len(d['test_files']) == 40
    assert d['none_counts']['test'] == {'low_volume': 2, 'noise': 4, 'percussion': 4}
    assert list(d['test_none_kinds']).count('noise') == 4


def test_load_dataset_sem_none_nao_tem_none(tmp_path):
    gravar_dataset(str(tmp_path), acordes=('C', 'G'), por_acorde=5)
    d = pipeline.load_dataset('mfcc', dataset_path=str(tmp_path), verbose=False)
    assert d['labels'] == ['C', 'G']
    assert '_none' not in d['y_train']
    assert d['X_train'].shape[1:] == (pipeline.N_MFCC, pipeline.MAX_PAD_LEN)
