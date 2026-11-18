"""Extratores de feature, scaler e one-hot dos rótulos."""
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import pipeline
from tests.sintetico import NOTAS_CHROMA, acorde

SR = pipeline.SAMPLE_RATE
FRAMES = pipeline.MAX_PAD_LEN  # 65


@pytest.mark.parametrize('feature', ['mfcc', 'chroma', 'chroma_cqt_t0', 'chroma_stft',
                                     'chroma_stft4096', 'chroma_stft8192', 'cqt', 'mfcc+chroma'])
def test_extrator_devolve_formato_certo(feature):
    feats = pipeline.EXTRACTORS[feature](acorde('C'))
    assert feats.shape == (pipeline.N_FEATURES[feature], FRAMES)
    assert np.all(np.isfinite(feats))


@pytest.mark.parametrize('feature', ['mfcc', 'chroma', 'chroma_cqt_t0', 'chroma_stft', 'cqt', 'mfcc+chroma'])
def test_audio_curto_demais_devolve_none(feature):
    curto = acorde('C', segundos=0.01)  # abaixo de MIN_VALID_DURATION (0,05 s)
    assert pipeline.EXTRACTORS[feature](curto) is None


def test_mfcc_le_direto_do_arquivo(tmp_path):
    import soundfile as sf
    arq = tmp_path / 'c.wav'
    sf.write(str(arq), acorde('C'), SR)
    assert pipeline.extract_features(str(arq)).shape == (pipeline.N_MFCC, FRAMES)


def test_mfcc_de_arquivo_que_nao_existe_devolve_none(tmp_path):
    assert pipeline.extract_features(str(tmp_path / 'nao_existe.wav')) is None


def test_audio_mais_longo_e_cortado_em_65_frames():
    assert pipeline.extract_chroma(acorde('C', segundos=4.0)).shape == (12, FRAMES)


@pytest.mark.parametrize('feature', ['chroma', 'chroma_stft8192'])
@pytest.mark.parametrize('nome', ['C', 'G', 'Am', 'Em'])
def test_chroma_destaca_as_notas_do_acorde(feature, nome):
    """As 3 notas mais fortes do chroma são as 3 notas do acorde (ex.: C = C, E, G)."""
    chroma = pipeline.EXTRACTORS[feature](acorde(nome))
    tres_maiores = set(np.argsort(chroma.mean(axis=1))[-3:])
    assert tres_maiores == NOTAS_CHROMA[nome]


def _features_falsas(n=50, seed=0):
    rng = np.random.default_rng(seed)
    # 12 features com médias e escalas bem diferentes, 65 frames
    return rng.standard_normal((n, 12, FRAMES)) * np.arange(1, 13)[None, :, None] + np.arange(12)[None, :, None] * 10


def test_scaler_por_feature_normaliza_cada_bin():
    X = _features_falsas()
    X_esc, scaler = pipeline.fit_scaler(X, 'feature')

    assert X_esc.shape == (50, FRAMES, 12)  # pronto pra CNN: (n, frames, features)
    por_bin = X_esc.reshape(-1, 12)
    np.testing.assert_allclose(por_bin.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(por_bin.std(axis=0), 1, atol=1e-6)
    assert len(scaler.mean_) == 12


def test_scaler_por_frame_normaliza_cada_instante():
    X = _features_falsas()
    X_esc, scaler = pipeline.fit_scaler(X, 'frame')

    assert X_esc.shape == (50, FRAMES, 12)
    por_frame = np.swapaxes(X_esc, 1, 2).reshape(-1, FRAMES)
    np.testing.assert_allclose(por_frame.mean(axis=0), 0, atol=1e-8)
    assert len(scaler.mean_) == FRAMES


def test_scaler_usa_so_a_media_do_treino():
    """O teste é escalado com a média do treino: se os dados mudam, a média não fica 0."""
    _, scaler = pipeline.fit_scaler(_features_falsas(seed=0), 'feature')
    teste = _features_falsas(seed=1) + 5.0
    teste_esc = pipeline.apply_scaler(scaler, teste, 'feature')
    assert teste_esc.reshape(-1, 12).mean() > 1


def test_encode_labels_gera_one_hot():
    classes = ['Am', 'C', 'G', '_none']
    encoder, y_treino, y_teste = pipeline.encode_labels(classes, ['C', '_none', 'C'], ['Am'])

    assert list(encoder.classes_) == classes
    assert y_treino.shape == (3, 4) and y_teste.shape == (1, 4)
    np.testing.assert_array_equal(y_treino.sum(axis=1), 1)  # exatamente um 1 por linha
    assert y_treino[1].argmax() == 3  # _none é a última classe
    assert y_teste[0].argmax() == 0


def test_features_for_model_reamostra_outra_taxa():
    """Áudio a 32 kHz é reamostrado pra 22.050 Hz antes do chroma. Sem isso as notas
    sairiam ~6 semitons fora e o C viraria outro acorde."""
    identidade = StandardScaler(with_mean=False, with_std=False).fit(np.zeros((2, 12)))
    config = pipeline.make_model_config('chroma_stft8192', 'feature', 'gap_bn', ['C'], False)

    x = pipeline.features_for_model(acorde('C', sr=32000), identidade, config, sample_rate=32000)

    assert x.shape == (1, FRAMES, 12)
    assert set(np.argsort(x[0].mean(axis=0))[-3:]) == NOTAS_CHROMA['C']


def test_features_for_model_audio_curto_devolve_none():
    identidade = StandardScaler(with_mean=False, with_std=False).fit(np.zeros((2, 12)))
    config = pipeline.make_model_config('chroma', 'feature', 'gap_bn', ['C'], False)
    assert pipeline.features_for_model(acorde('C', segundos=0.01), identidade, config) is None
