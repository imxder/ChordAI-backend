"""chroma_stft (Parte 7a): a cadeia explicita de pipeline.py tem que ser igual a
librosa.feature.chroma_stft com os mesmos parametros e tuning=0.0. Se uma versao
nova do librosa mudar algum padrao interno, este teste quebra antes da Parte 7b."""
import librosa
import numpy as np
import pytest

import pipeline as pl


def _audio(seed=0, seconds=pl.WINDOW_SECONDS):
    rng = np.random.default_rng(seed)
    t = np.arange(int(pl.SAMPLE_RATE * seconds)) / pl.SAMPLE_RATE
    # acorde de C (C4, E4, G4) + ruido, para os 12 bins nao ficarem triviais
    y = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.00))
    return (0.2 * y + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def _librosa_reference(y):
    return librosa.feature.chroma_stft(
        y=y, sr=pl.SAMPLE_RATE, n_fft=2048, hop_length=512, win_length=2048, window='hann',
        center=True, pad_mode='constant', tuning=0.0, n_chroma=12, norm=np.inf)


@pytest.mark.parametrize('seed', [0, 1])
def test_explicit_chain_matches_librosa(seed):
    y = _audio(seed)
    np.testing.assert_allclose(pl.chroma_stft_raw(y), _librosa_reference(y), rtol=1e-5, atol=1e-6)


def test_shape_and_extractor():
    feats = pl.extract_chroma_stft(_audio())
    assert feats.shape == (12, pl.MAX_PAD_LEN) == (12, 65)
    assert pl.EXTRACTORS['chroma_stft'] is pl.extract_chroma_stft
    assert pl.N_FEATURES['chroma_stft'] == 12
    # norm=inf: o maximo de cada frame com som e 1
    np.testing.assert_allclose(feats.max(axis=0), 1.0, rtol=1e-6)


def test_c_major_peaks_on_c_e_g():
    """Sanidade do banco de filtros (base_c=True: bin 0 = C)."""
    mean = pl.extract_chroma_stft(_audio()).mean(axis=1)
    assert set(np.argsort(mean)[-3:]) == {0, 4, 7}  # C, E, G


def test_silence_stays_zero():
    feats = pl.extract_chroma_stft(np.zeros(int(pl.SAMPLE_RATE * pl.WINDOW_SECONDS), dtype=np.float32))
    assert np.all(feats == 0)


def test_constants_are_explicit():
    assert pl.CHROMA_STFT_TUNING == 0.0
    assert pl.CHROMA_STFT_PAD_MODE == 'constant'
    assert (pl.CHROMA_STFT_N_FFT, pl.CHROMA_STFT_HOP_LENGTH, pl.CHROMA_STFT_WIN_LENGTH) == (2048, 512, 2048)
    assert pl.CHROMA_STFT_POWER == 2.0 and pl.CHROMA_STFT_NORM == np.inf
