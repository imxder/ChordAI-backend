"""Tratamento de "nenhum acorde" (Parte 5, aplicado na Parte 6) em
recognize_chord() e predict_note(): a classe "_none" e a confianca abaixo do
limiar viram desconhecido. O limiar e o rotulo "_none" vem do model_config.json
do modelo (aqui, um config montado no teste); sem config vale o pipeline antigo
(sem _none e sem limiar).
"""
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

import pipeline
import train_model

CLASSES = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G', '_none']
THRESHOLD = 0.45


class FixedModel:
    """Modelo falso: sempre devolve a mesma distribuicao de probabilidades."""

    def __init__(self, top_class, confidence, classes=CLASSES):
        probs = np.full(len(classes), (1.0 - confidence) / (len(classes) - 1))
        probs[classes.index(top_class)] = confidence
        self.probs = probs

    def predict(self, x, verbose=0):
        return self.probs[np.newaxis, :]


class IdentityScaler:
    def transform(self, x):
        return x


def make_config(threshold=THRESHOLD, with_none=True, classes=CLASSES):
    # o modelo e falso, entao a feature so precisa ser valida; mfcc + scaler por frame
    return pipeline.make_model_config('mfcc', 'frame', 'flatten', classes, with_none, threshold)


@pytest.fixture
def encoder():
    """Encoder com o model_config anexado, como faz load_trained_model()."""
    enc = LabelEncoder()
    enc.fit(CLASSES)
    enc.model_config_ = make_config()
    return enc


@pytest.fixture
def audio():
    """1,5 s de sinal bem acima do limiar de silencio (o conteudo nao importa)."""
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(train_model.SAMPLE_RATE * 1.5))).astype(np.float32)


HIGH = 0.95
LOW = THRESHOLD - 0.1


# --- recognize_chord (API HTTP / Flutter) ---------------------------------

def test_recognize_chord_none_class_is_unknown(audio, encoder):
    result = train_model.recognize_chord(audio, FixedModel('_none', HIGH), encoder, IdentityScaler())
    assert result['status'] == 'unknown'
    assert result['chord'] is None


def test_recognize_chord_low_confidence_is_unknown(audio, encoder):
    result = train_model.recognize_chord(audio, FixedModel('C', LOW), encoder, IdentityScaler())
    assert result['status'] == 'unknown'
    assert result['chord'] is None
    assert result['confidence'] == pytest.approx(LOW, abs=1e-4)


def test_recognize_chord_confident_chord_is_ok(audio, encoder):
    result = train_model.recognize_chord(audio, FixedModel('C', HIGH), encoder, IdentityScaler())
    assert result == {'status': 'ok', 'chord': 'C', 'confidence': pytest.approx(HIGH, abs=1e-4)}


def test_recognize_chord_silence_unchanged(encoder):
    silence = np.zeros(int(train_model.SAMPLE_RATE * 1.5), dtype=np.float32)
    result = train_model.recognize_chord(silence, FixedModel('C', HIGH), encoder, IdentityScaler())
    assert result == {'status': 'silence', 'chord': None, 'confidence': 0.0}


# --- predict_note (interface web do app.py) -------------------------------

def test_predict_note_none_class_is_unknown(audio, encoder):
    assert train_model.predict_note(audio, FixedModel('_none', HIGH), encoder, IdentityScaler()) == 'Acorde Desconhecido'


def test_predict_note_low_confidence_is_unknown(audio, encoder):
    assert train_model.predict_note(audio, FixedModel('Em', LOW), encoder, IdentityScaler()) == 'Acorde Desconhecido'


def test_predict_note_confident_chord_returns_name(audio, encoder):
    assert train_model.predict_note(audio, FixedModel('Em', HIGH), encoder, IdentityScaler()) == 'Em'


# --- o limiar e o rotulo _none vem do model_config -------------------------

def test_threshold_comes_from_config(audio, encoder):
    model = FixedModel('G', 0.6)
    assert train_model.predict_note(audio, model, encoder, IdentityScaler(), config=make_config(0.5)) == 'G'
    assert train_model.predict_note(audio, model, encoder, IdentityScaler(), config=make_config(0.7)) == 'Acorde Desconhecido'
    assert train_model.recognize_chord(audio, model, encoder, IdentityScaler(), config=make_config(0.7))['status'] == 'unknown'


def test_legacy_model_without_config_has_no_threshold(audio):
    """Modelo antigo (sem model_config.json): sem _none e sem limiar, como antes."""
    classes = CLASSES[:-1]
    enc = LabelEncoder().fit(classes)  # sem model_config_ anexado
    model = FixedModel('C', 0.2, classes)
    assert train_model.recognize_chord(audio, model, enc, IdentityScaler())['status'] == 'ok'
    assert train_model.predict_note(audio, model, enc, IdentityScaler()) == 'C'
