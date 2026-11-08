"""Testes do tratamento de "nenhum acorde" (Parte 5) em recognize_chord() e
predict_note(): classe "_none" e confianca abaixo do limiar viram desconhecido.

Ficam PULADOS enquanto experiments/results/parte5_recognize_chord.diff nao for
aplicado (Parte 6, junto com a troca do modelo em producao).
"""
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

import train_model

pytestmark = pytest.mark.skipif(
    not hasattr(train_model, 'CONFIDENCE_THRESHOLD'),
    reason='diff da Parte 5 (experiments/results/parte5_recognize_chord.diff) ainda nao aplicado',
)

CLASSES = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G', '_none']


class FixedModel:
    """Modelo falso: sempre devolve a mesma distribuicao de probabilidades."""

    def __init__(self, top_class, confidence):
        probs = np.full(len(CLASSES), (1.0 - confidence) / (len(CLASSES) - 1))
        probs[CLASSES.index(top_class)] = confidence
        self.probs = probs

    def predict(self, x, verbose=0):
        return self.probs[np.newaxis, :]


class IdentityScaler:
    def transform(self, x):
        return x


@pytest.fixture
def encoder():
    enc = LabelEncoder()
    enc.fit(CLASSES)
    return enc


@pytest.fixture
def audio():
    """1,5 s de sinal bem acima do limiar de silencio (o conteudo nao importa:
    o modelo e falso)."""
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(train_model.SAMPLE_RATE * 1.5))).astype(np.float32)


HIGH = 0.95
LOW = train_model.CONFIDENCE_THRESHOLD - 0.1 if hasattr(train_model, 'CONFIDENCE_THRESHOLD') else 0.0


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


# --- predict_note (interface web do app.py) -------------------------------

def test_predict_note_none_class_is_unknown(audio, encoder):
    assert train_model.predict_note(audio, FixedModel('_none', HIGH), encoder, IdentityScaler()) == 'Acorde Desconhecido'


def test_predict_note_low_confidence_is_unknown(audio, encoder):
    assert train_model.predict_note(audio, FixedModel('Em', LOW), encoder, IdentityScaler()) == 'Acorde Desconhecido'


def test_predict_note_confident_chord_returns_name(audio, encoder):
    assert train_model.predict_note(audio, FixedModel('Em', HIGH), encoder, IdentityScaler()) == 'Em'


def test_threshold_can_be_overridden(audio, encoder):
    model = FixedModel('G', 0.6)
    assert train_model.predict_note(audio, model, encoder, IdentityScaler(), confidence_threshold=0.5) == 'G'
    assert train_model.predict_note(audio, model, encoder, IdentityScaler(), confidence_threshold=0.7) == 'Acorde Desconhecido'
    assert train_model.recognize_chord(audio, model, encoder, IdentityScaler(), confidence_threshold=0.7)['status'] == 'unknown'
