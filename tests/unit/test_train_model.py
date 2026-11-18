"""train_model(): treino de ponta a ponta num dataset sintético pequeno, e os casos de erro
do carregamento e da inferência."""
import json
import os
import shutil

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

import pipeline
import train_model
from tests.sintetico import acorde, gravar_dataset

ROOT_MODELO = os.path.join(pipeline.ROOT_DIR, 'trained_model')


@pytest.fixture(scope='module')
def treinado(tmp_path_factory):
    """Treina uma vez (gap_bn + chroma_stft8192 + _none) e reaproveita nos testes do arquivo."""
    dataset = gravar_dataset(str(tmp_path_factory.mktemp('dataset')), por_acorde=10)
    saida = str(tmp_path_factory.mktemp('modelo'))
    original = pipeline.synth_negative_ids
    mp = pytest.MonkeyPatch()
    # 20 negativos sintéticos de cada tipo em vez de 370, só pro teste rodar em segundos
    mp.setattr(pipeline, 'synth_negative_ids', lambda split: original(split, n=20))
    try:
        resultado = train_model.train_model(out_dir=saida, feature='chroma_stft8192', scaler='feature',
                                            arch='gap_bn', with_none=True, seed=42, dataset_path=dataset,
                                            verbose=0)
    finally:
        mp.undo()
    return resultado, saida


def test_treino_salva_os_cinco_arquivos(treinado):
    (modelo, encoder, scaler), saida = treinado
    assert modelo is not None and encoder is not None and scaler is not None
    for nome in ('model.h5', 'encoder.joblib', 'scaler.joblib', 'model_config.json', 'metrics.json'):
        assert os.path.exists(os.path.join(saida, nome)), nome


def test_config_salvo_tem_limiar_calibrado(treinado):
    _, saida = treinado
    with open(os.path.join(saida, 'model_config.json'), encoding='utf-8') as f:
        config = json.load(f)
    assert config['feature'] == 'chroma_stft8192' and config['arch'] == 'gap_bn'
    assert config['classes'] == ['Am', 'C', 'Em', 'G', '_none']
    assert config['confidence_threshold'] in pipeline.THRESHOLD_GRID
    assert config['seed'] == 42 and config['trained_by'] == 'train_model.py'


def test_metricas_salvas_fazem_sentido(treinado):
    _, saida = treinado
    with open(os.path.join(saida, 'metrics.json'), encoding='utf-8') as f:
        m = json.load(f)
    assert 0 <= m['accuracy_chords'] <= 1 and 0 <= m['none_recall'] <= 1
    assert m['n_test_chord_windows'] == 8 * 3  # 8 arquivos de teste x 3 janelas de 1,5 s em 3 s
    assert m['epochs_trained'] >= 1 and m['n_params'] > 0


def test_modelo_treinado_reconhece_acorde_sintetico(treinado):
    """Os acordes sintéticos são fáceis: depois do treino, um C novo tem que sair C."""
    _, saida = treinado
    modelo, encoder, scaler = train_model.load_trained_model(saida)
    r = train_model.recognize_chord(acorde('C', seed=99), modelo, encoder, scaler)
    assert r['status'] == 'ok' and r['chord'] == 'C'


def test_dataset_vazio_nao_treina(tmp_path, monkeypatch):
    vazio = {'X_train': np.array([]), 'X_test': np.array([])}
    monkeypatch.setattr(pipeline, 'load_dataset', lambda *a, **k: vazio)
    saida = tmp_path / 'saida'
    assert train_model.train_model(out_dir=str(saida), dataset_path=str(tmp_path)) == (None, None, None)
    assert not saida.exists()  # não salva modelo nenhum


def test_modelo_corrompido_devolve_none(tmp_path):
    """Arquivo de modelo estragado: a API sobe e responde 503, em vez de quebrar."""
    pasta = tmp_path / 'trained_model'
    shutil.copytree(ROOT_MODELO, pasta)
    (pasta / 'model.h5').write_bytes(b'isso nao e um modelo keras')
    assert train_model.load_trained_model(str(pasta)) == (None, None, None)


class _Modelo:
    def __init__(self, n_saidas, vencedor):
        self.probs = np.full(n_saidas, 0.01)
        self.probs[vencedor] = 0.99

    def predict(self, x, verbose=0):
        return self.probs[np.newaxis, :]


class _Identidade:
    def transform(self, x):
        return x


def _encoder():
    enc = LabelEncoder().fit(['C', 'G', '_none'])
    enc.model_config_ = pipeline.make_model_config('chroma', 'feature', 'gap_bn', enc.classes_, True, 0.5)
    return enc


def test_audio_curto_demais_devolve_too_short():
    curto = acorde('C', segundos=0.01)  # 0,01 s com volume: não é silêncio, mas não dá feature
    r = train_model.recognize_chord(curto, _Modelo(3, 0), _encoder(), _Identidade())
    assert r == {'status': 'too_short', 'chord': None, 'confidence': 0.0}
    assert train_model.predict_note(curto, _Modelo(3, 0), _encoder(), _Identidade()) == 'N/A - Áudio curto'


def test_predict_note_silencio():
    silencio = np.zeros(33075, dtype=np.float32)
    assert train_model.predict_note(silencio, _Modelo(3, 0), _encoder(), _Identidade()) == 'Silêncio'


def test_saida_fora_das_classes_vira_desconhecido():
    """Modelo com mais saídas que classes no encoder: não inventa um acorde."""
    r = train_model.recognize_chord(acorde('C'), _Modelo(5, 4), _encoder(), _Identidade())
    assert r['status'] == 'unknown' and r['chord'] is None
