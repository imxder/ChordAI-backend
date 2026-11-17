"""POST /predict_chord com o modelo do trained_model/, que está no Git (roda no CI)."""
import io
import json
import os

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(ROOT, 'export', 'example.wav')


def _post(client, wav_bytes):
    return client.post('/predict_chord', data={'audio': (io.BytesIO(wav_bytes), 'janela.wav')},
                       content_type='multipart/form-data')


def test_acorde_devolve_ok_com_o_acorde(client):
    # example.wav: 1,5 s de um C (o rótulo está no example_expected.json)
    with open(os.path.join(ROOT, 'export', 'example_expected.json'), encoding='utf-8') as f:
        esperado = json.load(f)['true_label']
    with open(EXAMPLE, 'rb') as f:
        response = _post(client, f.read())
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'
    assert data['chord'] == esperado
    assert 0 < data['confidence'] <= 1


def test_silencio_devolve_silence(client):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(33075, dtype=np.float32), 22050, format='WAV', subtype='PCM_16')
    response = _post(client, buf.getvalue())
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'silence'
    assert data['chord'] is None


def test_sem_o_campo_audio_devolve_400(client):
    response = client.post('/predict_chord')
    assert response.status_code == 400
    assert response.get_json()['status'] == 'error'
