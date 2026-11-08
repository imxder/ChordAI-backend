"""Parte 7b: o modelo TFLite com a feature embutida, chamado como o app vai chamar
(WAV PCM16 -> float / 32768 -> [1, 33075] -> probabilidades)."""
import json
import os

import numpy as np
import pytest
import soundfile as sf

import pipeline as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT = os.path.join(ROOT, 'export')
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(EXPORT, 'chord_model.tflite')),
                                reason='export/ ainda nao foi gerado (python export/build_tflite.py)')


def _load(name):
    return json.load(open(os.path.join(EXPORT, name), encoding='utf-8'))


def _run(model_file, x):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=os.path.join(EXPORT, model_file))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    assert list(inp['shape']) == [1, 33075] and inp['dtype'] == np.float32
    it.set_tensor(inp['index'], x)
    it.invoke()
    return it.get_tensor(out['index'])[0]


def _example():
    audio, sr = sf.read(os.path.join(EXPORT, 'example.wav'), dtype='int16')
    assert sr == pl.SAMPLE_RATE and audio.ndim == 1
    return (audio.astype(np.float32) / 32768.0)[np.newaxis]


@pytest.mark.parametrize('model_file, key, atol', [('chord_model.tflite', 'float32', 1e-5),
                                                    ('chord_model_int8.tflite', 'int8_dynamic', 1e-5)])
def test_example_matches_expected(model_file, key, atol):
    expected = _load('example_expected.json')[key]
    labels = _load('labels.json')
    probs = _run(model_file, _example())
    assert len(probs) == len(labels)
    np.testing.assert_allclose(probs, [expected['probabilities'][c] for c in labels], atol=atol)
    assert labels[int(probs.argmax())] == expected['top_class']


def test_model_info_is_consistent():
    info, labels, cfg = _load('model_info.json'), _load('labels.json'), _load('model_config.json')
    assert labels == cfg['classes'] and info['none_label'] in labels
    assert info['sample_rate'] == pl.SAMPLE_RATE and info['input_samples'] == 33075
    assert info['confidence_threshold'] == cfg['confidence_threshold'] and 0 < info['confidence_threshold'] < 1
    assert info['silence_rms_threshold'] == pl.SILENCE_THRESHOLD


def test_only_builtin_ops_with_native_rfft():
    import tensorflow as tf
    report = _load('export_report.json')
    for name in ('float32', 'int8'):
        ops = report[f'ops_{name}']
        assert 'RFFT2D' in ops
        assert not any(op.startswith('Flex') or op == 'CUSTOM' for op in ops)
    assert report['int8_touched_window_or_chroma'] is False
    # e o arquivo de fato carrega so com o resolver de ops nativas
    it = tf.lite.Interpreter(model_path=os.path.join(EXPORT, 'chord_model.tflite'),
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
