"""model_config.json (Parte 6): a inferencia usa a feature/scaler/limiar salvos
junto do modelo; pastas sem o arquivo seguem o pipeline antigo."""
import json
import os

import numpy as np
import pytest

import pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGACY_DIR = os.path.join(ROOT, 'candidates', 'candidate_91_81')
NEW_DIR = os.path.join(ROOT, 'candidates', 'candidate_chroma_gapbn_none')


def test_folder_without_config_is_legacy_pipeline(tmp_path):
    cfg = pipeline.load_model_config(str(tmp_path))
    assert cfg['legacy'] is True
    assert (cfg['feature'], cfg['scaler'], cfg['arch']) == ('mfcc', 'frame', 'flatten')
    assert cfg['with_none'] is False and cfg['none_label'] is None
    assert cfg['confidence_threshold'] is None
    assert cfg['files'] == pipeline.LEGACY_FILES


def test_config_roundtrip_and_validation(tmp_path):
    cfg = pipeline.make_model_config('chroma', 'per_feature', 'gap_bn', ['C', 'G', '_none'], True, 0.45)
    assert cfg['scaler'] == 'feature' and cfg['arch'] == 'gap_bn' and cfg['none_label'] == '_none'
    (tmp_path / pipeline.CONFIG_FILENAME).write_text(json.dumps(cfg), encoding='utf-8')
    assert pipeline.load_model_config(str(tmp_path)) == cfg

    bad = dict(cfg, sample_rate=16000)
    (tmp_path / pipeline.CONFIG_FILENAME).write_text(json.dumps(bad), encoding='utf-8')
    with pytest.raises(ValueError):
        pipeline.load_model_config(str(tmp_path))


def test_features_for_model_matches_original_mfcc_path():
    """features_for_model com o config antigo == o processamento do _predict_raw original."""
    rng = np.random.default_rng(1)
    audio = (0.1 * rng.standard_normal(int(pipeline.SAMPLE_RATE * 1.5))).astype(np.float32)
    feats = pipeline.extract_features(audio, is_file=False)
    scaler = pipeline.StandardScaler().fit(rng.standard_normal((100, feats.shape[-1])))
    original = np.expand_dims(np.swapaxes(scaler.transform(feats.reshape(-1, feats.shape[-1])).reshape(feats.shape), 0, 1), 0)
    new = pipeline.features_for_model(audio, scaler, pipeline.legacy_model_config())
    np.testing.assert_allclose(new, original)


@pytest.mark.skipif(not os.path.exists(LEGACY_DIR), reason='candidate_91_81 ausente')
def test_load_legacy_candidate():
    model, encoder, scaler, cfg = pipeline.load_model_bundle(LEGACY_DIR)
    assert cfg.get('legacy') and cfg['classes'] == [str(c) for c in encoder.classes_]
    assert model.input_shape[1:] == (pipeline.MAX_PAD_LEN, pipeline.N_MFCC)


@pytest.mark.skipif(not os.path.exists(NEW_DIR), reason='candidato novo ausente')
def test_load_new_candidate():
    model, encoder, scaler, cfg = pipeline.load_model_bundle(NEW_DIR)
    assert (cfg['feature'], cfg['scaler'], cfg['arch']) == ('chroma', 'feature', 'gap_bn')
    assert cfg['with_none'] and cfg['none_label'] == '_none' and '_none' in encoder.classes_
    assert 0 < cfg['confidence_threshold'] < 1
    assert model.input_shape[1:] == (pipeline.MAX_PAD_LEN, 12)
