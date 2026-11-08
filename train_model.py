"""Treino e inferencia do reconhecedor de acordes do ChordAI.

A configuracao padrao e a final dos experimentos (ver experiments/results/comparativo.md):
chroma (chroma_cqt, 12 bins x 65 frames) + StandardScaler por feature + CNN 1D
com BatchNorm e GlobalAveragePooling + classe "_none" + limiar de confianca
calibrado na validacao.

Para reproduzir o baseline do TCC (~88% com validacao separada):
    FEATURE_TYPE = "mfcc"; SCALER_TYPE = "frame"; ARCH = "flatten"; WITH_NONE = False

Todo modelo salvo tem um model_config.json; a inferencia (_predict_raw) le esse
arquivo para extrair a feature e aplicar o scaler exatamente como no treino.
Pastas sem model_config.json (ex.: candidates/candidate_91_81) sao tratadas
como o pipeline antigo: 40 MFCC + scaler por frame, sem _none e sem limiar.
"""
import os
import time
import warnings

import numpy as np

import pipeline
from pipeline import (  # noqa: F401  (reexportados para app.py e scripts antigos)
    CROP_HOP_SECONDS,
    MAX_CROPS_PER_FILE,
    MAX_PAD_LEN,
    MIN_VALID_DURATION,
    N_MFCC,
    NONE_LABEL,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    WINDOW_SECONDS,
    _compute_max_pad_len,
    build_windowed_dataset,
    extract_features,
    list_dataset_files,
    make_windowed_segments,
    split_dataset_files,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuracao do treino (padrao = configuracao final dos experimentos)
# ---------------------------------------------------------------------------
FEATURE_TYPE = "chroma"   # "mfcc" | "chroma" | "cqt" | "mfcc+chroma"
SCALER_TYPE = "feature"   # "feature" (por coeficiente/bin) | "frame" (o original)
ARCH = "gap_bn"           # "gap_bn" | "flatten" (o original) | "cnn2d"
WITH_NONE = True          # classe "_none" (nenhum acorde) + limiar calibrado na validacao
SEED = 42

DATASET_PATH = 'dataset'
MODEL_DIR = 'trained_model'

# Caminhos do formato ANTIGO (sem model_config.json), mantidos para scripts que
# ainda os importam (train_ensemble_experiment.py, train_experiment_class_weight.py).
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, pipeline.LEGACY_FILES['model'])
ENCODER_SAVE_PATH = os.path.join(MODEL_DIR, pipeline.LEGACY_FILES['encoder'])
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, pipeline.LEGACY_FILES['scaler'])


def train_model(out_dir=MODEL_DIR, feature=None, scaler=None, arch=None, with_none=None, seed=None,
                dataset_path=DATASET_PATH, verbose=1):
    """Treina e salva em out_dir (modelo, encoder, scaler, model_config.json, metrics.json).
    Os argumentos None usam as constantes do topo. Retorna (model, encoder, scaler)."""
    feature = FEATURE_TYPE if feature is None else feature
    scaler_type = SCALER_TYPE if scaler is None else scaler
    arch = ARCH if arch is None else arch
    with_none = WITH_NONE if with_none is None else with_none
    seed = SEED if seed is None else seed

    print(f"Configuracao: feature={feature} scaler={scaler_type} arch={arch} with_none={with_none} seed={seed}")
    pipeline.set_seeds(seed)
    print("Montando treino / validacao (15% dos arquivos de treino) / teste (split por arquivo, 80/20)...")
    d = pipeline.load_dataset(feature, with_none=with_none, dataset_path=os.path.abspath(dataset_path))
    if len(d['X_train']) == 0 or len(d['X_test']) == 0:
        print("Nenhum dado encontrado no dataset. Verifique se a pasta 'dataset' contém subpastas com arquivos .wav de acordes.")
        return None, None, None

    encoder, y_train, y_val, y_test = pipeline.encode_labels(d['labels'], d['y_train'], d['y_val'], d['y_test'])
    X_train, scaler_obj = pipeline.fit_scaler(d['X_train'], scaler_type)
    X_val = pipeline.apply_scaler(scaler_obj, d['X_val'], scaler_type)
    X_test = pipeline.apply_scaler(scaler_obj, d['X_test'], scaler_type)

    model = pipeline.build_model(arch, X_train.shape[1:], len(d['labels']))
    print(f"Shape de entrada: {X_train.shape[1:]}  classes: {d['labels']}  parametros: {model.count_params()}")
    print("Iniciando treinamento (EarlyStopping / ReduceLROnPlateau na VALIDACAO)...")
    t0 = time.time()
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val),
                        callbacks=pipeline.training_callbacks(), verbose=verbose)
    train_seconds = time.time() - t0

    classes = np.array(encoder.classes_)
    threshold = None
    if with_none:
        # calibra o limiar na VALIDACAO, como na Parte 5 (janelas de silencio fora)
        val_rms = d['rms_val']
        n_val_chords = len(val_rms)
        gate = val_rms >= SILENCE_THRESHOLD
        probs_val = model.predict(X_val, verbose=0)
        rows = pipeline.threshold_curve(probs_val[:n_val_chords][gate], d['y_val'][:n_val_chords][gate],
                                        probs_val[n_val_chords:], classes)
        threshold = pipeline.choose_threshold(rows)
        print(f"Limiar de confianca calibrado na validacao: {threshold}")

    pred = classes[np.argmax(model.predict(X_test, verbose=0), axis=1)]
    y_test_raw = np.asarray(d['y_test'])
    is_chord = y_test_raw != NONE_LABEL
    metrics = {
        'accuracy': float(np.mean(pred == y_test_raw)),
        'accuracy_chords': float(np.mean(pred[is_chord] == y_test_raw[is_chord])),
        'n_test_chord_windows': int(is_chord.sum()),
        'n_params': int(model.count_params()),
        'epochs_trained': len(history.history['loss']),
        'train_seconds': round(train_seconds, 1),
        'classes': [str(c) for c in classes],
    }
    if with_none:
        metrics['none_recall'] = float(np.mean(pred[~is_chord] == NONE_LABEL))
    print(f"Acuracia no teste (janelas de acorde de {WINDOW_SECONDS}s): {metrics['accuracy_chords']:.4f}")

    config = pipeline.make_model_config(feature, scaler_type, arch, classes, with_none, threshold,
                                        extra={'seed': seed, 'trained_by': 'train_model.py'})
    pipeline.save_model_bundle(model, encoder, scaler_obj, config, out_dir, metrics=metrics)
    print(f"Modelo, encoder, scaler e model_config.json salvos em: {out_dir}")
    return model, encoder, scaler_obj


def load_trained_model(model_dir=MODEL_DIR):
    """Carrega (model, encoder, scaler) de model_dir. O model_config.json fica em
    encoder.model_config_ e e usado por _predict_raw() / recognize_chord() /
    predict_note(). Sem modelo salvo, lanca FileNotFoundError: o treino so acontece
    explicitamente, via `python train_model.py`."""
    if not pipeline.model_bundle_exists(model_dir):
        raise FileNotFoundError(f"Modelo não encontrado em {model_dir}. Rode python train_model.py")
    print(f"Carregando modelo de acordes de: {model_dir}")
    try:
        model, encoder, scaler, config = pipeline.load_model_bundle(model_dir)
        encoder.model_config_ = config
        desc = 'pipeline antigo (sem model_config.json)' if config.get('legacy') else (
            f"feature={config['feature']} scaler={config['scaler']} arch={config['arch']} "
            f"_none={config['with_none']} limiar={config['confidence_threshold']}")
        print(f"Modelo de acordes carregado: {desc}")
        return model, encoder, scaler
    except Exception as e:
        print(f"Erro ao carregar o modelo de acordes: {e}")
        print("Pode ser necessário retreinar o modelo ou verificar a integridade dos arquivos.")
        return None, None, None


def get_model_config(encoder, config=None):
    """Configuracao do modelo: a explicita, a anexada pelo load_trained_model()
    ou, na falta das duas, a do pipeline antigo."""
    if config is not None:
        return config
    cfg = getattr(encoder, 'model_config_', None)
    return cfg if cfg is not None else pipeline.legacy_model_config()


def _is_unknown(chord, confidence, config, confidence_threshold=None):
    """True quando nao ha acorde confiavel: classe invalida, a classe "_none" do
    modelo ou confianca abaixo do limiar do model_config.json (Parte 5)."""
    if chord is None:
        return True
    none_label = config.get('none_label')
    if none_label is not None and chord == none_label:
        return True
    threshold = config.get('confidence_threshold') if confidence_threshold is None else confidence_threshold
    return threshold is not None and confidence < threshold


def _predict_raw(audio_segment_np, model, encoder, scaler, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
                 max_pad_len=MAX_PAD_LEN, config=None):
    """Nucleo compartilhado de inferencia. Retorna (chord_or_None, confidence, status),
    onde status in {'silence', 'too_short', 'ok'}. A feature e o scaler seguem o
    model_config.json do modelo (n_mfcc/max_pad_len ficam por compatibilidade)."""
    config = get_model_config(encoder, config)
    if audio_segment_np.size > 0:
        if np.sqrt(np.mean(audio_segment_np ** 2)) < SILENCE_THRESHOLD:
            return None, 0.0, 'silence'

    features_final = pipeline.features_for_model(audio_segment_np, scaler, config, sample_rate=sample_rate)
    if features_final is None:
        return None, 0.0, 'too_short'

    prediction = model.predict(features_final, verbose=0)[0]
    predicted_class_idx = int(np.argmax(prediction))
    confidence = float(prediction[predicted_class_idx])

    if predicted_class_idx < len(encoder.classes_):
        predicted_chord = encoder.inverse_transform([predicted_class_idx])[0]
    else:
        predicted_chord = None

    return predicted_chord, confidence, 'ok'


def predict_note(audio_segment_np, model, encoder, scaler, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
                 max_pad_len=MAX_PAD_LEN, config=None, confidence_threshold=None):
    """Usado pela interface web (app.py). Devolve "Acorde Desconhecido" quando o
    modelo preve "_none" ou a confianca fica abaixo do limiar do model_config.json."""
    config = get_model_config(encoder, config)
    chord, confidence, status = _predict_raw(audio_segment_np, model, encoder, scaler, sample_rate, n_mfcc,
                                             max_pad_len, config=config)
    if status == 'silence':
        return "Silêncio"
    if status == 'too_short':
        return "N/A - Áudio curto"
    if _is_unknown(chord, confidence, config, confidence_threshold):
        return "Acorde Desconhecido"
    return chord


def recognize_chord(audio_segment_np, model, encoder, scaler, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
                    max_pad_len=MAX_PAD_LEN, config=None, confidence_threshold=None):
    """Usado pela API HTTP: retorna um dict serializavel em JSON para o cliente Flutter.

    status 'unknown' (chord=None) quando o modelo preve a classe "_none" ou
    quando a confianca fica abaixo do limiar do model_config.json."""
    config = get_model_config(encoder, config)
    chord, confidence, status = _predict_raw(audio_segment_np, model, encoder, scaler, sample_rate, n_mfcc,
                                             max_pad_len, config=config)
    if status in ('silence', 'too_short'):
        return {'status': status, 'chord': None, 'confidence': 0.0}
    if _is_unknown(chord, confidence, config, confidence_threshold):
        return {'status': 'unknown', 'chord': None, 'confidence': round(confidence, 4)}
    return {'status': 'ok', 'chord': str(chord), 'confidence': round(confidence, 4)}


if __name__ == '__main__':
    train_model()
