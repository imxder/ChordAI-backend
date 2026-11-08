"""Parte 7b: exporta o modelo C1 (chroma_stft n_fft 8192 + aug2) para TFLite com
a extracao de feature EMBUTIDA. O app manda audio bruto e recebe as probabilidades.

Grafo (todos os parametros vem de pipeline.chroma_stft_params / CHROMA_STFT_*):
  audio float32 [1, 33075] (1,5 s a 22050 Hz, mono, -1..1)
  -> padding de n_fft//2 com zeros nas duas pontas (= center=True, pad_mode='constant')
  -> tf.signal.stft (frame n_fft, hop, fft n_fft, janela Hann periodica do librosa como
     constante, sem padding extra) -> RFFT nativa do TFLite
  -> potencia |STFT|^2 -> banco de filtros chroma do pipeline (12 x 4097, constante)
  -> normalizacao por frame (norm=inf; frames com maximo < tiny(float32) ficam como estao)
  -> 65 frames -> StandardScaler por feature (12 medias / 12 desvios, constantes)
  -> CNN GAP+BN (pesos do .h5) -> softmax [1, n_classes] (inclui "_none")

Seed: entre 42, 43 e 44, a de maior acuracia nos acordes na VALIDACAO (nunca pelo teste);
empate -> menor val_loss. As metricas de validacao das tres vao para export_report.json.

Etapas: equivalencia com o librosa (200 janelas), conversao float32 e int8 (quantizacao
dinamica, so ops nativas), avaliacao no teste completo / degradado / negativos com o
interpretador TFLite e os arquivos para o app em export/.

Uso:  python export/build_tflite.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa: E402

import pipeline as pl  # noqa: E402
from experiments.data import DEGRADED_SEED, build, transform_test_degraded  # noqa: E402

SOURCE_GROUP = 'aug2_chroma_stft8192_scalerfeat_gapbn_none'
CANDIDATE_SEEDS = [42, 43, 44]
RESULTS_DIR = os.path.join(ROOT, 'experiments', 'results')
THRESHOLD_SUGGESTION = os.path.join(ROOT, 'experiments', 'results', 'threshold_aug2_chroma_stft8192',
                                    'suggestion.json')
EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_SAMPLES = int(round(pl.SAMPLE_RATE * pl.WINDOW_SECONDS))  # 33075
N_EQUIV_WINDOWS = 200
EQUIV_SEED = 0
MIN_CLASS_AGREEMENT = 0.995
MAX_INT8_DROP_PP = 0.5
MAX_FALSE_ACCEPT = 0.025  # item 0 da Parte 7b: acima disso, parar
MODEL_VERSION = 'c1-chroma_stft8192-aug2-gapbn-none-s{seed}'


# ---------------------------------------------------------------------------
# Etapa 0: escolha da seed na validacao
# ---------------------------------------------------------------------------
def choose_seed():
    """Acuracia nos acordes (argmax, todas as janelas de acorde da validacao) de cada seed;
    escolhe a maior (empate: menor best_val_loss). O teste nao participa."""
    train_files, _, _ = pl.load_split()
    _, val_files = pl.split_train_val(train_files)
    feature = json.load(open(os.path.join(RESULTS_DIR, f'{SOURCE_GROUP}_s{CANDIDATE_SEEDS[0]}', pl.CONFIG_FILENAME),
                             encoding='utf-8'))['feature']
    X, y, _, _ = build(val_files, feature, 'clean')
    sug = json.load(open(THRESHOLD_SUGGESTION, encoding='utf-8'))
    per_seed = {p['seed']: p for p in sug['limiar_por_seed']}
    rows = []
    for seed in CANDIDATE_SEEDS:
        run_dir = os.path.join(RESULTS_DIR, f'{SOURCE_GROUP}_s{seed}')
        model = tf.keras.models.load_model(os.path.join(run_dir, 'model.h5'), compile=False)
        scaler = joblib.load(os.path.join(run_dir, 'scaler.joblib'))
        classes = np.array(joblib.load(os.path.join(run_dir, 'encoder.joblib')).classes_)
        pred = classes[model.predict(pl.apply_scaler(scaler, X, 'feature'), verbose=0).argmax(1)]
        metrics = json.load(open(os.path.join(run_dir, 'metrics.json'), encoding='utf-8'))
        rows.append({'seed': seed, 'val_accuracy_chords': float(np.mean(pred == y)),
                     'best_val_loss': float(metrics['best_val_loss']),
                     'val_threshold': per_seed[seed]['limiar'], 'val_at_threshold': per_seed[seed]['validacao']})
    best = max(rows, key=lambda r: (r['val_accuracy_chords'], -r['best_val_loss']))
    for r in rows:
        print(f"[seed] {r['seed']}: validacao acc {r['val_accuracy_chords'] * 100:.2f}% | val_loss "
              f"{r['best_val_loss']:.4f} | limiar {r['val_threshold']:.2f}" + ('  <- escolhida' if r is best else ''))
    return best['seed'], {
        'chosen_seed': best['seed'],
        'criterion': 'maior acuracia nos acordes na VALIDACAO (argmax, todas as janelas de acorde dos 15% de '
                     'arquivos de validacao); empate -> menor best_val_loss. O teste nao participa da escolha.',
        'n_val_chord_windows': int(len(X)),
        'candidates': rows,
    }


# ---------------------------------------------------------------------------
# Modelo de origem
# ---------------------------------------------------------------------------
def load_source(seed):
    source_run = f'{SOURCE_GROUP}_s{seed}'
    source_dir = os.path.join(RESULTS_DIR, source_run)
    cfg = json.load(open(os.path.join(source_dir, pl.CONFIG_FILENAME), encoding='utf-8'))
    assert cfg['seed'] == seed
    sug = json.load(open(THRESHOLD_SUGGESTION, encoding='utf-8'))
    per_seed = next(p for p in sug['limiar_por_seed'] if p['seed'] == seed)
    cfg = dict(cfg, confidence_threshold=per_seed['limiar'],
               threshold_source=f'calibrado na validacao para este modelo (seed {seed}), criterio da Parte 5 '
                                f'(experiments/results/threshold_aug2_chroma_stft8192/suggestion.json)',
               source_experiment=f'experiments/results/{source_run}', model_version=MODEL_VERSION.format(seed=seed))
    pl.validate_config(cfg)
    model = tf.keras.models.load_model(os.path.join(source_dir, 'model.h5'), compile=False)
    encoder = joblib.load(os.path.join(source_dir, 'encoder.joblib'))
    scaler = joblib.load(os.path.join(source_dir, 'scaler.joblib'))
    assert cfg['scaler'] == 'feature' and cfg['feature'] in pl.CHROMA_STFT_N_FFT_BY_FEATURE
    assert list(encoder.classes_) == cfg['classes']
    return model, encoder, scaler, cfg, per_seed


# ---------------------------------------------------------------------------
# Grafo TF de ponta a ponta
# ---------------------------------------------------------------------------
class EndToEnd(tf.Module):
    """audio [b, 33075] -> chroma [b, 65, 12] -> scaler -> CNN -> softmax."""

    def __init__(self, model, scaler, feature):
        super().__init__()
        prm = pl.chroma_stft_params(feature)
        # o grafo so implementa o que o pipeline usa; qualquer mudanca de parametro tem que vir aqui
        assert prm['center'] and prm['pad_mode'] == 'constant', prm
        assert prm['power'] == 2.0 and prm['norm'] == np.inf, prm
        self.n_fft, self.hop, self.n_frames = prm['n_fft'], prm['hop_length'], prm['n_frames']
        self.pad = self.n_fft // 2
        n_frames_stft = 1 + (WINDOW_SAMPLES + 2 * self.pad - self.n_fft) // self.hop
        assert n_frames_stft == 1 + WINDOW_SAMPLES // self.hop == self.n_frames == 65, n_frames_stft
        self.window = tf.constant(pl.chroma_stft_window(feature))                       # [n_fft]
        self.chroma_fb = tf.constant(pl.chroma_stft_filterbank(prm['sample_rate'], self.n_fft).T)  # [n_fft/2+1, 12]
        self.norm_threshold = tf.constant(prm['norm_threshold'], tf.float32)
        self.mean = tf.constant(scaler.mean_.astype(np.float32))                         # [12]
        self.scale = tf.constant(scaler.scale_.astype(np.float32))                       # [12]
        self.model = model

    def chroma(self, audio):
        """[b, 33075] -> [b, 65, 12] (frames x chroma), igual a extract_chroma_stft antes do scaler."""
        x = tf.pad(audio, [[0, 0], [self.pad, self.pad]], mode='CONSTANT')  # center=True, zeros
        stft = tf.signal.stft(x, frame_length=self.n_fft, frame_step=self.hop, fft_length=self.n_fft,
                              window_fn=lambda length, dtype: self.window, pad_end=False)
        power = tf.math.square(tf.math.real(stft)) + tf.math.square(tf.math.imag(stft))  # |STFT|^2
        # A quantizacao dinamica (int8) quantiza toda constante grande usada como peso de
        # FULLY_CONNECTED/BATCH_MATMUL, e nao tem opcao para excluir um tensor. Multiplicar a
        # matriz por um 1.0 que depende da entrada (sign(|x| + 1) = 1, exato) faz dela uma
        # ativacao: o conversor nao a trata como peso e ela fica float32 nas duas versoes.
        one = tf.sign(tf.abs(audio[0, 0]) + 1.0)
        raw = tf.matmul(power, self.chroma_fb * one)                                      # [b, frames, 12]
        length = tf.reduce_max(tf.abs(raw), axis=-1, keepdims=True)                       # norm=inf
        length = tf.where(length < self.norm_threshold, tf.ones_like(length), length)     # fill=None
        out = raw / length
        return out[:, :self.n_frames, :]  # ja sao 65 frames (assert no __init__)

    def probs(self, audio):
        feats = (self.chroma(audio) - self.mean) / self.scale
        return self.model(feats, training=False)

    @tf.function(input_signature=[tf.TensorSpec([1, WINDOW_SAMPLES], tf.float32, name='audio')])
    def serve(self, audio):
        return {'probabilities': self.probs(audio)}

    @tf.function(input_signature=[tf.TensorSpec([None, WINDOW_SAMPLES], tf.float32)])
    def batch_chroma(self, audio):
        return self.chroma(audio)

    @tf.function(input_signature=[tf.TensorSpec([None, WINDOW_SAMPLES], tf.float32)])
    def batch_probs(self, audio):
        return self.probs(audio)


# ---------------------------------------------------------------------------
# Dados de teste (audio bruto, como o app recebe)
# ---------------------------------------------------------------------------
def load_test_audio():
    _, test_files, _ = pl.load_split()
    clean, degraded, labels, rms_clean = [], [], [], []
    for fidx, (path, label) in enumerate(test_files):
        for widx, seg in enumerate(pl.make_windowed_segments(path)):
            assert len(seg) == WINDOW_SAMPLES
            clean.append(seg)
            degraded.append(transform_test_degraded(seg, DEGRADED_SEED, fidx, widx, 0))
            labels.append(label)
            rms_clean.append(pl.rms(seg))
    negs, kinds = [], []
    for kind, source in pl.negative_specs('test', test_files):
        negs.append(pl.make_negative_audio(kind, source))
        kinds.append(kind)
    as_f32 = lambda xs: np.stack([np.asarray(x, dtype=np.float32) for x in xs])
    return {'clean': as_f32(clean), 'degraded': as_f32(degraded), 'labels': np.array(labels),
            'rms_clean': np.array(rms_clean), 'rms_degraded': np.array([pl.rms(x) for x in degraded]),
            'negatives': as_f32(negs), 'neg_kinds': np.array(kinds),
            'rms_negatives': np.array([pl.rms(x) for x in negs])}


def librosa_features(audio_batch, feature):
    return np.stack([pl.EXTRACTORS[feature](a) for a in audio_batch])  # [n, 12, 65]


def keras_probs(model, scaler, audio_batch, feature):
    return model.predict(pl.apply_scaler(scaler, librosa_features(audio_batch, feature), 'feature'),
                         verbose=0, batch_size=256)


# ---------------------------------------------------------------------------
# Etapa 2: equivalencia grafo TF x librosa
# ---------------------------------------------------------------------------
def equivalence(e2e, model, scaler, feature, data):
    idx = np.sort(np.random.default_rng(EQUIV_SEED).choice(len(data['clean']), N_EQUIV_WINDOWS, replace=False))
    audio = data['clean'][idx]
    ref = np.swapaxes(librosa_features(audio, feature), 1, 2)    # [n, 65, 12]
    got = e2e.batch_chroma(tf.constant(audio)).numpy()
    err = np.abs(got - ref)
    p_ref = keras_probs(model, scaler, audio, feature)
    p_tf = e2e.batch_probs(tf.constant(audio)).numpy()
    agree = float(np.mean(p_ref.argmax(1) == p_tf.argmax(1)))
    res = {'n_windows': int(len(idx)), 'feature_abs_err_max': float(err.max()),
           'feature_abs_err_mean': float(err.mean()), 'prob_abs_err_max': float(np.abs(p_ref - p_tf).max()),
           'class_agreement': agree, 'ok': agree >= MIN_CLASS_AGREEMENT}
    print(f"[equivalencia] {res['n_windows']} janelas | feature: erro abs max {res['feature_abs_err_max']:.2e}, "
          f"medio {res['feature_abs_err_mean']:.2e} | prob: erro abs max {res['prob_abs_err_max']:.2e} | "
          f"mesma classe: {agree * 100:.2f}%")
    return res


# ---------------------------------------------------------------------------
# Etapa 3: conversao
# ---------------------------------------------------------------------------
def convert(e2e, quantize):
    # congela os pesos da CNN (o Keras 3 nao e rastreado pelo tf.Module; sem isto o .tflite
    # fica com VAR_HANDLE/READ_VARIABLE e pesos nao inicializados)
    fn = convert_variables_to_constants_v2(e2e.serve.get_concrete_function())
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn])
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # sem SELECT_TF_OPS
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]  # quantizacao dinamica: pesos int8
    return conv.convert()


def op_names(tflite_bytes):
    it = tf.lite.Interpreter(model_content=tflite_bytes,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    return sorted({op['op_name'] for op in it._get_ops_details()})


def quantized_constants(tflite_bytes):
    """Tensores int8 do modelo (os pesos que a quantizacao dinamica tocou), com o shape."""
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    return sorted({(d['name'], tuple(int(s) for s in d['shape'])) for d in it.get_tensor_details()
                   if d['dtype'] == np.int8})


# ---------------------------------------------------------------------------
# Etapa 4: avaliacao
# ---------------------------------------------------------------------------
def tflite_probs(tflite_bytes, audio_batch):
    it = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=1)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    res = np.empty((len(audio_batch), out['shape'][-1]), np.float32)
    for i, a in enumerate(audio_batch):
        it.set_tensor(inp['index'], a[np.newaxis])
        it.invoke()
        res[i] = it.get_tensor(out['index'])[0]
    return res


def tflite_latency_ms(tflite_bytes, audio, n=50):
    it = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=1)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp['index'], audio[np.newaxis])
    it.invoke()
    t0 = time.perf_counter()
    for _ in range(n):
        it.set_tensor(inp['index'], audio[np.newaxis])
        it.invoke()
    return (time.perf_counter() - t0) / n * 1000


def metrics_from_probs(p_clean, p_deg, p_neg, data, classes, threshold):
    """Mesmo protocolo do threshold.py: argmax sem limiar e, com o filtro de silencio + limiar,
    cobertura / acuracia nos aceitos / falso aceite."""
    classes = np.asarray(classes)
    y = data['labels']

    def accepted(p):
        return (classes[p.argmax(1)] != pl.NONE_LABEL) & (p.max(1) >= threshold)

    def gated(p, rms):
        g = rms >= pl.SILENCE_THRESHOLD
        acc = accepted(p[g])
        correct = acc & (classes[p[g].argmax(1)] == y[g])
        return float(acc.mean()), float(correct.sum() / max(1, acc.sum()))

    cov, acc_acc = gated(p_clean, data['rms_clean'])
    cov_d, acc_acc_d = gated(p_deg, data['rms_degraded'])
    neg_gate = data['rms_negatives'] >= pl.SILENCE_THRESHOLD
    return {
        'accuracy_chords': float(np.mean(classes[p_clean.argmax(1)] == y)),
        'accuracy_degraded': float(np.mean(classes[p_deg.argmax(1)] == y)),
        'none_recall': float(np.mean(classes[p_neg.argmax(1)] == pl.NONE_LABEL)),
        'coverage': cov, 'accuracy_accepted': acc_acc,
        'coverage_degraded': cov_d, 'accuracy_accepted_degraded': acc_acc_d,
        'false_accept': float(accepted(p_neg[neg_gate]).sum() / len(p_neg)),
    }


# ---------------------------------------------------------------------------
# Etapa 5: arquivos para o app
# ---------------------------------------------------------------------------
def pcm16_roundtrip(audio):
    """O que o app vai ter depois de ler um WAV PCM16 e converter para float (-1..1)."""
    return (np.clip(np.round(audio * 32767), -32768, 32767).astype(np.int16).astype(np.float32) / 32768.0)


def save_app_files(f32, i8, cfg, data, per_seed, probs_f32_clean):
    with open(os.path.join(EXPORT_DIR, 'chord_model.tflite'), 'wb') as f:
        f.write(f32)
    with open(os.path.join(EXPORT_DIR, 'chord_model_int8.tflite'), 'wb') as f:
        f.write(i8)
    with open(os.path.join(EXPORT_DIR, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg['classes'], f, indent=2, ensure_ascii=False)
    with open(os.path.join(EXPORT_DIR, pl.CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    prm = pl.chroma_stft_params(cfg['feature'])
    info = {
        'model_version': cfg['model_version'],
        'seed': cfg['seed'],
        'source_experiment': cfg['source_experiment'],
        'sample_rate': pl.SAMPLE_RATE,
        'channels': 1,
        'input_samples': WINDOW_SAMPLES,
        'window_seconds': pl.WINDOW_SECONDS,
        'input': {'name': 'audio', 'dtype': 'float32', 'shape': [1, WINDOW_SAMPLES],
                  'range': [-1.0, 1.0], 'from_pcm16': 'amostra_int16 / 32768.0'},
        'output': {'name': 'probabilities', 'dtype': 'float32', 'shape': [1, len(cfg['classes'])],
                   'labels_file': 'labels.json'},
        'none_label': cfg['none_label'],
        'confidence_threshold': cfg['confidence_threshold'],
        'silence_rms_threshold': pl.SILENCE_THRESHOLD,
        'files': {'float32': 'chord_model.tflite', 'int8_dynamic': 'chord_model_int8.tflite'},
        'feature_inside_model': {'type': cfg['feature'], **{k: (v if v != np.inf else 'inf')
                                                            for k, v in prm.items()}},
        'validation_metrics_at_threshold': per_seed['validacao'],
    }
    with open(os.path.join(EXPORT_DIR, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # example.wav: uma janela de "C" acertada com confianca intermediaria (0,6-0,95), para que as
    # probabilidades de todas as classes sirvam de conferencia (com confianca 1,0 uma feature errada
    # ainda poderia dar o mesmo resultado)
    classes = np.asarray(cfg['classes'])
    conf = probs_f32_clean.max(1)
    ok = ((data['labels'] == 'C') & (classes[probs_f32_clean.argmax(1)] == 'C') & (conf >= 0.6) & (conf <= 0.95)
          & (data['rms_clean'] > 10 * pl.SILENCE_THRESHOLD))
    i = int(np.flatnonzero(ok)[0])
    sf.write(os.path.join(EXPORT_DIR, 'example.wav'), data['clean'][i], pl.SAMPLE_RATE, subtype='PCM_16')
    audio, sr = sf.read(os.path.join(EXPORT_DIR, 'example.wav'), dtype='int16')
    assert sr == pl.SAMPLE_RATE and len(audio) == WINDOW_SAMPLES
    x = (audio.astype(np.float32) / 32768.0)[np.newaxis]
    expected = {}
    for name, blob in (('float32', f32), ('int8_dynamic', i8)):
        p = tflite_probs(blob, x)[0]
        expected[name] = {'probabilities': {c: float(v) for c, v in zip(cfg['classes'], p)},
                          'top_class': cfg['classes'][int(p.argmax())], 'top_confidence': float(p.max())}
    with open(os.path.join(EXPORT_DIR, 'example_expected.json'), 'w', encoding='utf-8') as f:
        json.dump({'wav': 'example.wav', 'true_label': 'C', 'rms': pl.rms(x[0]),
                   'note': 'probabilidades obtidas lendo o WAV PCM16 e dividindo por 32768, como o app deve fazer',
                   **expected}, f, indent=2, ensure_ascii=False)
    return expected


def main():
    out = {}
    seed, out['seed_selection'] = choose_seed()
    model, encoder, scaler, cfg, per_seed = load_source(seed)
    source_run = f'{SOURCE_GROUP}_s{seed}'
    feature, classes, threshold = cfg['feature'], cfg['classes'], cfg['confidence_threshold']
    print(f'modelo: {source_run} | feature {feature} | classes {classes} | limiar {threshold}')
    fa = per_seed['teste']['false_accept']
    print(f"[item 0] seed {seed}: falso aceite no teste com o limiar dela = {fa * 100:.2f}%")
    if fa > MAX_FALSE_ACCEPT:
        raise SystemExit(f'falso aceite da seed {seed} = {fa * 100:.2f}% > {MAX_FALSE_ACCEPT * 100:.1f}%: parar e avisar')
    e2e = EndToEnd(model, scaler, feature)
    print('carregando o audio do teste (limpo, degradado e negativos)...', flush=True)
    data = load_test_audio()
    print(f"  {len(data['clean'])} janelas de acorde, {len(data['negatives'])} negativas")

    out['equivalence'] = equivalence(e2e, model, scaler, feature, data)
    if not out['equivalence']['ok']:
        raise SystemExit('equivalencia abaixo de 99,5%: investigar antes de converter')

    blobs = {}
    for name, quant in (('float32', False), ('int8', True)):
        # (as ops sao listadas sem o delegate XNNPACK, que o interpretador aplica por padrao)
        try:
            blobs[name] = convert(e2e, quant)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f'conversao {name} falhou so com ops nativas (sem SELECT_TF_OPS): {e}')
        out[f'ops_{name}'] = op_names(blobs[name])
        out[f'size_kb_{name}'] = round(len(blobs[name]) / 1024, 1)
        print(f'[{name}] {out[f"size_kb_{name}"]} KB | ops: {", ".join(out[f"ops_{name}"])}')
    out['int8_quantized_tensors'] = [{'name': n, 'shape': list(s)} for n, s in quantized_constants(blobs['int8'])]
    n_bins = e2e.n_fft // 2 + 1
    touched = [t for t in out['int8_quantized_tensors']
               if sorted(t['shape']) in ([12, n_bins], [1, 12, n_bins], [e2e.n_fft], [1, e2e.n_fft])]
    if touched:
        raise SystemExit(f'a quantizacao int8 tocou a janela ou a matriz de chroma: {touched}')
    out['int8_touched_window_or_chroma'] = bool(touched)
    print(f"[int8] tensores int8: {[tuple(t['shape']) for t in out['int8_quantized_tensors']]} | "
          f"janela/matriz chroma quantizadas: {'SIM' if touched else 'nao'}")

    print('avaliando Keras (librosa) e TFLite no teste completo...', flush=True)
    probs = {'keras': [keras_probs(model, scaler, data[k], feature) for k in ('clean', 'degraded', 'negatives')]}
    for name, blob in blobs.items():
        probs[name] = [tflite_probs(blob, data[k]) for k in ('clean', 'degraded', 'negatives')]
    out['metrics'] = {name: metrics_from_probs(*p, data, classes, threshold) for name, p in probs.items()}
    out['latency_ms_tflite_1thread'] = {name: round(tflite_latency_ms(blob, data['clean'][0]), 2)
                                        for name, blob in blobs.items()}
    drop = (out['metrics']['float32']['accuracy_chords'] - out['metrics']['int8']['accuracy_chords']) * 100
    drop_deg = (out['metrics']['float32']['accuracy_degraded'] - out['metrics']['int8']['accuracy_degraded']) * 100
    out['int8_drop_pp'] = {'accuracy_chords': round(drop, 3), 'accuracy_degraded': round(drop_deg, 3),
                           'exceeds_limit': max(drop, drop_deg) > MAX_INT8_DROP_PP}
    out['agreement_with_keras'] = {name: float(np.mean(probs[name][0].argmax(1) == probs['keras'][0].argmax(1)))
                                   for name in blobs}
    for name, m in out['metrics'].items():
        print(f"  {name:8s} acc {m['accuracy_chords'] * 100:.2f} | degradado {m['accuracy_degraded'] * 100:.2f} | "
              f"falso aceite {m['false_accept'] * 100:.2f} | aceitos {m['accuracy_accepted'] * 100:.2f} "
              f"(cobertura {m['coverage'] * 100:.2f})")

    out['example'] = save_app_files(blobs['float32'], blobs['int8'], cfg, data, per_seed, probs['float32'][0])
    out['source'] = {'run': source_run, 'seed': seed, 'threshold': threshold, 'versions': {
        'tensorflow': tf.__version__, 'librosa': pl.librosa.__version__, 'numpy': np.__version__}}
    with open(os.path.join(EXPORT_DIR, 'export_report.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'arquivos salvos em {EXPORT_DIR}')


if __name__ == '__main__':
    main()
