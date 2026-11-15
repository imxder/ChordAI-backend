"""Compara o modelo atual com modelos treinados nas duas fontes (meu dataset e o
Guitar Chords V3), todos avaliados nos mesmos testes.

Modelos:
  0    trained_model/ de hoje, so avaliado (nunca viu o Test do v3 nem a s1/g1)
  A    so o meu dataset (s1 + g1 + antiga), 8 acordes + _none
  B8   o meu + Training do v3, 8 acordes + _none
  B11  o meu + Training do v3, 8 acordes + D, A, E + _none, com peso por classe

A, B8 e B11 usam a mesma configuracao do modelo atual (chroma, scaler "feature",
gap_bn, limiar calibrado na validacao) e treinam com as seeds 42, 43 e 44. A
divisao treino/validacao/teste vem do scripts/dividir_treino.py.

Testes (os mesmos para todos, independentes do modelo):
  1. teste_v3: dataset_kaggle/Test, acuracia nas janelas dos 8 acordes do app
  2. take02: meus take02 da s1 e da g1, acuracia nos 8 acordes; nos extras
     (D, A, E) so o B11 consegue acertar, mas o resultado sai para todos
  3. sem acorde: falso aceite nos meus take04 (cordas abafadas) e take06
     (afinando) e nos negativos sinteticos de teste (ruido, percussao, acorde
     em volume baixo)

Uso:
  python -m scripts.comparar_modelos                  # roda o que faltar e gera o resumo
  python -m scripts.comparar_modelos --modelos 0 B8   # so alguns
  python -m scripts.comparar_modelos --so-resumo      # so regenera o comparativo.md

Pula o que ja tem metrics.json, entao pode ser interrompido e retomado. Nao
mexe em trained_model/, dataset/ nem no train_model.py."""
import argparse
import hashlib
import json
import os
import time

import numpy as np

import pipeline as pl
from scripts import dividir_treino, fontes

RESULTADOS = os.path.join(fontes.ROOT_DIR, 'experiments', 'results_duas_fontes')
CACHE = os.path.join(fontes.ROOT_DIR, 'experiments', 'cache')
MODELO_ATUAL = os.path.join(fontes.ROOT_DIR, 'trained_model')

FEATURE = 'chroma'
SCALER = 'feature'
ARCH = 'gap_bn'
SEEDS = [42, 43, 44]

# (fontes do treino, acordes, peso por classe)
MODELOS = {
    'A': ({'proprio'}, fontes.ACORDES, False),
    'B8': ({'proprio', 'kaggle_v3'}, fontes.ACORDES, False),
    'B11': ({'proprio', 'kaggle_v3'}, fontes.acordes_validos(), True),
}
# R = a receita do modelo atual retreinada (so dataset/, split sorteado por
# arquivo, como o train_model.py), para ter 3 seeds dela no Test do v3 e nao
# comparar com uma seed so.
# B8v = os mesmos dados do B8, mas com a validacao sorteada por arquivo (15%,
# como o train_model.py), em vez de separada por pessoa. Serve para saber se a
# diferenca do B8 para o R vem do jeito de validar.
# B11v = B11 com a validacao sorteada; B11vs = o mesmo sem o peso por classe
# (para separar o efeito dos acordes extras do efeito do peso).
# F = o modelo final candidato: B11vs treinado com tudo, inclusive os take02.
# Os take02 deixam de ser teste, entao a coluna take02 do F nao vale; a prova
# dele e o Test do v3 e o sem acorde gravado (take04 e take06).
# Fa = o F com negativos sinteticos de afinacao no treino (o teste de afinacao
# continua sendo o take06 real, que nao entra no treino).
# 0c = o modelo que esta no celular (export/chord_model.tflite), so avaliado.
# B11vsaf = B11vs com os cortes da s1 e da g1 levados ao La 440 no treino (as
# sessoes novas ficaram ~10 cents abaixo; a chroma_stft do celular usa tuning=0
# e nao compensa isso, a chroma_cqt do servidor estima a afinacao e compensa).
# B8vp = B8v com peso menor (PESO_PROPRIO) nas janelas de acorde das minhas
# sessoes novas, para ver se o modelo estava se especializando no meu violao.
ORDEM = ['0', '0c', 'R', 'A', 'B8', 'B8v', 'B8vp', 'B11', 'B11v', 'B11vs', 'B11vsaf', 'F', 'Fa']
PESO_PROPRIO = {'B8vp': 0.5}
FINAIS = ('F', 'Fa')  # treinam com os take02
COM_AFINACAO = ('Fa',)
AFINACAO_CORRIGIDA = ('B11vsaf',)
# nome -> (modelo base, validacao sorteada, forca o peso por classe a)
VARIANTES = {'B8v': ('B8', True, None), 'B8vp': ('B8', True, None), 'B11v': ('B11', True, None),
             'B11vs': ('B11', True, False),
             'B11vsaf': ('B11', True, False), 'F': ('B11', True, False), 'Fa': ('B11', True, False)}

# acordes que diferem por uma nota so: vale olhar a confusao entre eles
PARES_PERIGOSOS = [('A', 'Am'), ('E', 'Em'), ('D', 'Dm')]


# ---------------------------------------------------------------------------
# Features (com cache em experiments/cache/, que nao vai pro Git)
# ---------------------------------------------------------------------------
def _features(pares):
    """Janelas de 1,5 s de cada arquivo (as mesmas do treino de producao).
    Devolve X, y, rms e o indice do arquivo de cada janela."""
    chave = np.array([f'{fontes.caminho_relativo(p)}|{a}' for p, a in pares])
    digest = hashlib.md5('\n'.join(chave).encode()).hexdigest()[:10]
    caminho = os.path.join(CACHE, f'duas_fontes_{FEATURE}_{digest}.npz')
    if os.path.exists(caminho):
        d = np.load(caminho, allow_pickle=False)
        if np.array_equal(d['chave'], chave):
            return d['X'], d['y'], d['rms'], d['fidx']
    print(f'  extraindo {len(pares)} arquivos...', flush=True)
    X, y, _, fidx, rms = pl.build_features(pares, FEATURE)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(caminho, X=X, y=y, rms=rms, fidx=fidx, chave=chave)
    return X, y, rms, fidx


def _negativos_sinteticos(split, pares):
    """Os mesmos negativos sinteticos que o train_model.py usa no _none."""
    from experiments.negatives import build_negatives
    return build_negatives(split, pares, FEATURE)


def _gravados(caminhos):
    return _features([(c, pl.NONE_LABEL) for c in caminhos])


CORDAS_HZ = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]  # as 6 cordas soltas do violao
N_AFINACAO = pl.N_SYNTH_NEGATIVES  # por conjunto, como os outros tipos sinteticos


def audio_afinando(i):
    """Uma corda soando enquanto a tarraxa gira: a nota comeca desafinada (ate
    150 cents) e vai ate a afinacao certa em 0,5 a 1,5 s; varios harmonicos,
    decaimento e um pouco de ruido. Uma corda so, sem formar acorde, mas com as
    mesmas classes de altura que um acorde tem, que e o que confunde o modelo.
    Determinista: o mesmo i gera sempre o mesmo audio."""
    rng = np.random.default_rng([7, int(i)])
    n = int(pl.SAMPLE_RATE * pl.WINDOW_SECONDS)
    t = np.arange(n) / pl.SAMPLE_RATE
    alvo = float(rng.choice(CORDAS_HZ))
    cents = rng.uniform(-150, 150) * np.clip(1 - t / rng.uniform(0.5, 1.5), 0, 1)
    fase = 2 * np.pi * np.cumsum(alvo * 2 ** (cents / 1200)) / pl.SAMPLE_RATE
    x = sum(np.sin(h * fase + rng.uniform(0, 2 * np.pi)) / h ** rng.uniform(0.8, 1.5)
            for h in range(1, int(rng.integers(4, 10)) + 1))
    x = x * np.exp(-t / rng.uniform(0.6, 3.0))
    x = x + pl.pink_noise(n, rng) * rng.uniform(0.0, 0.05) * np.std(x)
    return np.clip(pl._set_rms(x, rng.uniform(0.01, 0.1)), -1.0, 1.0).astype(np.float32)


def negativos_afinacao(split):
    """Features (e rms) dos negativos de afinacao de um conjunto ('train', 'val'
    ou 'test'); os ids seguem as proporcoes do split dos outros sinteticos."""
    audios = [audio_afinando(i) for i in pl.synth_negative_ids(split, n=N_AFINACAO)]
    return np.array([pl.EXTRACTORS[FEATURE](a) for a in audios]), np.array([pl.rms(a) for a in audios])


def montar_treino(partes, afinacao=False):
    """Treino e validacao de um modelo: janelas de acorde + _none (sinteticos +
    meus sem acorde gravados). Devolve dict com X/y/rms de 'treino' e 'val' e
    quantas janelas de acorde vem antes dos negativos na validacao. afinacao:
    acrescenta os negativos sinteticos de afinacao ao _none."""
    out = {}
    for nome, split, none in (('treino', 'train', 'none_treino'), ('val', 'val', 'none_val')):
        X, y, rms, fidx = _features(partes[nome])
        novos = sessao_nova(partes[nome])[fidx] if len(fidx) else np.zeros(0, dtype=bool)
        Xs, _, rms_s = _negativos_sinteticos(split, partes[nome])
        Xg, _, rms_g, _ = _gravados(partes[none])
        if afinacao:
            Xa, rms_a = negativos_afinacao(split)
            Xs, rms_s = np.concatenate([Xs, Xa]), np.concatenate([rms_s, rms_a])
        out[f'n_acordes_{nome}'] = len(X)
        out[f'n_sinteticos_{nome}'] = len(Xs)
        out[f'n_gravados_{nome}'] = len(Xg)
        out[f'X_{nome}'] = np.concatenate([X, Xs, Xg])
        out[f'y_{nome}'] = np.concatenate([y, np.full(len(Xs) + len(Xg), pl.NONE_LABEL)])
        out[f'rms_{nome}'] = np.concatenate([rms, rms_s, rms_g])
        out[f'novos_{nome}'] = np.concatenate([novos, np.zeros(len(Xs) + len(Xg), dtype=bool)])
    return out


def sessao_nova(pares):
    """Para cada arquivo: True se e um corte das minhas sessoes novas (s1, g1),
    False para o v3 e para as gravacoes antigas (*_Alex_*), que ja estavam no
    dataset/ do modelo atual."""
    return np.array([fontes.caminho_relativo(c).startswith('dataset_proprio/') and '_Alex_' not in os.path.basename(c)
                     for c, _ in pares], dtype=bool)


def partes_de_teste():
    """Os testes sao fixos: saem da divisao com as duas fontes e todos os
    acordes, para nao depender de qual modelo esta sendo avaliado."""
    return dividir_treino.dividir(fontes.ler_sources(), {'proprio', 'kaggle_v3'}, fontes.acordes_validos())


def montar_testes(partes, low_volume_so_v3=False):
    """Janelas dos testes. teste_v3 e take02 com rotulo; sem acorde com o tipo
    de cada janela (take04, take06 ou o tipo sintetico). low_volume_so_v3: o
    low_volume sai so de arquivos do Test do v3 (para o modelo F, que treina
    com os take02)."""
    testes = {}
    X, y, rms, _ = _features(partes['teste_v3'])
    testes['teste_v3'] = {'X': X, 'y': y, 'rms': rms}
    X, y, rms, _ = _features(partes['teste_proprio'])
    testes['take02'] = {'X': X, 'y': y, 'rms': rms}

    gravados = partes['none_teste']
    Xg, _, rms_g, fidx = _gravados(gravados)
    tipo_g = np.array([_tipo_gravado(gravados[i]) for i in fidx])
    # low_volume sai dos arquivos de acorde de teste (das duas fontes, so os 8 acordes do app)
    origem = partes['teste_v3'] if low_volume_so_v3 else partes['teste_v3'] + partes['teste_proprio']
    base = [p for p in origem if p[1] in fontes.ACORDES]
    Xs, tipo_s, rms_s = _negativos_sinteticos('test', base)
    testes['sem_acorde'] = {'X': np.concatenate([Xg, Xs]), 'tipo': np.concatenate([tipo_g, tipo_s]),
                            'rms': np.concatenate([rms_g, rms_s])}
    return testes


def _tipo_gravado(caminho):
    take = os.path.basename(caminho).rsplit('_', 1)[0].rsplit('_', 1)[-1]
    return {'take04': 'gravado_abafadas', 'take06': 'gravado_afinando'}.get(take, f'gravado_{take}')


# ---------------------------------------------------------------------------
# Metricas (so numpy, testadas em tests/unit/test_comparar_modelos.py)
# ---------------------------------------------------------------------------
def predicoes(probs, classes):
    classes = np.asarray(classes)
    return classes[np.argmax(probs, axis=1)], probs.max(axis=1)


def aceito(pred, conf, rms, limiar):
    """O que o app mostraria como acorde: passa do filtro de silencio, a classe
    nao e _none e a confianca chega no limiar (pipeline._is_unknown)."""
    return (rms >= pl.SILENCE_THRESHOLD) & (pred != pl.NONE_LABEL) & (conf >= limiar)


def acuracia(pred, y, acordes):
    """Acuracia (argmax, sem limiar) nas janelas cujo rotulo esta em acordes.
    E a mesma conta do accuracy_chords do train_model.py."""
    m = np.isin(y, list(acordes))
    return None if not m.any() else float(np.mean(pred[m] == y[m])), int(m.sum())


def aceitos(pred, conf, rms, y, limiar, acordes):
    """Com o limiar: quanto das janelas o app mostraria (cobertura) e quanto do
    que mostrou estava certo."""
    m = np.isin(y, list(acordes))
    ok = aceito(pred[m], conf[m], rms[m], limiar)
    return {'cobertura': float(ok.mean()) if m.any() else None,
            'acuracia_aceitos': float(np.mean(pred[m][ok] == y[m][ok])) if ok.any() else None}


def falso_aceite(pred, conf, rms, tipos, limiar):
    """Fracao das janelas sem acorde que o app mostraria como acorde, por tipo."""
    ok = aceito(pred, conf, rms, limiar)
    por_tipo = {t: {'falso_aceite': float(ok[tipos == t].mean()), 'n': int((tipos == t).sum())}
                for t in sorted(set(tipos))}
    gravado = np.char.startswith(tipos.astype(str), 'gravado')
    return {'gravado': float(ok[gravado].mean()) if gravado.any() else None,
            'sintetico': float(ok[~gravado].mean()) if (~gravado).any() else None,
            'por_tipo': por_tipo}


def confusao(pred, y):
    """{rotulo verdadeiro: {predito: n}}."""
    out = {}
    for verdadeiro in sorted(set(y)):
        valores, contagens = np.unique(pred[y == verdadeiro], return_counts=True)
        out[str(verdadeiro)] = {str(v): int(c) for v, c in zip(valores, contagens)}
    return out


def confusao_pares(conf_matrix, pares=PARES_PERIGOSOS):
    """Para cada par (maior, menor): quantas janelas de um foram previstas
    como o outro, nos dois sentidos."""
    out = {}
    for a, b in pares:
        for x, z in ((a, b), (b, a)):
            linha = conf_matrix.get(x)
            if linha:
                out[f'{x}->{z}'] = {'n': linha.get(z, 0), 'de': sum(linha.values())}
    return out


def pesos_por_classe(y):
    """Peso 'balanced' por classe, contado em janelas (o que a loss enxerga):
    n_janelas / (n_classes * janelas_da_classe)."""
    classes, contagens = np.unique(y, return_counts=True)
    return {str(c): float(len(y) / (len(classes) * n)) for c, n in zip(classes, contagens)}


def metricas_de_saidas(saidas, classes, limiar):
    """Metricas a partir das probabilidades ja calculadas. saidas = {'teste_v3':
    (probs, y, rms), 'take02': (probs, y, rms), 'sem_acorde': (probs, tipo, rms)}.
    Serve para qualquer modelo: Keras, a media de varias seeds ou o TFLite."""
    m = {'limiar': limiar}
    probs, y, rms = saidas['teste_v3']
    pred, conf = predicoes(probs, classes)
    m['teste_v3'] = dict(zip(('acuracia', 'n_janelas'), acuracia(pred, y, fontes.ACORDES)))
    m['teste_v3'].update(aceitos(pred, conf, rms, y, limiar, fontes.ACORDES))
    m['teste_v3']['confusao'] = confusao(pred, y)

    probs, y, rms = saidas['take02']
    pred, conf = predicoes(probs, classes)
    m['take02'] = dict(zip(('acuracia', 'n_janelas'), acuracia(pred, y, fontes.ACORDES)))
    m['take02'].update(aceitos(pred, conf, rms, y, limiar, fontes.ACORDES))
    m['take02']['acuracia_extras'], m['take02']['n_janelas_extras'] = acuracia(pred, y, fontes.ACORDES_EXTRAS)
    m['take02']['confusao'] = confusao(pred, y)
    m['take02']['pares'] = confusao_pares(m['take02']['confusao'])

    probs, tipo, rms = saidas['sem_acorde']
    pred, conf = predicoes(probs, classes)
    m['sem_acorde'] = falso_aceite(pred, conf, rms, tipo, limiar)
    return m


def _rotulo_do_teste(nome):
    return 'tipo' if nome == 'sem_acorde' else 'y'


def _probs_keras(model, scaler, X):
    return model.predict(pl.apply_scaler(scaler, X, SCALER), verbose=0)


def avaliar(model, scaler, classes, limiar, testes):
    saidas = {k: (_probs_keras(model, scaler, t['X']), t[_rotulo_do_teste(k)], t['rms']) for k, t in testes.items()}
    return metricas_de_saidas(saidas, classes, limiar)


def curva_limiar(model, scaler, classes, testes):
    """Para cada limiar da grade: quanto do Test do v3 o app mostraria, quanto
    do que mostrou estava certo e o falso aceite do sem acorde. So informativa:
    o limiar de verdade e o calibrado na validacao."""
    def prever(X):
        return predicoes(model.predict(pl.apply_scaler(scaler, X, SCALER), verbose=0), classes)

    v3 = testes['teste_v3']
    pred_v, conf_v = prever(v3['X'])
    sa = testes['sem_acorde']
    pred_s, conf_s = prever(sa['X'])
    linhas = []
    for t in pl.THRESHOLD_GRID:
        a = aceitos(pred_v, conf_v, v3['rms'], v3['y'], float(t), fontes.ACORDES)
        f = falso_aceite(pred_s, conf_s, sa['rms'], sa['tipo'], float(t))
        linhas.append({'limiar': float(t), 'cobertura_v3': a['cobertura'], 'acuracia_aceitos_v3': a['acuracia_aceitos'],
                       'falso_aceite_gravado': f['gravado'], 'falso_aceite_sintetico': f['sintetico']})
    return linhas


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------
def calibrar_limiar(model, d, classes):
    """Igual ao train_model.py: janelas de acorde da validacao (sem as de
    silencio) contra os negativos da validacao, criterio da Parte 5."""
    n = d['n_acordes_val']
    probs = model.predict(d['X_val_f'], verbose=0)
    gate = d['rms_val'][:n] >= pl.SILENCE_THRESHOLD
    linhas = pl.threshold_curve(probs[:n][gate], d['y_val'][:n][gate], probs[n:], classes)
    return pl.choose_threshold(linhas)


AFINADOS = os.path.join(CACHE, 'afinados')


def afinacao_do_take(caminhos):
    """Afinacao do take em cents (mediana dos cortes): negativo = abaixo do La 440."""
    import librosa
    cents = []
    for c in caminhos:
        y, sr = librosa.load(c, sr=pl.SAMPLE_RATE)
        cents.append(100 * librosa.estimate_tuning(y=y, sr=sr))
    return float(np.median(cents))


def copia_afinada(caminho, cents, destino=AFINADOS):
    """Copia do corte com a nota deslocada em `cents` (para cima se positivo),
    em destino/. Nunca altera o original. Se a copia ja existe, so devolve o caminho."""
    import librosa
    import soundfile as sf
    saida = os.path.join(destino, f'{cents:+.1f}c', os.path.basename(caminho))
    if not os.path.exists(saida):
        y, sr = librosa.load(caminho, sr=pl.SAMPLE_RATE)
        os.makedirs(os.path.dirname(saida), exist_ok=True)
        sf.write(saida, librosa.effects.pitch_shift(y, sr=sr, n_steps=cents / 100), sr)
    return saida


def corrigir_afinacao(partes, linhas=None):
    """Treino e validacao com os cortes das minhas sessoes novas (as que tem
    take: s1, g1) trazidos para o La 440, take por take. As gravacoes antigas,
    o v3, os testes e o sem acorde ficam como estao. Devolve as partes novas e
    a correcao aplicada em cada take (em cents)."""
    linhas = fontes.ler_sources() if linhas is None else linhas
    take_de = {os.path.join(fontes.ROOT_DIR, l['caminho']): l['take'] for l in linhas
               if l['fonte'] == 'proprio' and l['sessao'] != 'antiga' and l['acorde'] != pl.NONE_LABEL}
    por_take = {}
    for caminho, _ in partes['treino'] + partes['val']:
        if caminho in take_de:
            por_take.setdefault(take_de[caminho], []).append(caminho)
    correcao = {t: round(-afinacao_do_take(cs), 1) for t, cs in sorted(por_take.items())}

    def trocar(pares):
        return sorted((copia_afinada(c, correcao[take_de[c]]) if c in take_de else c, a) for c, a in pares)

    return dict(partes, treino=trocar(partes['treino']), val=trocar(partes['val'])), correcao


def validacao_sorteada(partes):
    """Junta treino e validacao dos acordes e sorteia 15% por arquivo
    (estratificado por acorde), como o train_model.py. Os testes nao entram.
    O sem acorde gravado fica como esta."""
    treino, val = pl.split_train_val(sorted(partes['treino'] + partes['val']))
    return dict(partes, treino=treino, val=val, pessoas_val_v3=[])


def montar_receita_atual():
    """Treino e validacao do jeito do train_model.py: dataset/ sorteado 80/20
    por arquivo e 15% do treino para validacao, com os negativos sinteticos.
    O teste desse sorteio nao e usado; a avaliacao e nos testes fixos."""
    from experiments.data import build
    d = pl.load_dataset(FEATURE, with_none=True, dataset_path=fontes.DATASET_ANTIGO,
                        build_fn=lambda files, feat: build(files, feat, 'clean'),
                        negatives_fn=lambda split, files, feat: _negativos_sinteticos(split, files),
                        verbose=False)
    rms_val = _features(d['val_files'])[2]
    n = len(rms_val)
    return {'X_treino': d['X_train'], 'y_treino': d['y_train'], 'X_val': d['X_val'], 'y_val': d['y_val'],
            # a calibracao do limiar so olha o rms das janelas de acorde (as n primeiras)
            'rms_val': np.concatenate([rms_val, np.ones(len(d['X_val']) - n)]),
            'n_acordes_treino': int((d['y_train'] != pl.NONE_LABEL).sum()),
            'n_sinteticos_treino': int((d['y_train'] == pl.NONE_LABEL).sum()), 'n_gravados_treino': 0,
            'n_acordes_val': n, 'n_sinteticos_val': len(d['X_val']) - n, 'n_gravados_val': 0}, {
            'treino': d['train_files'], 'val': d['val_files'], 'none_treino': [], 'none_val': []}


def sufixo(feature, n_aug):
    """Pedaco do nome que diz a feature e o augmentation, quando nao sao os
    padroes (chroma sem augmentation): B11vs_stft8192_aug2."""
    s = '' if feature == 'chroma' else '_' + feature.replace('chroma_', '')
    return s + (f'_aug{n_aug}' if n_aug else '')


def _features_aug(pares, seed, n_aug):
    """n_aug versoes aumentadas de cada janela de acorde do treino (ruido,
    ganho, passa-banda e deslocamento: experiments/augment.py, o mesmo aug2
    da Parte 3 e do modelo do celular). Dependem da seed, entao o cache tambem."""
    from experiments.data import transform_train_aug
    chave = np.array([f'{fontes.caminho_relativo(p)}|{a}' for p, a in pares])
    digest = hashlib.md5('\n'.join(chave).encode()).hexdigest()[:10]
    caminho = os.path.join(CACHE, f'duas_fontes_{FEATURE}_aug{n_aug}_s{seed}_{digest}.npz')
    if os.path.exists(caminho):
        d = np.load(caminho, allow_pickle=False)
        if np.array_equal(d['chave'], chave) and 'fidx' in d:
            return d['X'], d['y'], d['fidx']
    print(f'  extraindo {n_aug} versoes aumentadas de {len(pares)} arquivos...', flush=True)
    X, y, _, fidx, _ = pl.build_features(pares, FEATURE, transform_train_aug, n_aug, seed)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(caminho, X=X, y=y, fidx=fidx, chave=chave)
    return X, y, fidx


def treinar(nome, seed, testes, dados_cache, n_aug=0):
    if nome == 'R':
        fontes_usadas, acordes, com_peso = {'dataset/'}, fontes.ACORDES, False
    elif nome in VARIANTES:
        base, _, peso = VARIANTES[nome]
        fontes_usadas, acordes, com_peso = MODELOS[base]
        com_peso = com_peso if peso is None else peso
    else:
        fontes_usadas, acordes, com_peso = MODELOS[nome]
    nome_saida = nome + sufixo(FEATURE, n_aug)
    pasta = os.path.join(RESULTADOS, f'{nome_saida}_s{seed}')
    print(f'=== {nome_saida} seed {seed} ===', flush=True)
    if nome not in dados_cache:
        if nome == 'R':
            d, partes = montar_receita_atual()
            dados_cache[nome] = (partes, d)
        else:
            partes = dividir_treino.dividir(fontes.ler_sources(), fontes_usadas, acordes)
            if nome in FINAIS:  # o modelo final treina tambem com os take02
                partes = dict(partes, treino=partes['treino'] + partes['teste_proprio'])
            if nome in VARIANTES and VARIANTES[nome][1]:
                partes = validacao_sorteada(partes)
            if nome in AFINACAO_CORRIGIDA:
                partes, correcao = corrigir_afinacao(partes)
                partes['correcao_cents'] = correcao
            dados_cache[nome] = (partes, montar_treino(partes, afinacao=nome in COM_AFINACAO))
    partes, d = dados_cache[nome]

    import keras
    keras.backend.clear_session()
    pl.set_seeds(seed)
    X_bruto, y_bruto = d['X_treino'], d['y_treino']
    novos = d.get('novos_treino', np.zeros(len(y_bruto), dtype=bool))
    n_aumentadas = 0
    if n_aug:  # so as janelas de acorde do treino; validacao e testes ficam limpos
        Xa, ya, fa = _features_aug(partes['treino'], seed, n_aug)
        X_bruto, y_bruto, n_aumentadas = np.concatenate([X_bruto, Xa]), np.concatenate([y_bruto, ya]), len(Xa)
        novos = np.concatenate([novos, sessao_nova(partes['treino'])[fa]])
    peso_proprio = PESO_PROPRIO.get(nome)
    sample_weight = np.where(novos, peso_proprio, 1.0) if peso_proprio is not None else None

    rotulos = list(acordes) + [pl.NONE_LABEL]
    encoder, y_treino, y_val = pl.encode_labels(rotulos, y_bruto, d['y_val'])
    classes = np.array(encoder.classes_)
    X_treino, scaler = pl.fit_scaler(X_bruto, SCALER)
    d['X_val_f'] = pl.apply_scaler(scaler, d['X_val'], SCALER)

    pesos = pesos_por_classe(y_bruto) if com_peso else None
    class_weight = {i: pesos[str(c)] for i, c in enumerate(classes)} if pesos else None

    model = pl.build_model(ARCH, X_treino.shape[1:], len(classes))
    t0 = time.time()
    hist = model.fit(X_treino, y_treino, epochs=200, batch_size=32, validation_data=(d['X_val_f'], y_val),
                     callbacks=pl.training_callbacks(), class_weight=class_weight,
                     sample_weight=sample_weight, verbose=2)
    segundos = time.time() - t0
    limiar = calibrar_limiar(model, d, classes)

    metricas = avaliar(model, scaler, classes, limiar, testes)
    if nome in FINAIS:
        metricas['curva_limiar'] = curva_limiar(model, scaler, classes, testes)
    metricas.update({
        'take02_no_treino': nome in FINAIS,
        'modelo': nome_saida, 'seed': seed, 'fontes': sorted(fontes_usadas), 'classes': [str(c) for c in classes],
        'feature': FEATURE, 'n_aug': n_aug, 'janelas_aumentadas': n_aumentadas,
        'peso_proprio': peso_proprio, 'janelas_sessoes_novas': int(novos.sum()),
        'peso_por_classe': pesos,
        'arquivos': {k: len(partes[k]) for k in ('treino', 'val', 'none_treino', 'none_val')},
        'janelas': {k: d[k] for k in ('n_acordes_treino', 'n_sinteticos_treino', 'n_gravados_treino',
                                      'n_acordes_val', 'n_sinteticos_val', 'n_gravados_val')},
        'pessoas_val_v3': partes.get('pessoas_val_v3', []),
        'correcao_afinacao_cents': partes.get('correcao_cents'),
        'epocas': len(hist.history['loss']), 'segundos_treino': round(segundos, 1),
    })
    config = pl.make_model_config(FEATURE, SCALER, ARCH, classes, True, limiar,
                                  extra={'seed': seed, 'trained_by': 'scripts/comparar_modelos.py',
                                         'modelo': nome_saida, 'n_aug': n_aug})
    pl.save_model_bundle(model, encoder, scaler, config, pasta, metrics=metricas)
    print(f'  v3={metricas["teste_v3"]["acuracia"]:.4f} take02={metricas["take02"]["acuracia"]:.4f} '
          f'limiar={limiar}', flush=True)


def avaliar_modelo_atual(testes):
    model, encoder, scaler, config = pl.load_model_bundle(MODELO_ATUAL)
    if config['feature'] != FEATURE or pl.scaler_kind(config['scaler']) != SCALER:
        raise SystemExit(f'trained_model/ usa {config["feature"]}/{config["scaler"]}, o script espera {FEATURE}/{SCALER}')
    metricas = avaliar(model, scaler, encoder.classes_, config['confidence_threshold'], testes)
    metricas.update({'modelo': '0', 'seed': config.get('seed'), 'classes': [str(c) for c in encoder.classes_],
                     'origem': 'trained_model/ (so avaliado)'})
    _salvar_metricas('0', metricas)


def _salvar_metricas(nome, metricas):
    pasta = os.path.join(RESULTADOS, nome)
    os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)


EXPORT = os.path.join(fontes.ROOT_DIR, 'export')


def avaliar_tflite(partes, low_volume_so_v3=False):
    """Avalia o modelo que esta no celular (export/chord_model.tflite), que
    recebe o audio bruto da janela e tem a feature embutida. As janelas sao as
    mesmas dos outros modelos (pipeline.make_windowed_segments)."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=os.path.join(EXPORT, 'chord_model.tflite'))
    interp.allocate_tensors()
    entrada, saida = interp.get_input_details()[0]['index'], interp.get_output_details()[0]['index']
    with open(os.path.join(EXPORT, 'model_info.json'), encoding='utf-8') as f:
        info = json.load(f)
    with open(os.path.join(EXPORT, 'labels.json'), encoding='utf-8') as f:
        classes = json.load(f)

    def prob(janela):
        interp.set_tensor(entrada, np.asarray(janela, dtype=np.float32)[np.newaxis])
        interp.invoke()
        return interp.get_tensor(saida)[0]

    def de_arquivos(pares):
        probs, rotulos, rms = [], [], []
        for caminho, rotulo in pares:
            for janela in pl.make_windowed_segments(caminho):
                probs.append(prob(janela))
                rotulos.append(rotulo)
                rms.append(pl.rms(janela))
        return np.array(probs), np.array(rotulos), np.array(rms)

    gravados = [(c, _tipo_gravado(c)) for c in partes['none_teste']]
    origem = partes['teste_v3'] if low_volume_so_v3 else partes['teste_v3'] + partes['teste_proprio']
    base = [p for p in origem if p[1] in fontes.ACORDES]
    sint = [pl.make_negative_audio(k, s) for k, s in pl.negative_specs('test', base)]
    tipos_sint = [k for k, _ in pl.negative_specs('test', base)]
    pg, tg, rg = de_arquivos(gravados)
    saidas = {
        'teste_v3': de_arquivos(partes['teste_v3']),
        'take02': de_arquivos(partes['teste_proprio']),
        'sem_acorde': (np.concatenate([pg, np.array([prob(a) for a in sint])]),
                       np.concatenate([tg, np.array(tipos_sint)]),
                       np.concatenate([rg, np.array([pl.rms(a) for a in sint])])),
    }
    metricas = metricas_de_saidas(saidas, classes, info['confidence_threshold'])
    metricas.update({'modelo': '0c', 'seed': info.get('seed'), 'classes': classes,
                     'origem': f'export/chord_model.tflite ({info.get("model_version")}), o modelo do celular'})
    _salvar_metricas('0c', metricas)


def avaliar_ensemble(nome, testes):
    """Media das probabilidades das seeds ja treinadas de um modelo (cada uma
    com o seu scaler). Nao treina nada. O limiar e a mediana dos limiares das
    seeds; serve so de referencia."""
    pastas = sorted(p for p in os.listdir(RESULTADOS) if p.startswith(f'{nome}_s') and p[len(nome) + 2:].isdigit())
    if not pastas:
        raise SystemExit(f'nenhuma seed de {nome} em {RESULTADOS}')
    somas, classes, limiares, info = None, None, [], None
    for pasta in pastas:
        model, encoder, scaler, config = pl.load_model_bundle(os.path.join(RESULTADOS, pasta))
        if config['feature'] != FEATURE:
            raise SystemExit(f'{pasta} usa {config["feature"]}; rode com --feature {config["feature"]}')
        if classes is not None and list(encoder.classes_) != list(classes):
            raise SystemExit(f'{pasta} tem outras classes')
        classes = encoder.classes_
        limiares.append(config['confidence_threshold'])
        probs = {k: _probs_keras(model, scaler, t['X']) for k, t in testes.items()}
        somas = probs if somas is None else {k: somas[k] + probs[k] for k in somas}
        with open(os.path.join(RESULTADOS, pasta, 'metrics.json'), encoding='utf-8') as f:
            info = json.load(f)
    saidas = {k: (somas[k] / len(pastas), t[_rotulo_do_teste(k)], t['rms']) for k, t in testes.items()}
    metricas = metricas_de_saidas(saidas, classes, float(np.median(limiares)))
    metricas.update({'modelo': f'{nome}_ens', 'seed': None, 'seeds': [int(p.rsplit('_s', 1)[1]) for p in pastas],
                     'classes': [str(c) for c in classes], 'take02_no_treino': info.get('take02_no_treino', False),
                     'origem': f'media das probabilidades de {", ".join(pastas)}'})
    _salvar_metricas(f'{nome}_ens', metricas)


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
def _ler_metricas():
    out = {}
    if not os.path.isdir(RESULTADOS):
        return out
    for pasta in sorted(os.listdir(RESULTADOS)):
        caminho = os.path.join(RESULTADOS, pasta, 'metrics.json')
        if os.path.exists(caminho):
            with open(caminho, encoding='utf-8') as f:
                m = json.load(f)
            out.setdefault(m['modelo'], []).append(m)
    return out


def _celula(valores):
    valores = [v for v in valores if v is not None]
    if not valores:
        return '-'
    pct = [100 * v for v in valores]
    if len(pct) == 1:
        return f'{pct[0]:.1f}%'
    return f'{np.mean(pct):.1f}% ({min(pct):.1f}–{max(pct):.1f})'


def resumo(metricas):
    colunas = [
        ('Teste v3, 8 acordes', lambda m: m['teste_v3']['acuracia']),
        ('take02, 8 acordes', lambda m: None if m.get('take02_no_treino') else m['take02']['acuracia']),
        ('take02, extras (D, A, E)', lambda m: None if m.get('take02_no_treino') or 'D' not in m['classes']
         else m['take02']['acuracia_extras']),
        ('Falso aceite, gravado', lambda m: m['sem_acorde']['gravado']),
        ('Falso aceite, sintético', lambda m: m['sem_acorde']['sintetico']),
    ]
    linhas = ['# Comparativo: modelo atual × treino com as duas fontes', '',
              'Gerado por `python -m scripts.comparar_modelos` a partir dos `metrics.json` desta pasta.',
              'Média das seeds, com o menor e o maior entre parênteses. O modelo 0 é um só (não tem seed).', '',
              '| Modelo | Seeds | ' + ' | '.join(c for c, _ in colunas) + ' | Limiar |',
              '|---|---|' + '---|' * len(colunas) + '---|']
    for nome in ORDEM + sorted(k for k in metricas if k not in ORDEM):
        ms = metricas.get(nome)
        if not ms:
            continue
        if ms[0].get('seeds'):
            seeds = 'média de ' + ', '.join(str(s) for s in ms[0]['seeds'])
        else:
            seeds = '-' if nome in ('0', '0c') else ', '.join(str(m['seed']) for m in ms)
        limiares = sorted({m['limiar'] for m in ms})
        linhas.append(f'| {nome} | {seeds} | ' + ' | '.join(_celula([f(m) for m in ms]) for _, f in colunas)
                      + f' | {", ".join(str(x) for x in limiares)} |')

    exemplo = next((ms[0] for ms in metricas.values() if ms), None)
    if exemplo:
        linhas += ['', 'Tamanho dos testes (janelas de 1,5 s):',
                   f'- Teste v3: {exemplo["teste_v3"]["n_janelas"]} janelas dos 8 acordes',
                   f'- take02: {exemplo["take02"]["n_janelas"]} janelas dos 8 acordes e '
                   f'{exemplo["take02"]["n_janelas_extras"]} dos extras',
                   '- Sem acorde: ' + ', '.join(f'{t} {v["n"]}' for t, v in exemplo['sem_acorde']['por_tipo'].items())]

    b11 = metricas.get('B11')
    if b11:
        linhas += ['', '## B11: confusão entre acordes que diferem por uma nota (take02)', '',
                   '| Seed | ' + ' | '.join(b11[0]['take02']['pares']) + ' |',
                   '|---|' + '---|' * len(b11[0]['take02']['pares'])]
        for m in b11:
            linhas.append(f'| {m["seed"]} | ' + ' | '.join(f'{v["n"]} de {v["de"]}' for v in m['take02']['pares'].values())
                          + ' |')

    linhas += ['', '## Como ler', '',
               '- Acurácia: argmax em todas as janelas, sem limiar (a mesma conta do `accuracy_chords` do `train_model.py`).',
               '- Falso aceite: fração das janelas sem acorde que o app mostraria como acorde (passa do filtro de '
               'silêncio, a classe não é `_none` e a confiança chega no limiar do modelo).',
               '- Os take02 são do mesmo dia, cômodo e microfone do treino: o número é otimista e serve para comparar '
               'modelos, não para prometer acerto.',
               '- O modelo 0 e o A nunca viram o Test do v3; o 0 também nunca viu a s1 nem a g1.',
               '- O B11 muda duas coisas em relação ao B8 (os acordes extras e o peso por classe). Se ele perder '
               'acurácia, não dá para dizer qual das duas causou.']
    return '\n'.join(linhas) + '\n'


def main(argv=None):
    global FEATURE
    parser = argparse.ArgumentParser(description='Treina e compara os modelos 0, A, B8 e B11.')
    parser.add_argument('--modelos', nargs='+', default=[], choices=ORDEM)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--feature', default='chroma', choices=sorted(pl.EXTRACTORS),
                        help='chroma = a do servidor (trained_model/); chroma_stft8192 = a do celular (TFLite)')
    parser.add_argument('--aug', type=int, default=0, choices=[0, 1, 2],
                        help='versoes aumentadas por janela de treino (2 = o aug2 do modelo do celular)')
    parser.add_argument('--ensemble', nargs='+', default=[], metavar='MODELO',
                        help='avalia a media das seeds ja treinadas (ex.: B11vs), sem treinar')
    parser.add_argument('--so-resumo', action='store_true')
    args = parser.parse_args(argv)
    FEATURE = args.feature
    if not (args.modelos or args.ensemble or args.so_resumo):
        args.modelos = ORDEM

    if not args.so_resumo:
        testes = montar_testes(partes_de_teste())
        testes_por_modelo = {}
        dados_cache = {}
        for nome in args.ensemble:
            avaliar_ensemble(nome + sufixo(FEATURE, args.aug), testes)
        for nome in args.modelos:
            if nome == '0':
                if not os.path.exists(os.path.join(RESULTADOS, '0', 'metrics.json')):
                    avaliar_modelo_atual(testes)
                continue
            if nome == '0c':
                if not os.path.exists(os.path.join(RESULTADOS, '0c', 'metrics.json')):
                    avaliar_tflite(partes_de_teste())
                continue
            for seed in args.seeds:
                nome_saida = nome + sufixo(FEATURE, args.aug)
                if os.path.exists(os.path.join(RESULTADOS, f'{nome_saida}_s{seed}', 'metrics.json')):
                    print(f'[pula] {nome_saida}_s{seed}', flush=True)
                    continue
                if nome in FINAIS and 'final' not in testes_por_modelo:
                    testes_por_modelo['final'] = montar_testes(partes_de_teste(), low_volume_so_v3=True)
                treinar(nome, seed, testes_por_modelo.get('final' if nome in FINAIS else nome, testes),
                        dados_cache, n_aug=args.aug)

    texto = resumo(_ler_metricas())
    os.makedirs(RESULTADOS, exist_ok=True)
    with open(os.path.join(RESULTADOS, 'comparativo.md'), 'w', encoding='utf-8') as f:
        f.write(texto)
    print(texto)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
