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
ORDEM = ['0', 'R', 'A', 'B8', 'B11']

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


def montar_treino(partes):
    """Treino e validacao de um modelo: janelas de acorde + _none (sinteticos +
    meus sem acorde gravados). Devolve dict com X/y/rms de 'treino' e 'val' e
    quantas janelas de acorde vem antes dos negativos na validacao."""
    out = {}
    for nome, split, none in (('treino', 'train', 'none_treino'), ('val', 'val', 'none_val')):
        X, y, rms, _ = _features(partes[nome])
        Xs, _, rms_s = _negativos_sinteticos(split, partes[nome])
        Xg, _, rms_g, _ = _gravados(partes[none])
        out[f'n_acordes_{nome}'] = len(X)
        out[f'n_sinteticos_{nome}'] = len(Xs)
        out[f'n_gravados_{nome}'] = len(Xg)
        out[f'X_{nome}'] = np.concatenate([X, Xs, Xg])
        out[f'y_{nome}'] = np.concatenate([y, np.full(len(Xs) + len(Xg), pl.NONE_LABEL)])
        out[f'rms_{nome}'] = np.concatenate([rms, rms_s, rms_g])
    return out


def partes_de_teste():
    """Os testes sao fixos: saem da divisao com as duas fontes e todos os
    acordes, para nao depender de qual modelo esta sendo avaliado."""
    return dividir_treino.dividir(fontes.ler_sources(), {'proprio', 'kaggle_v3'}, fontes.acordes_validos())


def montar_testes(partes):
    """Janelas dos testes. teste_v3 e take02 com rotulo; sem acorde com o tipo
    de cada janela (take04, take06 ou o tipo sintetico)."""
    testes = {}
    X, y, rms, _ = _features(partes['teste_v3'])
    testes['teste_v3'] = {'X': X, 'y': y, 'rms': rms}
    X, y, rms, _ = _features(partes['teste_proprio'])
    testes['take02'] = {'X': X, 'y': y, 'rms': rms}

    gravados = partes['none_teste']
    Xg, _, rms_g, fidx = _gravados(gravados)
    tipo_g = np.array([_tipo_gravado(gravados[i]) for i in fidx])
    # low_volume sai dos arquivos de acorde de teste (das duas fontes, so os 8 acordes do app)
    base = [p for p in partes['teste_v3'] + partes['teste_proprio'] if p[1] in fontes.ACORDES]
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


def avaliar(model, scaler, classes, limiar, testes):
    def prever(X):
        return predicoes(model.predict(pl.apply_scaler(scaler, X, SCALER), verbose=0), classes)

    m = {'limiar': limiar}
    t = testes['teste_v3']
    pred, conf = prever(t['X'])
    m['teste_v3'] = dict(zip(('acuracia', 'n_janelas'), acuracia(pred, t['y'], fontes.ACORDES)))
    m['teste_v3'].update(aceitos(pred, conf, t['rms'], t['y'], limiar, fontes.ACORDES))
    m['teste_v3']['confusao'] = confusao(pred, t['y'])

    t = testes['take02']
    pred, conf = prever(t['X'])
    m['take02'] = dict(zip(('acuracia', 'n_janelas'), acuracia(pred, t['y'], fontes.ACORDES)))
    m['take02'].update(aceitos(pred, conf, t['rms'], t['y'], limiar, fontes.ACORDES))
    m['take02']['acuracia_extras'], m['take02']['n_janelas_extras'] = acuracia(pred, t['y'], fontes.ACORDES_EXTRAS)
    m['take02']['confusao'] = confusao(pred, t['y'])
    m['take02']['pares'] = confusao_pares(m['take02']['confusao'])

    t = testes['sem_acorde']
    pred, conf = prever(t['X'])
    m['sem_acorde'] = falso_aceite(pred, conf, t['rms'], t['tipo'], limiar)
    return m


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


def treinar(nome, seed, testes, dados_cache):
    if nome == 'R':
        fontes_usadas, acordes, com_peso = {'dataset/'}, fontes.ACORDES, False
    else:
        fontes_usadas, acordes, com_peso = MODELOS[nome]
    pasta = os.path.join(RESULTADOS, f'{nome}_s{seed}')
    print(f'=== {nome} seed {seed} ===', flush=True)
    if nome not in dados_cache:
        if nome == 'R':
            d, partes = montar_receita_atual()
            dados_cache[nome] = (partes, d)
        else:
            partes = dividir_treino.dividir(fontes.ler_sources(), fontes_usadas, acordes)
            dados_cache[nome] = (partes, montar_treino(partes))
    partes, d = dados_cache[nome]

    import keras
    keras.backend.clear_session()
    pl.set_seeds(seed)
    rotulos = list(acordes) + [pl.NONE_LABEL]
    encoder, y_treino, y_val = pl.encode_labels(rotulos, d['y_treino'], d['y_val'])
    classes = np.array(encoder.classes_)
    X_treino, scaler = pl.fit_scaler(d['X_treino'], SCALER)
    d['X_val_f'] = pl.apply_scaler(scaler, d['X_val'], SCALER)

    pesos = pesos_por_classe(d['y_treino']) if com_peso else None
    class_weight = {i: pesos[str(c)] for i, c in enumerate(classes)} if pesos else None

    model = pl.build_model(ARCH, X_treino.shape[1:], len(classes))
    t0 = time.time()
    hist = model.fit(X_treino, y_treino, epochs=200, batch_size=32, validation_data=(d['X_val_f'], y_val),
                     callbacks=pl.training_callbacks(), class_weight=class_weight, verbose=2)
    segundos = time.time() - t0
    limiar = calibrar_limiar(model, d, classes)

    metricas = avaliar(model, scaler, classes, limiar, testes)
    metricas.update({
        'modelo': nome, 'seed': seed, 'fontes': sorted(fontes_usadas), 'classes': [str(c) for c in classes],
        'peso_por_classe': pesos,
        'arquivos': {k: len(partes[k]) for k in ('treino', 'val', 'none_treino', 'none_val')},
        'janelas': {k: d[k] for k in ('n_acordes_treino', 'n_sinteticos_treino', 'n_gravados_treino',
                                      'n_acordes_val', 'n_sinteticos_val', 'n_gravados_val')},
        'pessoas_val_v3': partes.get('pessoas_val_v3', []),
        'epocas': len(hist.history['loss']), 'segundos_treino': round(segundos, 1),
    })
    config = pl.make_model_config(FEATURE, SCALER, ARCH, classes, True, limiar,
                                  extra={'seed': seed, 'trained_by': 'scripts/comparar_modelos.py',
                                         'modelo': nome})
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
    pasta = os.path.join(RESULTADOS, '0')
    os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)


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
        ('take02, 8 acordes', lambda m: m['take02']['acuracia']),
        ('take02, extras (D, A, E)', lambda m: m['take02']['acuracia_extras'] if 'D' in m['classes'] else None),
        ('Falso aceite, gravado', lambda m: m['sem_acorde']['gravado']),
        ('Falso aceite, sintético', lambda m: m['sem_acorde']['sintetico']),
    ]
    linhas = ['# Comparativo: modelo atual × treino com as duas fontes', '',
              'Gerado por `python -m scripts.comparar_modelos` a partir dos `metrics.json` desta pasta.',
              'Média das seeds, com o menor e o maior entre parênteses. O modelo 0 é um só (não tem seed).', '',
              '| Modelo | Seeds | ' + ' | '.join(c for c, _ in colunas) + ' | Limiar |',
              '|---|---|' + '---|' * len(colunas) + '---|']
    for nome in ORDEM:
        ms = metricas.get(nome)
        if not ms:
            continue
        seeds = ', '.join(str(m['seed']) for m in ms) if nome != '0' else '-'
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
    parser = argparse.ArgumentParser(description='Treina e compara os modelos 0, A, B8 e B11.')
    parser.add_argument('--modelos', nargs='+', default=ORDEM, choices=ORDEM)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--so-resumo', action='store_true')
    args = parser.parse_args(argv)

    if not args.so_resumo:
        testes = montar_testes(partes_de_teste())
        dados_cache = {}
        for nome in args.modelos:
            if nome == '0':
                if not os.path.exists(os.path.join(RESULTADOS, '0', 'metrics.json')):
                    avaliar_modelo_atual(testes)
                continue
            for seed in args.seeds:
                if os.path.exists(os.path.join(RESULTADOS, f'{nome}_s{seed}', 'metrics.json')):
                    print(f'[pula] {nome}_s{seed}', flush=True)
                    continue
                treinar(nome, seed, testes, dados_cache)

    texto = resumo(_ler_metricas())
    os.makedirs(RESULTADOS, exist_ok=True)
    with open(os.path.join(RESULTADOS, 'comparativo.md'), 'w', encoding='utf-8') as f:
        f.write(texto)
    print(texto)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
