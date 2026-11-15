"""Divisao treino/validacao/teste a partir do sources.csv (scripts/dividir_treino.py)."""
from scripts import dividir_treino, fontes


def linha(caminho, acorde, fonte, sessao, take, uso='treino'):
    return {'caminho': caminho, 'acorde': acorde, 'fonte': fonte, 'sessao': sessao, 'take': take, 'uso': uso}


def cortes(acorde, sessao, take, n=8, instrumento='violao'):
    nome = f'{acorde}_{instrumento}_{sessao}_{take}'
    return [linha(f'dataset_proprio/{fontes.rotulo(acorde)}/{nome}_{i:03d}.wav', fontes.rotulo(acorde),
                  'proprio', sessao, nome) for i in range(1, n + 1)]


def v3(acorde, pessoa, uso, n=2):
    pasta = 'Training' if uso == 'treino' else 'Test'
    return [linha(f'dataset_kaggle/{pasta}/{acorde}/{acorde}_{pessoa}_{i}.wav', acorde, 'kaggle_v3', pessoa,
                  f'{acorde}_{pessoa}_{i}', uso) for i in range(1, n + 1)]


def montar():
    linhas = []
    for acorde in ('C', 'G', 'D'):
        linhas += cortes(acorde, 's1', 'take01') + cortes(acorde, 's1', 'take02')
    linhas += cortes('none', 's1', 'take01') + cortes('none', 's1', 'take04')
    linhas.append(linha('dataset_proprio/C/C_Alex_Acoustic_Mic.wav', 'C', 'proprio', 'antiga', 'C_Alex_Acoustic_Mic'))
    for i in range(20):
        linhas += v3('C', f'AcusticPlug{i}', 'treino') + v3('G', f'AcusticPlug{i}', 'treino')
    linhas += v3('C', 'AcusticPlug9 ', 'treino')  # o espaco no fim que o v3 tem
    linhas += v3('C', 'Electric1_LInda', 'teste') + v3('G', 'Electric1_LInda', 'teste')
    return linhas


def nomes(pares):
    return [p.rsplit('/', 1)[-1] for p, _ in pares]


def test_take02_e_v3_de_teste_ficam_fora_do_treino():
    p = dividir_treino.dividir(montar(), {'proprio', 'kaggle_v3'}, ['C', 'G'], raiz='/r')
    assert all('take02' in n for n in nomes(p['teste_proprio'])) and len(p['teste_proprio']) == 16
    assert all('LInda' in n for n in nomes(p['teste_v3'])) and len(p['teste_v3']) == 4
    vistos = nomes(p['treino']) + nomes(p['val'])
    assert not any('take02' in n or 'LInda' in n for n in vistos)
    assert 'C_Alex_Acoustic_Mic.wav' in nomes(p['treino'])


def test_validacao_separa_pessoas_do_v3_e_1_de_cada_4_cortes():
    p = dividir_treino.dividir(montar(), {'proprio', 'kaggle_v3'}, ['C', 'G'], raiz='/r')
    pessoa = lambda n: n.split('_')[1]
    val_v3 = {pessoa(n) for n in nomes(p['val']) if 'Plug' in n}
    treino_v3 = {pessoa(n) for n in nomes(p['treino']) if 'Plug' in n}
    assert val_v3 and not (val_v3 & treino_v3)
    assert len(val_v3) == round(0.15 * 20)
    meus_val = [n for n in nomes(p['val']) if '_s1_' in n]
    assert sorted({n[-7:-4] for n in meus_val}) == ['004', '008']
    # 'AcusticPlug9 ' (com espaco) e 'AcusticPlug9' sao a mesma pessoa
    assert ('AcusticPlug9' in val_v3) == all('AcusticPlug9_' not in n for n in nomes(p['treino']))


def test_acordes_extras_so_entram_quando_pedidos():
    linhas = montar()
    so_app = dividir_treino.dividir(linhas, {'proprio'}, fontes.ACORDES, raiz='/r')
    com_extras = dividir_treino.dividir(linhas, {'proprio'}, fontes.acordes_validos(), raiz='/r')
    assert 'D' not in {a for _, a in so_app['treino'] + so_app['teste_proprio']}
    assert 'D' in {a for _, a in com_extras['treino']} and 'D' in {a for _, a in com_extras['teste_proprio']}


def test_sem_acorde_gravado_tem_takes_so_de_teste():
    p = dividir_treino.dividir(montar(), {'proprio'}, ['C'], raiz='/r')
    assert len(p['none_teste']) == 8 and all('take04' in c for c in p['none_teste'])
    assert len(p['none_treino']) == 6 and len(p['none_val']) == 2


def test_so_fontes_pedidas():
    p = dividir_treino.dividir(montar(), {'proprio'}, ['C', 'G'], raiz='/r')
    assert not p['teste_v3'] and not any('Plug' in n for n in nomes(p['treino']))
