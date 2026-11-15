"""montar_sources: copia as gravacoes antigas e lista o V3 sem estragar as linhas dos takes."""
import os

import numpy as np
import soundfile as sf

from scripts import fontes, montar_sources


def wav(caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    sf.write(caminho, np.zeros(1600, dtype=np.float32), 16000)


def test_copia_antigas_lista_kaggle_e_mantem_os_takes(tmp_path):
    antigo = tmp_path / 'dataset'
    wav(str(antigo / 'Am' / 'Am_Alex_Acoustic_Mic.wav'))
    wav(str(antigo / 'C' / 'C_Alex_Electric_Plugged_2.wav'))
    wav(str(antigo / 'C' / 'C_AcousticGuitar_RodrigoMercador_1.wav'))
    kaggle = tmp_path / 'dataset_kaggle'
    wav(str(kaggle / 'Training' / 'G' / 'G_AcousticGuitar_RodrigoMercador_3.wav'))
    wav(str(kaggle / 'Test' / 'G' / 'G_AcusticPlug26_1.wav'))
    proprio = tmp_path / 'dataset_proprio'
    sources = str(tmp_path / 'sources.csv')
    fontes.salvar_sources([{'caminho': 'dataset_proprio/Am/Am_violao_s1_take01_001.wav', 'acorde': 'Am',
                            'fonte': 'proprio', 'sessao': 's1', 'take': 'Am_violao_s1_take01', 'uso': 'treino'}],
                          sources)

    for _ in range(2):
        resultado = montar_sources.montar(str(antigo), str(proprio), str(kaggle), sources)

    assert resultado == (2, 2, 1)
    assert sorted(os.listdir(proprio / 'C')) == ['C_Alex_Electric_Plugged_2.wav']
    assert os.path.exists(antigo / 'Am' / 'Am_Alex_Acoustic_Mic.wav')

    linhas = {os.path.basename(l['caminho']): l for l in fontes.ler_sources(sources)}
    assert len(linhas) == 5
    assert (linhas['Am_Alex_Acoustic_Mic.wav']['instrumento'], linhas['Am_Alex_Acoustic_Mic.wav']['microfone']) == ('violao', 'microfone')
    assert (linhas['C_Alex_Electric_Plugged_2.wav']['instrumento'], linhas['C_Alex_Electric_Plugged_2.wav']['sessao']) == ('guitarra', 'antiga')
    assert linhas['G_AcousticGuitar_RodrigoMercador_3.wav']['uso'] == 'treino'
    assert linhas['G_AcusticPlug26_1.wav']['uso'] == 'teste'
    assert linhas['G_AcusticPlug26_1.wav']['fonte'] == 'kaggle_v3'
    assert linhas['G_AcousticGuitar_RodrigoMercador_3.wav']['sessao'] == 'AcousticGuitar_RodrigoMercador'
    assert 'Am_violao_s1_take01_001.wav' in linhas
