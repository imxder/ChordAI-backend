"""Download dos datasets (baixar_dataset.py), sem rede: so a arrumacao das pastas."""
import baixar_dataset


def test_pasta_unica_na_raiz_sobe_um_nivel(tmp_path):
    # Como o Kaggle entregou o meu dataset: tudo dentro de chordai_dataset_kaggle/
    (tmp_path / 'chordai_dataset_kaggle' / 'C').mkdir(parents=True)
    (tmp_path / 'chordai_dataset_kaggle' / 'C' / 'C_violao_s1_take01_001.wav').write_bytes(b'x')
    (tmp_path / 'chordai_dataset_kaggle' / 'metadata.csv').write_text('arquivo\n')
    baixar_dataset.tirar_pasta_de_cima(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['C', 'metadata.csv']
    assert (tmp_path / 'C' / 'C_violao_s1_take01_001.wav').exists()


def test_v3_com_training_e_test_na_raiz_nao_muda(tmp_path):
    for pasta in ('Training', 'Test'):
        (tmp_path / pasta / 'C').mkdir(parents=True)
    baixar_dataset.tirar_pasta_de_cima(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['Test', 'Training']


def test_pasta_que_ja_tem_arquivo_nao_baixa_de_novo(tmp_path, monkeypatch):
    (tmp_path / 'C').mkdir()
    chamadas = []
    monkeypatch.setattr(baixar_dataset, 'baixar_zip', lambda *args: chamadas.append(args))
    assert baixar_dataset.baixar_do_kaggle('dono/dataset', str(tmp_path)) is None
    assert chamadas == []


def test_main_escolhe_o_dataset_pela_opcao(monkeypatch):
    chamadas = []
    monkeypatch.setattr(baixar_dataset, 'baixar_do_kaggle',
                        lambda slug, destino: chamadas.append((slug, destino.rsplit('/', 1)[-1])))
    monkeypatch.setattr(baixar_dataset, 'baixar_antigo', lambda: chamadas.append('antigo'))

    assert baixar_dataset.main([]) == 0
    assert baixar_dataset.main(['--proprio']) == 0
    assert baixar_dataset.main(['--antigo']) == 0
    assert chamadas == [
        (baixar_dataset.GUITAR_CHORDS_V3, 'dataset_kaggle'),
        (baixar_dataset.DATASET_PROPRIO, 'dataset_proprio'),
        'antigo',
    ]


def _zip(caminho, arquivos):
    import zipfile
    with zipfile.ZipFile(caminho, 'w') as z:
        for nome, conteudo in arquivos.items():
            z.writestr(nome, conteudo)


def test_baixa_o_zip_e_grava_no_destino(tmp_path):
    origem = tmp_path / 'origem.zip'
    origem.write_bytes(b'conteudo do zip')
    destino = tmp_path / 'baixado.zip'
    baixar_dataset.baixar_zip(origem.as_uri(), str(destino))  # file:// em vez da rede
    assert destino.read_bytes() == b'conteudo do zip'


def test_extrair_ignora_a_pasta_do_mac(tmp_path, capsys):
    zip_path = tmp_path / 'd.zip'
    _zip(zip_path, {'C/c1.wav': b'1', 'C/c2.wav': b'2', '__MACOSX/C/._c1.wav': b'lixo'})
    baixar_dataset.extrair(str(zip_path), str(tmp_path / 'saida'))
    assert sorted(p.name for p in (tmp_path / 'saida').iterdir()) == ['C']
    assert '2 arquivos .wav' in capsys.readouterr().out


def test_baixar_do_kaggle_baixa_extrai_e_arruma(tmp_path, monkeypatch):
    urls = []

    def baixar(url, destino_zip):
        urls.append(url)
        _zip(destino_zip, {'chordai_dataset_kaggle/C/c1.wav': b'1'})

    monkeypatch.setattr(baixar_dataset, 'baixar_zip', baixar)
    destino = tmp_path / 'dataset_proprio'
    baixar_dataset.baixar_do_kaggle(baixar_dataset.DATASET_PROPRIO, str(destino))
    assert urls == ['https://www.kaggle.com/api/v1/datasets/download/imxder/chordai-guitar-chords-v1']
    assert (destino / 'C' / 'c1.wav').exists()  # a pasta de cima sumiu


def test_baixar_antigo_usa_o_gdown(tmp_path, monkeypatch):
    import sys
    import types
    baixados = []

    def download(url, destino, quiet=False):
        baixados.append(url)
        _zip(destino, {'dataset/C/c1.wav': b'1'})

    monkeypatch.setitem(sys.modules, 'gdown', types.SimpleNamespace(download=download))
    monkeypatch.setattr(baixar_dataset, 'ROOT_DIR', str(tmp_path))
    baixar_dataset.baixar_antigo()
    assert baixados == [f'https://drive.google.com/uc?id={baixar_dataset.DRIVE_ANTIGO}']
    assert (tmp_path / 'dataset' / 'C' / 'c1.wav').exists()
