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
