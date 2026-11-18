# Testes e cobertura

O Playbook pede **75% de cobertura** no backend próprio de um projeto Mobile. Aqui o backend que roda de verdade é o pipeline de treino (`pipeline.py`, `train_model.py`) e a API (`api/app.py`).

## Números (18/11/2026)

Medido com `pytest-cov` e só o que o CI roda (sem o dataset em disco):

| Arquivo | Linhas | Cobertas | Cobertura |
|---|---|---|---|
| `api/app.py` | 35 | 35 | 100% |
| `pipeline.py` | 459 | 455 | 99% |
| `train_model.py` | 121 | 121 | 100% |
| **Produção (os três acima)** | **615** | **611** | **99%** |
| `export/build_tflite.py` | 305 | 249 | 82% |
| `scripts/comparar_modelos.py` | 447 | 401 | 90% |
| `scripts/cortar_takes.py` | 205 | 203 | 99% |
| `scripts/exportar_ableton.py` | 135 | 133 | 99% |
| `scripts/dividir_treino.py`, `fontes.py`, `montar_sources.py`, `baixar_dataset.py` | 242 | 242 | 100% |
| **Total do repositório** | **1949** | **1839** | **94%** |

Antes desta rodada de testes (17/11) o código de produção estava em 312 de 634 linhas, **49,2%**.

O CI manda o `coverage.xml` pro SonarCloud em todo push; o badge do README e o histórico ficam lá: [sonarcloud.io/component_measures?id=imxder_ChordAI-backend&metric=coverage](https://sonarcloud.io/component_measures?id=imxder_ChordAI-backend&metric=coverage). O SonarCloud conta por linha e o coverage.py por comando, então o número dele pode sair um pouco diferente.

## Como gerar o relatório

```bash
python -m pytest -m "not dataset" --cov=. --cov-report=term-missing
coverage report --include="api/*,pipeline.py,train_model.py"
```

O primeiro roda os testes e mede; o segundo mostra só os três arquivos de produção. `--cov-report=html` gera uma página em `htmlcov/index.html` com as linhas não cobertas marcadas em vermelho.

## O que ficou fora da conta, e por quê

Está tudo escrito no [`.coveragerc`](../.coveragerc) e no `sonar-project.properties`:

- **`api/serve.py`**: só joga a saída num arquivo de log e sobe o waitress na porta 5000. Testar seria subir um servidor de verdade no CI. A rota `/predict_chord`, que é o que importa, está em `api/app.py` e tem teste.
- **`experiments/`**: os scripts de pesquisa que rodei na minha máquina pra comparar features e modelos (os resultados estão em `experiments/results*`). Não fazem parte do app, da API nem do treino que gera os modelos, e dependem do dataset inteiro em disco. O que eles usam do `pipeline.py` tem teste.
- **Os blocos `if __name__ == '__main__':`**: só rodam quando chamo o arquivo pela linha de comando (`python train_model.py`).

O que continua sem teste: no `export/build_tflite.py`, a escolha da seed e o carregamento do modelo C1, que leem os `model.h5` de `experiments/results/` (eles não vão pro Git); no `comparar_modelos.py`, a montagem dos dados de cada variante (R, V3, B8v...), que precisa do `data/sources.csv` com os áudios em disco; e no `pipeline.py`, quatro linhas de casos raros.

## Como os testes funcionam sem o dataset

Os áudios do dataset não vão pro Git, então os testes usam **acordes sintéticos** (`tests/sintetico.py`): cada nota é uma senoide com dois harmônicos e um decaimento leve, como uma corda. Como a gente sabe quais notas estão no sinal, dá pra conferir que o chroma de um C destaca C, E e G.

O export (`tests/unit/test_build_tflite.py`) monta o grafo com a chroma_stft dentro, com uma CNN pequena de pesos aleatórios, e confere que a feature dele bate com a do librosa (erro menor que 0,0001), que a conversão só usa ops nativas do TFLite e que o int8 não quantiza a janela nem a matriz de chroma.

O teste de treino (`tests/unit/test_train_model.py`) monta um dataset sintético com 4 acordes x 10 arquivos numa pasta temporária e treina o modelo de ponta a ponta, com 20 negativos sintéticos de cada tipo em vez de 370 (só pra rodar em segundos). O código de produção não muda pro teste.

## Onde está cada teste

| Arquivo | O que confere |
|---|---|
| `tests/unit/test_dataset_split.py` | listagem do dataset, split por arquivo sem vazamento, validação, janelas de 1,5 s |
| `tests/unit/test_features_scaler.py` | formato de cada extrator, chroma das notas certas, scaler, one-hot, reamostragem |
| `tests/unit/test_negativos.py` | negativos `_none`: reproduzíveis, acima do silêncio, divisão dos ids, `load_dataset` com `_none` |
| `tests/unit/test_limiar_e_modelo.py` | curva e escolha do limiar, as 3 CNNs, salvar e carregar o modelo |
| `tests/unit/test_train_model.py` | treino de ponta a ponta, modelo corrompido, áudio curto |
| `tests/unit/test_unknown_chord.py` | `_none` e confiança abaixo do limiar viram desconhecido |
| `tests/unit/test_model_config.py` | `model_config.json` e o pipeline antigo |
| `tests/unit/test_chroma_stft.py` | a chroma_stft montada à mão bate com a do librosa |
| `tests/unit/test_export_tflite.py` | o `.tflite` dá as mesmas probabilidades do modelo em Python |
| `tests/unit/test_build_tflite.py` | o export: grafo = librosa, ops nativas, int8, arquivos do app e hash |
| `tests/unit/test_comparar_modelos_fluxo.py` | o fluxo da comparação de modelos: cache, treino, ensemble, .tflite, comparativo.md |
| `tests/integration/test_api.py` | `/predict_chord`: acorde, silêncio, 400 e 503, e a API subindo sem modelo |
| os outros `tests/unit/test_*.py` | os scripts de `scripts/` (cortar takes, fontes, comparar modelos...) |
