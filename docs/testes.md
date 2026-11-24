# Testes e cobertura

O Playbook pede 75% de cobertura no backend próprio de um projeto Mobile. Aqui o backend que roda de verdade é o pipeline de treino (`pipeline.py`, `train_model.py`) e a API (`api/app.py`).

## Números (23/11/2026)

Medido em 23/11/2026, num Mac, com `python -m pytest -m "not dataset" --cov=. --cov-report=term-missing` e só o que o CI roda (sem o dataset em disco). Os números saíram iguais aos de 18/11: de lá para cá só mudou documentação.

São 224 testes em 18 arquivos, e os 224 passam. A contagem vem de `python -m pytest --collect-only -q`, e um teste com vários casos conta uma vez por caso.

Cobertura:

| Arquivo | Linhas | Cobertas | Cobertura |
|---|---|---|---|
| `api/app.py` | 35 | 35 | 100% |
| `pipeline.py` | 459 | 455 | 99% |
| `train_model.py` | 121 | 121 | 100% |
| Produção (os três acima) | 615 | 611 | 99% |
| `export/build_tflite.py` | 305 | 249 | 82% |
| `scripts/comparar_modelos.py` | 447 | 401 | 90% |
| `scripts/cortar_takes.py` | 205 | 203 | 99% |
| `scripts/exportar_ableton.py` | 135 | 133 | 99% |
| `scripts/dividir_treino.py`, `fontes.py`, `montar_sources.py`, `baixar_dataset.py` | 242 | 242 | 100% |
| Total do repositório | 1949 | 1839 | 94% |

Antes desta rodada de testes (17/11) o código de produção estava em 312 de 634 linhas, 49,2%.

O CI manda o `coverage.xml` para o SonarCloud em todo push; o badge do README e o histórico ficam lá: [sonarcloud.io/component_measures?id=imxder_ChordAI-backend&metric=coverage](https://sonarcloud.io/component_measures?id=imxder_ChordAI-backend&metric=coverage). O SonarCloud conta por linha e o coverage.py por comando, então o número dele pode sair um pouco diferente.

## Como gerar o relatório

```bash
python -m pytest -m "not dataset" --cov=. --cov-report=term-missing
coverage report --include="api/*,pipeline.py,train_model.py"
```

O primeiro roda os testes e mede; o segundo mostra só os três arquivos de produção. `--cov-report=html` gera uma página em `htmlcov/index.html` com as linhas não cobertas marcadas em vermelho.

## O que ficou fora da conta, e por quê

Está tudo escrito no [`.coveragerc`](../.coveragerc) e no `sonar-project.properties`:

- `api/serve.py`: só joga a saída num arquivo de log e sobe o waitress na porta 5000. Testar seria subir um servidor de verdade no CI. A rota `/predict_chord`, que é o que importa, está em `api/app.py` e tem teste.
- `experiments/`: os scripts de pesquisa rodados na máquina de desenvolvimento para comparar features e modelos (os resultados estão em `experiments/results*`). Não fazem parte do app, da API nem do treino que gera os modelos, e dependem do dataset inteiro em disco. O que eles usam do `pipeline.py` tem teste.
- Os blocos `if __name__ == '__main__':`: só rodam quando o arquivo é chamado pela linha de comando (`python train_model.py`).

O que continua sem teste: no `export/build_tflite.py`, a escolha da seed e o carregamento do modelo C1, que leem os `model.h5` de `experiments/results/` (eles não vão para o Git); no `comparar_modelos.py`, a montagem dos dados de cada variante (R, V3, B8v...), que precisa do `data/sources.csv` com os áudios em disco; e no `pipeline.py`, quatro linhas de casos raros.

## Como os testes funcionam sem o dataset

Os áudios do dataset não vão para o Git, então os testes usam acordes sintéticos (`tests/sintetico.py`): cada nota é uma senoide com dois harmônicos e um decaimento leve, como uma corda. Como se sabe quais notas estão no sinal, dá para conferir que o chroma de um C destaca C, E e G.

O export (`tests/unit/test_build_tflite.py`) monta o grafo com a chroma_stft dentro, com uma CNN pequena de pesos aleatórios, e confere que a feature dele bate com a do librosa (erro menor que 0,0001), que a conversão só usa ops nativas do TFLite e que o int8 não quantiza a janela nem a matriz de chroma.

O teste de treino (`tests/unit/test_train_model.py`) monta um dataset sintético com 4 acordes x 10 arquivos numa pasta temporária e treina o modelo de ponta a ponta, com 20 negativos sintéticos de cada tipo em vez de 370 (só para rodar em segundos). O código de produção não muda para o teste.

## Onde está cada teste

### Dados e divisão: 74

Cobre baixar, montar o `sources.csv`, cortar os takes e dividir treino, validação e teste.

| Arquivo | O que garante | Testes |
|---|---|---:|
| `tests/unit/test_baixar_dataset.py` | o download arruma as pastas e não baixa de novo (sem rede) | 8 |
| `tests/unit/test_montar_sources.py` | o `sources.csv` junta as gravações antigas e o V3 sem estragar as linhas | 4 |
| `tests/unit/test_dataset_split.py` | o split é por arquivo, sem vazamento, sempre igual, e as janelas têm 1,5 s | 8 |
| `tests/unit/test_dividir_treino.py` | a divisão das duas fontes: o `take02` e o Test do V3 ficam fora do treino | 9 |
| `tests/unit/test_cortar_takes.py` | o corte do take em um arquivo por batida, e o que ele descarta | 29 |
| `tests/unit/test_exportar_ableton.py` | a exportação dos takes de um projeto do Ableton | 16 |

### Features, treino e limiar: 94

Cobre a chroma, o scaler, a classe `_none`, as CNNs, o limiar e o treino de ponta a ponta.

| Arquivo | O que garante | Testes |
|---|---|---:|
| `tests/unit/test_features_scaler.py` | os extratores de feature, o scaler e o one-hot dos rótulos | 35 |
| `tests/unit/test_chroma_stft.py` | a cadeia explícita da chroma é igual à do librosa; se uma versão nova do librosa mudar, quebra aqui | 8 |
| `tests/unit/test_negativos.py` | os áudios da classe `_none` são sempre os mesmos e passam do limiar de silêncio | 13 |
| `tests/unit/test_limiar_e_modelo.py` | o limiar nunca aumenta cobertura e falso aceite juntos, as CNNs e o salvar e carregar do modelo | 14 |
| `tests/unit/test_model_config.py` | o `model_config.json` e a leitura de pastas antigas, sem ele | 6 |
| `tests/unit/test_unknown_chord.py` | `_none` e confiança baixa viram "desconhecido" | 9 |
| `tests/unit/test_train_model.py` | o treino ponta a ponta num dataset sintético, e os casos de erro | 9 |

### Exportação para TFLite: 17

Cobre o grafo com a chroma dentro, a equivalência com o librosa e o `.tflite` do app.

| Arquivo | O que garante | Testes |
|---|---|---:|
| `tests/unit/test_build_tflite.py` | o grafo da exportação tem a mesma chroma do librosa, só usa operações nativas e o int8 não quantiza a janela nem a matriz | 12 |
| `tests/unit/test_export_tflite.py` | o `.tflite` de `export/` dá as probabilidades do `example_expected.json` e o `model_info.json` tem o hash certo | 5 |

### Comparação de modelos: 33

Cobre as métricas dos experimentos e o fluxo do `comparar_modelos.py`.

| Arquivo | O que garante | Testes |
|---|---|---:|
| `tests/unit/test_comparar_modelos.py` | as métricas do comparativo seguem a regra do app (silêncio, `_none` e limiar) | 19 |
| `tests/unit/test_comparar_modelos_fluxo.py` | o `comparar_modelos.py` de ponta a ponta, com cache de features, ensemble e `comparativo.md` | 14 |

### API Flask: 6

Cobre o `POST /predict_chord`.

| Arquivo | O que garante | Testes |
|---|---|---:|
| `tests/integration/test_api.py` | a API devolve `ok`, `silence`, 400 e 503 certos, com o modelo do `trained_model/` | 6 |

| Total | | 224 |
