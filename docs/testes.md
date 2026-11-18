# Testes e cobertura

O Playbook pede **75% de cobertura** no backend próprio de um projeto Mobile. Aqui o backend que roda de verdade é o pipeline de treino (`pipeline.py`, `train_model.py`) e a API (`api/app.py`).

## Números (18/11/2026)

Medido com `pytest-cov` e só o que o CI roda (sem o dataset em disco):

| Arquivo | Linhas | Cobertas | Cobertura |
|---|---|---|---|
| `api/app.py` | 35 | 29 | 83% |
| `pipeline.py` | 459 | 431 | 94% |
| `train_model.py` | 121 | 121 | 100% |
| **Total** | **615** | **581** | **94%** |

Antes desta rodada de testes (17/11) eram 312 de 634 linhas, **49,2%**.

O CI manda o `coverage.xml` pro SonarCloud em todo push, e o histórico fica lá: [sonarcloud.io/component_measures?id=imxder_ChordAI-backend&metric=coverage](https://sonarcloud.io/component_measures?id=imxder_ChordAI-backend&metric=coverage).

## Como gerar o relatório

```bash
python -m pytest -m "not dataset" --cov=. --cov-report=term-missing
coverage report --include="api/*,pipeline.py,train_model.py"
```

O primeiro roda os testes e mede; o segundo mostra só os três arquivos de produção. `--cov-report=html` gera uma página em `htmlcov/index.html` com as linhas não cobertas marcadas em vermelho.

## O que ficou fora da conta, e por quê

Está tudo escrito no [`.coveragerc`](../.coveragerc):

- **`api/serve.py`**: só joga a saída num arquivo de log e sobe o waitress na porta 5000. Testar seria subir um servidor de verdade no CI. A rota `/predict_chord`, que é o que importa, está em `api/app.py` e tem teste.
- **Os blocos `if __name__ == '__main__':`**: só rodam quando chamo o arquivo pela linha de comando (`python train_model.py`).

Sem essas duas exclusões o total seria 92% (583 de 634).

As linhas que continuam sem teste são as mensagens de inicialização da API (o `print` de "modelo carregado" quando o Flask sobe), o `_process_file` do `pipeline.py` (ele roda, mas dentro dos processos do joblib, que o coverage não acompanha) e alguns ramos de erro raros.

## Como os testes funcionam sem o dataset

Os áudios do dataset não vão pro Git, então os testes usam **acordes sintéticos** (`tests/sintetico.py`): cada nota é uma senoide com dois harmônicos e um decaimento leve, como uma corda. Como a gente sabe quais notas estão no sinal, dá pra conferir que o chroma de um C destaca C, E e G.

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
| `tests/integration/test_api.py` | `/predict_chord`: acorde, silêncio, 400 e 503 |
| os outros `tests/unit/test_*.py` | os scripts de `scripts/` (cortar takes, fontes, comparar modelos...) |
