# Modelo para o celular (TFLite com a feature embutida)

O app manda **áudio bruto** para o modelo e recebe as probabilidades dos acordes. A extração de feature (chroma da STFT) está dentro do `.tflite`, então o app não precisa implementar nenhuma feature em Dart.

| Arquivo | O que é |
|---|---|
| `chord_model.tflite` | **Modelo recomendado para o app** (float32). |
| `chord_model_int8.tflite` | Alternativa: mesmo modelo com quantização dinâmica (pesos da CNN em int8). |
| `labels.json` | Ordem exata das classes na saída, incluindo `"_none"`. |
| `model_info.json` | Taxa de amostragem, tamanho da entrada, limiar de confiança, limiar de silêncio e versão. |
| `model_config.json` | Configuração completa do modelo de origem (formato do backend). |
| `example.wav` e `example_expected.json` | Uma janela de teste e as probabilidades que o app deve obter com ela. |
| `build_tflite.py` | Script que gera tudo isto e roda os testes de equivalência. |
| `export_report.json` | Números da última geração: equivalência, ops, tamanhos e métricas. |

O modelo é o C1 da Parte 7a: chroma_stft com n_fft 8192, tuning 0, augmentation aug2, CNN GAP+BN e classe `_none`.

A seed (42, 43 ou 44) é escolhida pela maior acurácia nos acordes na **validação**, nunca pelo teste. A escolhida foi a **44**, e as métricas de validação das três estão em `export_report.json` (`seed_selection`).

## Qual arquivo usar

Use o **`chord_model.tflite` (float32)**:
- **O int8 economiza pouco.** São 289 KB contra 419 KB, porque a janela e a matriz de chroma continuam em float32 de propósito.
- **A latência é a mesma** nas duas versões.
- **O int8 depende de um contorno na exportação.** O conversor não tem opção para deixar a matriz de chroma fora da quantização int8. Para protegê-la, o grafo a multiplica por um 1,0 que depende da entrada (`sign(|x| + 1)`). Isso funciona e é verificado a cada geração, mas é um detalhe frágil, que uma versão nova do conversor pode mudar.

O `chord_model_int8.tflite` continua sendo gerado como alternativa. No teste, ele perde 0,06 p.p. em relação ao float32. Os números estão em `export_report.json` e na seção "Parte 7a" de `experiments/results/comparativo.md`.

## Como o app deve chamar o modelo

1. **Áudio.** Grave ou reamostre o sinal para **22 050 Hz, mono**. Use os valores de `sample_rate` e `input_samples` do `model_info.json`.
2. **PCM16 → float.** Converta cada amostra com `float = int16 / 32768.0`. O resultado fica entre −1 e 1.
3. **Janela.** Pegue **33 075 amostras**, que equivalem a 1,5 s.
4. **Silêncio.** Calcule o RMS: `sqrt(média(x²))`. Se ficar abaixo de `silence_rms_threshold` (0,003), mostre "silêncio" e não chame o modelo.
5. **Inferência.** Passe a entrada `float32 [1, 33075]` e leia a saída `float32 [1, n_classes]`, que já é o softmax. As classes estão na ordem de `labels.json`.
6. **Decisão.** Pegue a classe de maior probabilidade.
   - Se for `"_none"`, ou se a probabilidade for menor que `confidence_threshold` do `model_info.json`, trate como **desconhecido**.
   - Senão, mostre o acorde com essa confiança.

Essa é a mesma regra da API: `status` `ok`, `unknown` ou `silence`.

## Conferir a integração

Leia `example.wav` (PCM16, 22 050 Hz, 33 075 amostras), faça os passos 2 e 5 e compare com `example_expected.json`. As probabilidades devem bater com erro de ~1e-5 no float32, e a classe e a confiança devem ser as mesmas.

## Regerar

```bash
python -m experiments.run_seeds --only aug2_chroma_stft8192_scalerfeat_gapbn_none   # se o .h5 não existir
python -m experiments.threshold --feature aug2_chroma_stft8192
python export/build_tflite.py
```

Todos os parâmetros da feature vêm das constantes `CHROMA_STFT_*` e de `chroma_stft_params()` em `pipeline.py`. O `requirements.lock.txt` fixa o librosa (1.0.0) e o TensorFlow (2.21.0) usados.
