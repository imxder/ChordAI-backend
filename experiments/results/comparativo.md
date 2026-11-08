# Comparativo de experimentos — ChordAI

Média ± desvio padrão (amostral) sobre as seeds 42, 43, 44. Teste: o mesmo do baseline (20% dos arquivos, split por arquivo, `random_state=42`, 1545 janelas de 1,5 s), usado **só** na avaliação final. A validação dos callbacks (EarlyStopping/ReduceLROnPlateau) usa 15% dos arquivos de treino (estratificado, `random_state=42`). "Acc. degradado" = mesmo teste com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz. Latência por janela (batch 1, só o modelo, sem extração de features, CPU do PC, medida por `bench_latency.py` com todos os modelos nas mesmas condições): "ms predict" = `model.predict` (como `app.py` faz hoje, dominado pelo overhead do Keras); "ms compilado" = `tf.function(model)`; "ms TFLite" = interpretador TFLite com 1 thread (formato que iria para o celular). TFLite (KB) = modelo float32 sem quantização. Nomes: `_scalerfeat` = StandardScaler por feature (o do baseline normaliza por frame); `aug2` = 2 versões aumentadas por janela de treino; `_detune` = +pitch shift de ±40 cents; `_gapbn` = CNN 1D com BatchNorm + GlobalAveragePooling1D (variante A); `_cnn2d` = CNN 2D pequena (variante B).

| Experimento | Seeds | Accuracy (%) | Macro-F1 (%) | Acc. degradado (%) | Parâmetros | TFLite (KB) | ms predict | ms compilado | ms TFLite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| feat_chroma_scalerfeat | 3 | 96.03 ± 0.63 | 96.00 ± 0.65 | 94.80 ± 0.93 | 259.144 | 1019 | 108.4 | 2.24 | 0.08 |
| feat_chroma_scalerfeat_cnn2d | 3 | 95.77 ± 0.72 | 95.75 ± 0.72 | 95.56 ± 0.69 | 94.600 | 372 | 110.8 | 2.78 | 0.23 |
| aug2_chroma | 3 | 95.77 ± 0.73 | 95.76 ± 0.74 | 94.71 ± 1.18 | 259.144 | 1019 | 109.6 | 2.36 | 0.08 |
| feat_chroma | 3 | 95.75 ± 0.54 | 95.73 ± 0.57 | 95.12 ± 0.42 | 259.144 | 1019 | 110.3 | 2.00 | 0.08 |
| feat_chroma_scalerfeat_gapbn | 3 | 95.43 ± 0.36 | 95.40 ± 0.33 | 94.69 ± 0.51 | 46.920 | 188 | 109.2 | 2.06 | 0.05 |
| feat_mfcc | 3 | 88.00 ± 0.94 | 88.00 ± 0.93 | 76.12 ± 0.73 | 268.104 | 1054 | 108.2 | 2.17 | 0.11 |

## Efeitos isolados (diferença pareada por seed, em pontos percentuais)

| Efeito (B − A) | A | B | Δ Accuracy (p.p.) | Δ Macro-F1 (p.p.) | Δ Acc. degradado (p.p.) |
|---|---|---|---:|---:|---:|
| Scaler por feature (chroma) | feat_chroma | feat_chroma_scalerfeat | +0.28 ± 0.77 | +0.26 ± 0.80 | -0.32 ± 0.62 |
| Chroma no lugar de MFCC (scaler do baseline) | feat_mfcc | feat_chroma | +7.75 ± 0.78 | +7.73 ± 0.76 | +19.01 ± 1.03 |
| Augmentation (aug2, chroma) | feat_chroma | aug2_chroma | +0.02 ± 0.97 | +0.03 ± 1.00 | -0.41 ± 1.37 |
| Total: baseline → melhor chroma sem aug | feat_mfcc | feat_chroma_scalerfeat | +8.03 ± 0.47 | +7.99 ± 0.46 | +18.68 ± 1.63 |
| GAP+BN (A − feat_chroma_scalerfeat) | feat_chroma_scalerfeat | feat_chroma_scalerfeat_gapbn | -0.60 ± 0.49 | -0.60 ± 0.53 | -0.11 ± 0.45 |
| CNN 2D (B − feat_chroma_scalerfeat) | feat_chroma_scalerfeat | feat_chroma_scalerfeat_cnn2d | -0.26 ± 1.30 | -0.24 ± 1.33 | +0.76 ± 1.24 |
| CNN 2D − GAP+BN (B − A) | feat_chroma_scalerfeat_gapbn | feat_chroma_scalerfeat_cnn2d | +0.35 ± 0.84 | +0.36 ± 0.83 | +0.86 ± 1.01 |

## F1 por classe (%, média das seeds)

| Experimento | Am | Bb | Bdim | C | Dm | Em | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| feat_chroma_scalerfeat | 97.0 | 94.7 | 92.3 | 97.0 | 96.5 | 97.5 | 96.3 | 96.7 |
| feat_chroma_scalerfeat_cnn2d | 95.2 | 94.2 | 92.4 | 96.8 | 96.9 | 96.8 | 96.3 | 97.3 |
| aug2_chroma | 94.8 | 95.6 | 93.4 | 95.4 | 95.6 | 97.9 | 97.1 | 96.4 |
| feat_chroma | 95.3 | 95.0 | 92.7 | 97.6 | 95.0 | 97.3 | 96.6 | 96.4 |
| feat_chroma_scalerfeat_gapbn | 95.7 | 95.1 | 92.2 | 94.9 | 95.5 | 96.9 | 96.4 | 96.7 |
| feat_mfcc | 89.3 | 87.1 | 86.9 | 87.2 | 88.3 | 89.2 | 88.2 | 87.7 |

## Pares mais confundidos (média de erros A→B + B→A por seed, janelas de teste)

Os 8 pares mais confundidos pelo `feat_mfcc`. Valores menores = melhor.

| Par | feat_mfcc | feat_chroma_scalerfeat | feat_chroma_scalerfeat_cnn2d | aug2_chroma | feat_chroma | feat_chroma_scalerfeat_gapbn |
|---|---:|---:|---:|---:|---:|---:|
| C × Em | 15.3 | 2.0 | 0.0 | 1.7 | 1.0 | 1.7 |
| Bb × Bdim | 15.0 | 6.0 | 6.3 | 4.3 | 4.0 | 4.0 |
| Bdim × F | 13.3 | 9.3 | 8.3 | 7.0 | 8.3 | 9.0 |
| Dm × G | 12.3 | 3.7 | 2.7 | 6.7 | 6.3 | 3.3 |
| Em × F | 11.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Am × Bb | 10.0 | 4.0 | 4.7 | 3.3 | 3.7 | 3.7 |
| Bb × Dm | 9.7 | 4.0 | 3.3 | 4.0 | 5.3 | 4.7 |
| C × G | 9.3 | 2.3 | 1.0 | 3.0 | 1.7 | 3.0 |

## Parte 5 — classe "_none" (nenhum acorde)

Accuracy dos acordes = nas mesmas 1545 janelas de acorde dos outros experimentos (prever "_none" conta como erro). Recall _none = % das janelas negativas do teste classificadas como "_none". Sem limiar de confiança (ver `threshold/threshold.md`).

| Experimento | Seeds | Accuracy acordes (%) | Acordes → _none (%) | Recall _none (%) | Recall low_volume | Recall noise | Recall percussion | Acc. degradado (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| feat_chroma_scalerfeat_gapbn_none | 3 | 95.32 ± 0.26 | 0.91 ± 0.11 | 98.21 ± 0.00 | 95.11 ± 0.77 | 99.55 ± 0.78 | 100.00 ± 0.00 | 93.14 ± 0.45 |

Efeito da classe _none (`feat_chroma_scalerfeat_gapbn_none` − `feat_chroma_scalerfeat_gapbn`, pareado por seed): Δ accuracy acordes = -0.11 ± 0.10 p.p.; Δ acc. degradado = -1.55 ± 0.39 p.p.

Negativos por split: train: low_volume 252, noise 252, percussion 252; val: low_volume 45, noise 44, percussion 44; test: low_volume 75, noise 74, percussion 74

## Parte 6 — modelo em produção (`trained_model/`)

`chroma` + scaler `feature` + `gap_bn` + classe `_none`, pesos da seed 42 (`experiments/results/feat_chroma_scalerfeat_gapbn_none_s42`), **limiar de confiança 0.50**. O limiar foi calibrado na validação para este modelo (o mesmo valor que `train_model.py` calibra sozinho); o **0,45 da Parte 5 era a média das 3 seeds** (`threshold/threshold.md`). Medido pelo caminho de produção (`load_trained_model` + `recognize_chord`, áudio bruto) em `parte6/equivalence.json`; o caminho do experimento dá os mesmos números (diferença 0,00 p.p.).

| Janelas de acorde do teste | Janelas | % |
|---|---:|---:|
| Certas | 1444 | 93.46 |
| Erradas (outro acorde) | 26 | 1.68 |
| `unknown` (_none ou confiança < limiar) | 15 | 0.97 |
| `silence` (RMS < 0,003; o app trata à parte) | 60 | 3.88 |
| **Total** | 1545 | 100.00 |

Acurácia sem as janelas de silêncio: **97.24%** (1444/1485); nos aceitos: 98.23%; cobertura: 98.99%. Negativos do teste (223): 98.21% viram `unknown`, falso aceite 1.79%.

---

<sub>Split: `list_dataset_files()` agora ordena os arquivos (`sorted(os.listdir())`), então o teste é o mesmo em qualquer sistema operacional (SHA-256 da lista em `split_hash.txt`; conferir com `python -m experiments.split_hash --check`). O modelo `candidates/candidate_91_81` (o que ficava em `trained_model/` antes da Parte 6) não aparece na tabela: foi treinado com um split antigo, dependente da ordem do `os.listdir()`, e não é comparável. Rodadas anteriores: `experiments/results_v1_val_no_teste/` (teste usado como validação) e `experiments/results_v2_split_nao_ordenado/` (validação separada, mas split com a ordem do NTFS, que compartilha só 59 dos 297 arquivos de teste com o atual).</sub>
