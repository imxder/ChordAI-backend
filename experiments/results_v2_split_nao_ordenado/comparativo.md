# Comparativo de experimentos — ChordAI

Média ± desvio padrão (amostral) sobre as seeds 42, 43, 44. Teste: o mesmo do baseline (20% dos arquivos, split por arquivo, `random_state=42`, 1557 janelas de 1,5 s), usado **só** na avaliação final. A validação dos callbacks (EarlyStopping/ReduceLROnPlateau) usa 15% dos arquivos de treino (estratificado, `random_state=42`). "Acc. degradado" = mesmo teste com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz. ms/janela = média de `model.predict` (batch 1), sem extração de features. Nomes: `_scalerfeat` = StandardScaler por feature (o do baseline normaliza por frame); `aug2` = 2 versões aumentadas por janela de treino; `_detune` = +pitch shift de ±40 cents.

| Experimento | Seeds | Accuracy (%) | Macro-F1 (%) | Acc. degradado (%) | Parâmetros | ms/janela |
|---|---:|---:|---:|---:|---:|---:|
| aug2_chroma | 3 | 96.77 ± 0.42 | 96.74 ± 0.41 | 95.85 ± 0.52 | 259.144 | 117.4 |
| feat_chroma_scalerfeat | 3 | 96.32 ± 0.21 | 96.31 ± 0.22 | 95.74 ± 0.35 | 259.144 | 113.5 |
| feat_chroma | 3 | 96.30 ± 0.54 | 96.30 ± 0.53 | 95.78 ± 0.50 | 259.144 | 113.3 |
| feat_cqt_scalerfeat | 3 | 96.21 ± 0.55 | 96.20 ± 0.54 | 94.13 ± 0.96 | 282.184 | 118.2 |
| aug2_chroma_detune | 3 | 96.19 ± 0.10 | 96.19 ± 0.06 | 95.40 ± 0.45 | 259.144 | 117.2 |
| feat_mfcc | 3 | 88.42 ± 1.32 | 88.47 ± 1.27 | 72.36 ± 0.90 | 268.104 | 110.7 |
| feat_mfcc_scalerfeat | 3 | 87.00 ± 0.65 | 86.97 ± 0.67 | 71.53 ± 1.55 | 268.104 | 111.3 |

## Efeitos isolados (diferença pareada por seed, em pontos percentuais)

| Efeito (B − A) | A | B | Δ Accuracy (p.p.) | Δ Macro-F1 (p.p.) | Δ Acc. degradado (p.p.) |
|---|---|---|---:|---:|---:|
| Scaler por feature (MFCC) | feat_mfcc | feat_mfcc_scalerfeat | -1.41 ± 0.83 | -1.50 ± 0.80 | -0.83 ± 1.07 |
| Scaler por feature (chroma) | feat_chroma | feat_chroma_scalerfeat | +0.02 ± 0.74 | +0.01 ± 0.74 | -0.04 ± 0.67 |
| Chroma no lugar de MFCC (scaler do baseline) | feat_mfcc | feat_chroma | +7.88 ± 1.43 | +7.83 ± 1.40 | +23.42 ± 0.87 |
| Chroma no lugar de MFCC (scaler por feature) | feat_mfcc_scalerfeat | feat_chroma_scalerfeat | +9.31 ± 0.79 | +9.34 ± 0.83 | +24.21 ± 1.80 |
| CQT no lugar de chroma (scaler por feature) | feat_chroma_scalerfeat | feat_cqt_scalerfeat | -0.11 ± 0.74 | -0.11 ± 0.74 | -1.61 ± 0.85 |
| Augmentation (aug2, chroma) | feat_chroma | aug2_chroma | +0.47 ± 0.94 | +0.44 ± 0.91 | +0.06 ± 1.01 |
| Augmentation + detune (aug2, chroma) | feat_chroma | aug2_chroma_detune | -0.11 ± 0.46 | -0.11 ± 0.47 | -0.39 ± 0.95 |
| Detune (sobre aug2) | aug2_chroma | aug2_chroma_detune | -0.58 ± 0.51 | -0.55 ± 0.47 | -0.45 ± 0.06 |
| Total: baseline → melhor chroma sem aug | feat_mfcc | feat_chroma_scalerfeat | +7.90 ± 1.35 | +7.84 ± 1.31 | +23.38 ± 1.25 |

## F1 por classe (%, média das seeds)

| Experimento | Am | Bb | Bdim | C | Dm | Em | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aug2_chroma | 97.6 | 92.3 | 97.6 | 97.1 | 95.8 | 97.3 | 98.5 | 97.8 |
| feat_chroma_scalerfeat | 96.9 | 92.1 | 97.2 | 95.7 | 96.3 | 96.5 | 98.4 | 97.3 |
| feat_chroma | 96.2 | 94.1 | 96.3 | 96.5 | 95.1 | 96.4 | 98.2 | 97.6 |
| feat_cqt_scalerfeat | 96.2 | 92.4 | 94.6 | 97.3 | 97.1 | 96.8 | 97.9 | 97.4 |
| aug2_chroma_detune | 94.6 | 93.7 | 97.3 | 96.0 | 94.9 | 96.6 | 98.2 | 98.1 |
| feat_mfcc | 90.4 | 86.5 | 89.1 | 89.7 | 87.1 | 84.6 | 93.3 | 86.9 |
| feat_mfcc_scalerfeat | 90.9 | 80.9 | 86.0 | 87.3 | 88.8 | 84.9 | 90.9 | 86.2 |

## Pares mais confundidos (média de erros A→B + B→A por seed, janelas de teste)

Os 8 pares mais confundidos pelo `feat_mfcc`. Valores menores = melhor.

| Par | feat_mfcc | aug2_chroma | feat_chroma_scalerfeat | feat_chroma | feat_cqt_scalerfeat | aug2_chroma_detune | feat_mfcc_scalerfeat |
|---|---:|---:|---:|---:|---:|---:|---:|
| Em × G | 21.7 | 1.7 | 1.7 | 1.7 | 1.3 | 0.3 | 19.0 |
| C × Em | 17.3 | 2.7 | 3.7 | 3.7 | 1.0 | 3.7 | 19.0 |
| Bb × Dm | 14.7 | 9.7 | 7.3 | 9.3 | 5.3 | 9.7 | 21.0 |
| Bb × Bdim | 13.7 | 3.0 | 3.7 | 1.3 | 7.3 | 1.7 | 16.7 |
| Bdim × Dm | 11.0 | 3.7 | 3.7 | 6.0 | 3.0 | 5.3 | 7.3 |
| Am × Bb | 8.3 | 2.7 | 3.0 | 1.3 | 3.0 | 3.0 | 7.7 |
| Dm × G | 7.7 | 0.0 | 0.3 | 0.0 | 0.7 | 0.7 | 3.7 |
| Am × C | 7.3 | 0.3 | 2.3 | 2.3 | 1.3 | 4.3 | 7.3 |

---

<sub>O modelo `candidates/candidate_91_81` (o mesmo de `trained_model/`) não aparece na tabela: ele dá 97,24% neste teste contra ~91,8% documentado, o que indica que foi treinado com outro split treino/teste (provavelmente porque `list_dataset_files()` não ordena o `os.listdir()`, cuja ordem varia entre sistemas) e viu parte destes arquivos no treino. Por isso não é comparável. Os resultados da primeira rodada (que usava o teste como validação, como `train_model.py`) estão em `experiments/results_v1_val_no_teste/`.</sub>
