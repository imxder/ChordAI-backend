# Comparativo de experimentos — ChordAI

Média ± desvio padrão (amostral) sobre as seeds 42, 43, 44. Teste: o mesmo do baseline (20% dos arquivos, split por arquivo, `random_state=42`, 1545 janelas de 1,5 s), usado **só** na avaliação final. A validação dos callbacks (EarlyStopping/ReduceLROnPlateau) usa 15% dos arquivos de treino (estratificado, `random_state=42`). "Acc. degradado" = mesmo teste com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz. ms/janela = média de `model.predict` (batch 1), sem extração de features. Nomes: `_scalerfeat` = StandardScaler por feature (o do baseline normaliza por frame); `aug2` = 2 versões aumentadas por janela de treino; `_detune` = +pitch shift de ±40 cents.

| Experimento | Seeds | Accuracy (%) | Macro-F1 (%) | Acc. degradado (%) | Parâmetros | ms/janela |
|---|---:|---:|---:|---:|---:|---:|
| feat_chroma_scalerfeat | 3 | 96.03 ± 0.63 | 96.00 ± 0.65 | 94.80 ± 0.93 | 259.144 | 115.6 |
| aug2_chroma | 3 | 95.77 ± 0.73 | 95.76 ± 0.74 | 94.71 ± 1.18 | 259.144 | 114.7 |
| feat_chroma | 3 | 95.75 ± 0.54 | 95.73 ± 0.57 | 95.12 ± 0.42 | 259.144 | 116.7 |
| feat_mfcc | 3 | 88.00 ± 0.94 | 88.00 ± 0.93 | 76.12 ± 0.73 | 268.104 | 116.0 |

## Efeitos isolados (diferença pareada por seed, em pontos percentuais)

| Efeito (B − A) | A | B | Δ Accuracy (p.p.) | Δ Macro-F1 (p.p.) | Δ Acc. degradado (p.p.) |
|---|---|---|---:|---:|---:|
| Scaler por feature (chroma) | feat_chroma | feat_chroma_scalerfeat | +0.28 ± 0.77 | +0.26 ± 0.80 | -0.32 ± 0.62 |
| Chroma no lugar de MFCC (scaler do baseline) | feat_mfcc | feat_chroma | +7.75 ± 0.78 | +7.73 ± 0.76 | +19.01 ± 1.03 |
| Augmentation (aug2, chroma) | feat_chroma | aug2_chroma | +0.02 ± 0.97 | +0.03 ± 1.00 | -0.41 ± 1.37 |
| Total: baseline → melhor chroma sem aug | feat_mfcc | feat_chroma_scalerfeat | +8.03 ± 0.47 | +7.99 ± 0.46 | +18.68 ± 1.63 |

## F1 por classe (%, média das seeds)

| Experimento | Am | Bb | Bdim | C | Dm | Em | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| feat_chroma_scalerfeat | 97.0 | 94.7 | 92.3 | 97.0 | 96.5 | 97.5 | 96.3 | 96.7 |
| aug2_chroma | 94.8 | 95.6 | 93.4 | 95.4 | 95.6 | 97.9 | 97.1 | 96.4 |
| feat_chroma | 95.3 | 95.0 | 92.7 | 97.6 | 95.0 | 97.3 | 96.6 | 96.4 |
| feat_mfcc | 89.3 | 87.1 | 86.9 | 87.2 | 88.3 | 89.2 | 88.2 | 87.7 |

## Pares mais confundidos (média de erros A→B + B→A por seed, janelas de teste)

Os 8 pares mais confundidos pelo `feat_mfcc`. Valores menores = melhor.

| Par | feat_mfcc | feat_chroma_scalerfeat | aug2_chroma | feat_chroma |
|---|---:|---:|---:|---:|
| C × Em | 15.3 | 2.0 | 1.7 | 1.0 |
| Bb × Bdim | 15.0 | 6.0 | 4.3 | 4.0 |
| Bdim × F | 13.3 | 9.3 | 7.0 | 8.3 |
| Dm × G | 12.3 | 3.7 | 6.7 | 6.3 |
| Em × F | 11.0 | 0.0 | 0.0 | 0.0 |
| Am × Bb | 10.0 | 4.0 | 3.3 | 3.7 |
| Bb × Dm | 9.7 | 4.0 | 4.0 | 5.3 |
| C × G | 9.3 | 2.3 | 3.0 | 1.7 |

---

<sub>Split: `list_dataset_files()` agora ordena os arquivos (`sorted(os.listdir())`), então o teste é o mesmo em qualquer sistema operacional (SHA-256 da lista em `split_hash.txt`; conferir com `python -m experiments.split_hash --check`). O modelo `candidates/candidate_91_81` (o mesmo de `trained_model/`) não aparece na tabela: foi treinado com um split antigo, dependente da ordem do `os.listdir()`, e não é comparável. Rodadas anteriores: `experiments/results_v1_val_no_teste/` (teste usado como validação) e `experiments/results_v2_split_nao_ordenado/` (validação separada, mas split com a ordem do NTFS, que compartilha só 59 dos 297 arquivos de teste com o atual).</sub>
