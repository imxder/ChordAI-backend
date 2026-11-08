# Comparativo de experimentos — ChordAI

Todos avaliados no mesmo conjunto de teste (split por arquivo, 80/20, `random_state=42`), em janelas de 1,5 s. Tempo = média de `model.predict` para uma janela (batch 1), sem extração de features. "Acc. degradado" = mesmo teste com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz (robustez a ruído/microfone).

| Experimento | Accuracy | Macro-F1 | Acc. degradado | Parâmetros | ms/janela |
|---|---:|---:|---:|---:|---:|
| ref_candidate_91_81 | 97.24% | 97.21% | — | 268.104 | 100.09 |
| aug1_chroma | 97.05% | 97.01% | 96.34% | 259.144 | 114.28 |
| aug1_chroma_detune | 97.05% | 97.05% | 96.21% | 259.144 | 112.21 |
| feat_chroma | 96.85% | 96.83% | 96.02% | 259.144 | 110.49 |
| feat_mfcc_chroma_scalerfeat | 96.85% | 96.81% | 94.09% | 271.944 | 111.23 |
| aug2_chroma | 96.66% | 96.64% | 96.66% | 259.144 | 112.39 |
| aug2_chroma_detune | 96.53% | 96.48% | 95.76% | 259.144 | 111.54 |
| feat_cqt | 96.47% | 96.46% | 94.35% | 282.184 | 111.18 |
| baseline_mfcc | 89.92% | 89.94% | 72.25% | 268.104 | 111.52 |
| feat_mfcc | 89.92% | 89.94% | — | 268.104 | 111.33 |
| feat_mfcc_chroma | 89.72% | 89.75% | 74.76% | 271.944 | 111.79 |

## F1 por classe (%)

| Experimento | Am | Bb | Bdim | C | Dm | Em | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ref_candidate_91_81 | 98.5 | 92.7 | 97.4 | 97.0 | 97.8 | 99.3 | 97.4 | 97.7 |
| aug1_chroma | 97.2 | 92.8 | 98.5 | 96.7 | 96.0 | 98.3 | 98.4 | 98.2 |
| aug1_chroma_detune | 97.5 | 96.9 | 94.8 | 96.7 | 95.8 | 97.8 | 98.4 | 98.5 |
| feat_chroma | 98.5 | 92.5 | 97.2 | 97.3 | 96.8 | 97.3 | 98.4 | 96.7 |
| feat_mfcc_chroma_scalerfeat | 98.7 | 91.9 | 97.7 | 97.7 | 95.2 | 97.6 | 97.9 | 97.7 |
| aug2_chroma | 96.7 | 92.8 | 97.4 | 96.2 | 97.6 | 97.9 | 97.9 | 96.7 |
| aug2_chroma_detune | 97.9 | 91.9 | 97.7 | 96.4 | 94.5 | 98.3 | 96.6 | 98.5 |
| feat_cqt | 94.7 | 93.1 | 96.1 | 97.7 | 97.3 | 97.3 | 98.7 | 96.7 |
| baseline_mfcc | 91.6 | 86.1 | 91.8 | 90.3 | 91.7 | 87.0 | 92.6 | 88.5 |
| feat_mfcc | 91.6 | 86.1 | 91.8 | 90.3 | 91.7 | 87.0 | 92.6 | 88.5 |
| feat_mfcc_chroma | 92.3 | 82.9 | 92.3 | 91.2 | 91.9 | 88.1 | 92.4 | 86.9 |

## Pares mais confundidos (erros A→B + B→A, janelas de teste)

Os 8 pares mais confundidos pelo `baseline_mfcc`. Valores menores = melhor.

| Par | baseline_mfcc | aug1_chroma | aug1_chroma_detune | feat_chroma | feat_mfcc_chroma_scalerfeat | aug2_chroma | aug2_chroma_detune | feat_cqt | feat_mfcc | feat_mfcc_chroma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C × Em | 19 | 2 | 2 | 2 | 4 | 4 | 3 | 0 | 19 | 9 |
| Em × G | 17 | 0 | 0 | 1 | 2 | 0 | 1 | 0 | 17 | 23 |
| Bb × Dm | 13 | 10 | 8 | 8 | 12 | 4 | 13 | 4 | 13 | 13 |
| Bb × Bdim | 10 | 2 | 0 | 2 | 2 | 2 | 2 | 6 | 10 | 11 |
| Bb × F | 10 | 5 | 1 | 5 | 6 | 6 | 5 | 4 | 10 | 13 |
| Bdim × G | 10 | 1 | 3 | 3 | 4 | 3 | 1 | 5 | 10 | 8 |
| Am × G | 8 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 8 | 5 |
| Bb × C | 7 | 5 | 1 | 5 | 5 | 5 | 5 | 5 | 7 | 7 |
