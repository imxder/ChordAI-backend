# Limiar de confiança — ChordAI (`feat_chroma_cqt_t0_scalerfeat_gapbn_none`)

- **Validação** (usada para ESCOLHER o limiar): 944 janelas de acorde (de 985; 41 com RMS < 0.003, para as quais o app já devolve "silence") + 133 negativas (low_volume: 45, noise: 44, percussion: 44).
- **Teste** (só REPORTADO, não participa da escolha): 1485 janelas de acorde (de 1545; 60 com RMS < 0.003, para as quais o app já devolve "silence") + 223 negativas (low_volume: 75, noise: 74, percussion: 74).
- Média ± desvio sobre as seeds 42, 43, 44. Valores em %.

"Útil" = janelas de acorde aceitas **e** certas (cobertura × acurácia). "Falso aceite" = negativas aceitas como algum acorde. Colunas "degr." = mesmo teste de acordes com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz.

**Limiar escolhido na validação: 0.4** — critério: maior cobertura com falso aceite <= 5% e acuracia nos aceitos >= 98% (modelo feat_chroma_cqt_t0_scalerfeat_gapbn_none, media das seeds [42, 43, 44]).

| No limiar escolhido | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---|---:|---:|---:|---:|
| Validação (escolha) | 98.9 ± 0.6 | 98.1 ± 0.5 | 97.0 ± 0.1 | 1.0 ± 0.4 |
| Teste (reportado) | 98.9 ± 0.6 | 97.4 ± 0.3 | 96.3 ± 0.9 | 1.5 ± 0.3 |

No teste degradado, com o mesmo limiar: cobertura 97.5%, acurácia nos aceitos 96.3%.

**Limiar de cada seed** (escolhido na validação da própria seed, mesmo critério; é assim que o modelo de produção é calibrado): seed 42: 0.50, seed 43: 0.30, seed 44: 0.50. No teste, cada seed no seu limiar: cobertura 98.8 ± 0.5, acurácia nos aceitos 97.4 ± 0.7, falso aceite 1.6 ± 0.3.

![curva](threshold.png)

## Com classe _none (`feat_chroma_cqt_t0_scalerfeat_gapbn_none`) — validação (escolha)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---:|---:|---:|---:|---:|
| 0.30 | 99.4 ± 0.4 | 97.7 ± 0.3 | 97.1 ± 0.1 | 1.0 ± 0.4 |
| 0.35 | 99.2 ± 0.6 | 97.8 ± 0.5 | 97.0 ± 0.1 | 1.0 ± 0.4 |
| 0.40 **←** | 98.9 ± 0.6 | 98.1 ± 0.5 | 97.0 ± 0.1 | 1.0 ± 0.4 |
| 0.45 | 98.8 ± 0.8 | 98.2 ± 0.6 | 97.0 ± 0.2 | 1.0 ± 0.4 |
| 0.50 | 98.3 ± 0.6 | 98.5 ± 0.5 | 96.8 ± 0.1 | 1.0 ± 0.4 |
| 0.55 | 97.8 ± 0.6 | 98.7 ± 0.5 | 96.6 ± 0.3 | 0.8 ± 0.0 |
| 0.60 | 97.2 ± 1.0 | 99.1 ± 0.6 | 96.4 ± 0.4 | 0.8 ± 0.0 |
| 0.65 | 97.0 ± 1.0 | 99.2 ± 0.5 | 96.2 ± 0.4 | 0.8 ± 0.0 |
| 0.70 | 96.5 ± 0.5 | 99.5 ± 0.2 | 96.0 ± 0.3 | 0.8 ± 0.0 |
| 0.75 | 96.0 ± 0.5 | 99.6 ± 0.2 | 95.6 ± 0.3 | 0.8 ± 0.0 |
| 0.80 | 95.5 ± 0.3 | 99.6 ± 0.2 | 95.2 ± 0.1 | 0.5 ± 0.4 |
| 0.85 | 94.7 ± 0.2 | 99.7 ± 0.2 | 94.5 ± 0.3 | 0.5 ± 0.4 |
| 0.90 | 93.9 ± 0.3 | 99.8 ± 0.1 | 93.7 ± 0.4 | 0.3 ± 0.4 |
| 0.95 | 91.8 ± 0.2 | 99.8 ± 0.1 | 91.7 ± 0.2 | 0.3 ± 0.4 |

## Com classe _none (`feat_chroma_cqt_t0_scalerfeat_gapbn_none`) — teste (reportado)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite | Cobertura (degr.) | Acurácia (degr.) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 99.3 ± 0.3 | 97.2 ± 0.5 | 96.5 ± 0.8 | 1.8 ± 0.0 | 98.0 ± 0.4 | 96.0 ± 0.5 |
| 0.35 | 99.0 ± 0.6 | 97.3 ± 0.4 | 96.3 ± 0.9 | 1.8 ± 0.0 | 97.8 ± 0.6 | 96.1 ± 0.3 |
| 0.40 **←** | 98.9 ± 0.6 | 97.4 ± 0.3 | 96.3 ± 0.9 | 1.5 ± 0.3 | 97.5 ± 0.8 | 96.3 ± 0.2 |
| 0.45 | 98.6 ± 0.8 | 97.5 ± 0.3 | 96.1 ± 1.1 | 1.5 ± 0.3 | 97.2 ± 0.9 | 96.5 ± 0.2 |
| 0.50 | 98.2 ± 0.9 | 97.6 ± 0.3 | 95.9 ± 1.2 | 1.5 ± 0.3 | 96.8 ± 1.0 | 96.7 ± 0.2 |
| 0.55 | 97.7 ± 1.2 | 98.0 ± 0.2 | 95.7 ± 1.3 | 1.3 ± 0.0 | 96.0 ± 1.3 | 97.1 ± 0.1 |
| 0.60 | 96.9 ± 1.5 | 98.2 ± 0.1 | 95.2 ± 1.6 | 1.3 ± 0.0 | 95.3 ± 1.5 | 97.4 ± 0.3 |
| 0.65 | 96.4 ± 1.7 | 98.4 ± 0.2 | 94.8 ± 1.7 | 1.3 ± 0.0 | 94.5 ± 1.5 | 97.7 ± 0.3 |
| 0.70 | 95.8 ± 1.7 | 98.6 ± 0.3 | 94.4 ± 1.8 | 1.2 ± 0.3 | 93.7 ± 1.7 | 97.8 ± 0.2 |
| 0.75 | 94.9 ± 1.9 | 98.9 ± 0.4 | 93.8 ± 1.9 | 1.0 ± 0.3 | 92.7 ± 1.7 | 98.0 ± 0.2 |
| 0.80 | 94.0 ± 2.2 | 99.0 ± 0.3 | 93.0 ± 2.2 | 0.4 ± 0.4 | 91.8 ± 2.0 | 98.1 ± 0.1 |
| 0.85 | 92.9 ± 2.2 | 99.2 ± 0.2 | 92.1 ± 2.2 | 0.4 ± 0.4 | 90.8 ± 2.2 | 98.3 ± 0.1 |
| 0.90 | 91.3 ± 2.4 | 99.4 ± 0.1 | 90.8 ± 2.4 | 0.4 ± 0.4 | 89.3 ± 2.6 | 98.6 ± 0.2 |
| 0.95 | 88.9 ± 2.6 | 99.5 ± 0.1 | 88.5 ± 2.5 | 0.3 ± 0.3 | 86.3 ± 3.0 | 98.8 ± 0.2 |
