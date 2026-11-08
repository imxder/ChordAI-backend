# Limiar de confiança — ChordAI (`feat_chroma_stft_scalerfeat_gapbn_none`)

- **Validação** (usada para ESCOLHER o limiar): 944 janelas de acorde (de 985; 41 com RMS < 0.003, para as quais o app já devolve "silence") + 133 negativas (low_volume: 45, noise: 44, percussion: 44).
- **Teste** (só REPORTADO, não participa da escolha): 1485 janelas de acorde (de 1545; 60 com RMS < 0.003, para as quais o app já devolve "silence") + 223 negativas (low_volume: 75, noise: 74, percussion: 74).
- Média ± desvio sobre as seeds 42, 43, 44. Valores em %.

"Útil" = janelas de acorde aceitas **e** certas (cobertura × acurácia). "Falso aceite" = negativas aceitas como algum acorde. Colunas "degr." = mesmo teste de acordes com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz.

**Limiar escolhido na validação: 0.5** — critério: maior cobertura com falso aceite <= 5% e acuracia nos aceitos >= 98% (modelo feat_chroma_stft_scalerfeat_gapbn_none, media das seeds [42, 43, 44]).

| No limiar escolhido | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---|---:|---:|---:|---:|
| Validação (escolha) | 97.7 ± 0.5 | 98.2 ± 0.5 | 95.9 ± 0.5 | 0.5 ± 0.4 |
| Teste (reportado) | 97.9 ± 0.6 | 95.7 ± 1.2 | 93.6 ± 1.7 | 0.9 ± 0.4 |

No teste degradado, com o mesmo limiar: cobertura 95.0%, acurácia nos aceitos 94.0%.

**Limiar de cada seed** (escolhido na validação da própria seed, mesmo critério; é assim que o modelo de produção é calibrado): seed 42: 0.55, seed 43: 0.55, seed 44: 0.45. No teste, cada seed no seu limiar: cobertura 97.6 ± 1.1, acurácia nos aceitos 95.9 ± 1.0, falso aceite 0.9 ± 0.4.

![curva](threshold.png)

## Com classe _none (`feat_chroma_stft_scalerfeat_gapbn_none`) — validação (escolha)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---:|---:|---:|---:|---:|
| 0.30 | 99.6 ± 0.1 | 96.8 ± 0.3 | 96.4 ± 0.3 | 0.5 ± 0.4 |
| 0.35 | 99.4 ± 0.3 | 96.9 ± 0.1 | 96.3 ± 0.4 | 0.5 ± 0.4 |
| 0.40 | 98.9 ± 0.4 | 97.2 ± 0.4 | 96.2 ± 0.5 | 0.5 ± 0.4 |
| 0.45 | 98.4 ± 0.4 | 97.6 ± 0.6 | 96.1 ± 0.6 | 0.5 ± 0.4 |
| 0.50 **←** | 97.7 ± 0.5 | 98.2 ± 0.5 | 95.9 ± 0.5 | 0.5 ± 0.4 |
| 0.55 | 96.8 ± 0.7 | 98.6 ± 0.5 | 95.4 ± 0.7 | 0.3 ± 0.4 |
| 0.60 | 96.3 ± 0.7 | 98.8 ± 0.4 | 95.1 ± 0.6 | 0.3 ± 0.4 |
| 0.65 | 95.7 ± 0.6 | 99.0 ± 0.3 | 94.8 ± 0.6 | 0.3 ± 0.4 |
| 0.70 | 95.1 ± 0.5 | 99.3 ± 0.2 | 94.4 ± 0.8 | 0.0 ± 0.0 |
| 0.75 | 94.5 ± 1.0 | 99.4 ± 0.1 | 94.0 ± 1.0 | 0.0 ± 0.0 |
| 0.80 | 93.7 ± 0.9 | 99.6 ± 0.1 | 93.3 ± 0.9 | 0.0 ± 0.0 |
| 0.85 | 92.8 ± 1.4 | 99.7 ± 0.1 | 92.5 ± 1.5 | 0.0 ± 0.0 |
| 0.90 | 91.3 ± 2.3 | 99.8 ± 0.1 | 91.1 ± 2.2 | 0.0 ± 0.0 |
| 0.95 | 88.9 ± 3.1 | 99.9 ± 0.0 | 88.8 ± 3.1 | 0.0 ± 0.0 |

## Com classe _none (`feat_chroma_stft_scalerfeat_gapbn_none`) — teste (reportado)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite | Cobertura (degr.) | Acurácia (degr.) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 99.6 ± 0.1 | 94.7 ± 1.5 | 94.3 ± 1.4 | 1.0 ± 0.3 | 96.7 ± 0.7 | 93.2 ± 1.1 |
| 0.35 | 99.4 ± 0.0 | 94.8 ± 1.5 | 94.2 ± 1.5 | 1.0 ± 0.3 | 96.5 ± 0.7 | 93.3 ± 1.2 |
| 0.40 | 99.2 ± 0.3 | 95.0 ± 1.3 | 94.2 ± 1.5 | 1.0 ± 0.3 | 96.2 ± 0.6 | 93.5 ± 1.0 |
| 0.45 | 98.7 ± 0.3 | 95.2 ± 1.3 | 94.0 ± 1.5 | 1.0 ± 0.3 | 95.7 ± 0.7 | 93.7 ± 1.1 |
| 0.50 **←** | 97.9 ± 0.6 | 95.7 ± 1.2 | 93.6 ± 1.7 | 0.9 ± 0.4 | 95.0 ± 0.7 | 94.0 ± 1.1 |
| 0.55 | 97.1 ± 0.7 | 96.1 ± 1.1 | 93.4 ± 1.7 | 0.9 ± 0.4 | 93.5 ± 0.8 | 94.9 ± 1.0 |
| 0.60 | 96.3 ± 0.9 | 96.5 ± 0.9 | 93.0 ± 1.7 | 0.6 ± 0.5 | 92.5 ± 0.9 | 95.3 ± 0.8 |
| 0.65 | 95.5 ± 1.1 | 96.9 ± 0.7 | 92.6 ± 1.7 | 0.6 ± 0.5 | 91.2 ± 1.0 | 95.9 ± 0.7 |
| 0.70 | 94.5 ± 1.4 | 97.2 ± 0.4 | 91.9 ± 1.6 | 0.6 ± 0.5 | 89.9 ± 1.0 | 96.4 ± 0.4 |
| 0.75 | 93.8 ± 1.6 | 97.5 ± 0.4 | 91.4 ± 1.6 | 0.6 ± 0.5 | 88.6 ± 0.9 | 96.8 ± 0.5 |
| 0.80 | 92.7 ± 1.8 | 97.8 ± 0.3 | 90.6 ± 1.8 | 0.6 ± 0.5 | 87.3 ± 1.4 | 97.2 ± 0.3 |
| 0.85 | 91.6 ± 2.0 | 98.0 ± 0.3 | 89.8 ± 2.0 | 0.6 ± 0.5 | 85.5 ± 1.2 | 97.5 ± 0.2 |
| 0.90 | 90.0 ± 2.4 | 98.3 ± 0.2 | 88.4 ± 2.2 | 0.6 ± 0.5 | 83.4 ± 1.9 | 98.0 ± 0.2 |
| 0.95 | 87.7 ± 2.8 | 98.7 ± 0.4 | 86.5 ± 2.6 | 0.6 ± 0.5 | 80.3 ± 2.5 | 98.4 ± 0.3 |
