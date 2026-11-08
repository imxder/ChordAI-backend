# Limiar de confiança — ChordAI (`feat_chroma_stft4096_scalerfeat_gapbn_none`)

- **Validação** (usada para ESCOLHER o limiar): 944 janelas de acorde (de 985; 41 com RMS < 0.003, para as quais o app já devolve "silence") + 133 negativas (low_volume: 45, noise: 44, percussion: 44).
- **Teste** (só REPORTADO, não participa da escolha): 1485 janelas de acorde (de 1545; 60 com RMS < 0.003, para as quais o app já devolve "silence") + 223 negativas (low_volume: 75, noise: 74, percussion: 74).
- Média ± desvio sobre as seeds 42, 43, 44. Valores em %.

"Útil" = janelas de acorde aceitas **e** certas (cobertura × acurácia). "Falso aceite" = negativas aceitas como algum acorde. Colunas "degr." = mesmo teste de acordes com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz.

**Limiar escolhido na validação: 0.45** — critério: maior cobertura com falso aceite <= 5% e acuracia nos aceitos >= 98% (modelo feat_chroma_stft4096_scalerfeat_gapbn_none, media das seeds [42, 43, 44]).

| No limiar escolhido | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---|---:|---:|---:|---:|
| Validação (escolha) | 98.4 ± 0.2 | 98.0 ± 0.2 | 96.5 ± 0.2 | 0.5 ± 0.4 |
| Teste (reportado) | 98.7 ± 0.5 | 96.2 ± 0.5 | 94.9 ± 0.9 | 0.6 ± 0.3 |

No teste degradado, com o mesmo limiar: cobertura 95.6%, acurácia nos aceitos 95.2%.

**Limiar de cada seed** (escolhido na validação da própria seed, mesmo critério; é assim que o modelo de produção é calibrado): seed 42: 0.40, seed 43: 0.50, seed 44: 0.45. No teste, cada seed no seu limiar: cobertura 98.4 ± 1.2, acurácia nos aceitos 96.4 ± 0.1, falso aceite 0.6 ± 0.3.

![curva](threshold.png)

## Com classe _none (`feat_chroma_stft4096_scalerfeat_gapbn_none`) — validação (escolha)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---:|---:|---:|---:|---:|
| 0.30 | 99.6 ± 0.1 | 97.3 ± 0.2 | 96.9 ± 0.2 | 0.5 ± 0.4 |
| 0.35 | 99.4 ± 0.3 | 97.4 ± 0.4 | 96.9 ± 0.2 | 0.5 ± 0.4 |
| 0.40 | 99.1 ± 0.6 | 97.5 ± 0.5 | 96.7 ± 0.2 | 0.5 ± 0.4 |
| 0.45 **←** | 98.4 ± 0.2 | 98.0 ± 0.2 | 96.5 ± 0.2 | 0.5 ± 0.4 |
| 0.50 | 97.8 ± 0.5 | 98.5 ± 0.3 | 96.3 ± 0.3 | 0.5 ± 0.4 |
| 0.55 | 97.2 ± 0.6 | 98.8 ± 0.3 | 96.0 ± 0.3 | 0.5 ± 0.4 |
| 0.60 | 96.6 ± 0.3 | 99.1 ± 0.2 | 95.7 ± 0.2 | 0.5 ± 0.4 |
| 0.65 | 96.2 ± 0.3 | 99.2 ± 0.2 | 95.5 ± 0.3 | 0.5 ± 0.4 |
| 0.70 | 95.9 ± 0.3 | 99.4 ± 0.1 | 95.3 ± 0.3 | 0.5 ± 0.4 |
| 0.75 | 95.3 ± 0.2 | 99.5 ± 0.2 | 94.9 ± 0.2 | 0.5 ± 0.4 |
| 0.80 | 94.9 ± 0.1 | 99.6 ± 0.2 | 94.5 ± 0.2 | 0.5 ± 0.4 |
| 0.85 | 94.4 ± 0.2 | 99.7 ± 0.2 | 94.1 ± 0.3 | 0.5 ± 0.4 |
| 0.90 | 93.4 ± 0.1 | 99.7 ± 0.1 | 93.1 ± 0.1 | 0.5 ± 0.4 |
| 0.95 | 91.7 ± 0.5 | 99.8 ± 0.1 | 91.6 ± 0.5 | 0.5 ± 0.4 |

## Com classe _none (`feat_chroma_stft4096_scalerfeat_gapbn_none`) — teste (reportado)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite | Cobertura (degr.) | Acurácia (degr.) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 99.6 ± 0.0 | 95.6 ± 0.8 | 95.2 ± 0.8 | 0.6 ± 0.3 | 96.5 ± 0.6 | 94.6 ± 0.9 |
| 0.35 | 99.5 ± 0.1 | 95.7 ± 0.8 | 95.2 ± 0.8 | 0.6 ± 0.3 | 96.4 ± 0.6 | 94.6 ± 0.9 |
| 0.40 | 99.2 ± 0.2 | 95.8 ± 0.8 | 95.0 ± 0.8 | 0.6 ± 0.3 | 96.1 ± 0.5 | 94.8 ± 0.7 |
| 0.45 **←** | 98.7 ± 0.5 | 96.2 ± 0.5 | 94.9 ± 0.9 | 0.6 ± 0.3 | 95.6 ± 0.5 | 95.2 ± 0.6 |
| 0.50 | 98.1 ± 0.9 | 96.6 ± 0.0 | 94.7 ± 0.9 | 0.6 ± 0.3 | 95.0 ± 0.5 | 95.6 ± 0.4 |
| 0.55 | 97.5 ± 1.1 | 96.9 ± 0.2 | 94.5 ± 1.0 | 0.4 ± 0.4 | 94.0 ± 0.6 | 96.1 ± 0.2 |
| 0.60 | 96.8 ± 0.9 | 97.2 ± 0.1 | 94.1 ± 0.8 | 0.4 ± 0.4 | 93.1 ± 0.5 | 96.3 ± 0.2 |
| 0.65 | 96.2 ± 1.0 | 97.4 ± 0.2 | 93.7 ± 0.8 | 0.4 ± 0.4 | 92.3 ± 0.5 | 96.6 ± 0.2 |
| 0.70 | 95.6 ± 1.1 | 97.7 ± 0.2 | 93.4 ± 0.9 | 0.3 ± 0.5 | 91.4 ± 0.5 | 97.0 ± 0.3 |
| 0.75 | 94.8 ± 1.1 | 97.8 ± 0.2 | 92.8 ± 0.9 | 0.3 ± 0.5 | 90.3 ± 0.5 | 97.2 ± 0.2 |
| 0.80 | 94.1 ± 1.2 | 98.1 ± 0.3 | 92.3 ± 0.8 | 0.3 ± 0.5 | 89.1 ± 0.3 | 97.7 ± 0.2 |
| 0.85 | 93.1 ± 1.2 | 98.3 ± 0.2 | 91.6 ± 1.1 | 0.3 ± 0.5 | 87.8 ± 0.5 | 98.0 ± 0.3 |
| 0.90 | 91.9 ± 1.3 | 98.4 ± 0.2 | 90.4 ± 1.2 | 0.3 ± 0.5 | 85.7 ± 0.5 | 98.4 ± 0.1 |
| 0.95 | 89.7 ± 1.5 | 98.8 ± 0.2 | 88.6 ± 1.3 | 0.3 ± 0.5 | 83.2 ± 0.9 | 98.7 ± 0.1 |
