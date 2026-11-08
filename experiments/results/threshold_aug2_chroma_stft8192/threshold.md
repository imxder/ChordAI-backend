# Limiar de confiança — ChordAI (`aug2_chroma_stft8192_scalerfeat_gapbn_none`)

- **Validação** (usada para ESCOLHER o limiar): 944 janelas de acorde (de 985; 41 com RMS < 0.003, para as quais o app já devolve "silence") + 133 negativas (low_volume: 45, noise: 44, percussion: 44).
- **Teste** (só REPORTADO, não participa da escolha): 1485 janelas de acorde (de 1545; 60 com RMS < 0.003, para as quais o app já devolve "silence") + 223 negativas (low_volume: 75, noise: 74, percussion: 74).
- Média ± desvio sobre as seeds 42, 43, 44. Valores em %.

"Útil" = janelas de acorde aceitas **e** certas (cobertura × acurácia). "Falso aceite" = negativas aceitas como algum acorde. Colunas "degr." = mesmo teste de acordes com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz.

**Limiar escolhido na validação: 0.5** — critério: maior cobertura com falso aceite <= 5% e acuracia nos aceitos >= 98% (modelo aug2_chroma_stft8192_scalerfeat_gapbn_none, media das seeds [42, 43, 44]).

| No limiar escolhido | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---|---:|---:|---:|---:|
| Validação (escolha) | 98.1 ± 0.4 | 98.5 ± 0.2 | 96.6 ± 0.4 | 0.0 ± 0.0 |
| Teste (reportado) | 98.7 ± 0.2 | 97.4 ± 0.6 | 96.0 ± 0.7 | 1.5 ± 0.9 |

No teste degradado, com o mesmo limiar: cobertura 98.2%, acurácia nos aceitos 96.9%.

**Limiar de cada seed** (escolhido na validação da própria seed, mesmo critério; é assim que o modelo de produção é calibrado): seed 42: 0.50, seed 43: 0.50, seed 44: 0.50. No teste, cada seed no seu limiar: cobertura 98.7 ± 0.2, acurácia nos aceitos 97.4 ± 0.6, falso aceite 1.5 ± 0.9.

![curva](threshold.png)

## Com classe _none (`aug2_chroma_stft8192_scalerfeat_gapbn_none`) — validação (escolha)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---:|---:|---:|---:|---:|
| 0.30 | 99.6 ± 0.2 | 97.1 ± 0.3 | 96.7 ± 0.4 | 0.0 ± 0.0 |
| 0.35 | 99.5 ± 0.3 | 97.2 ± 0.1 | 96.7 ± 0.4 | 0.0 ± 0.0 |
| 0.40 | 99.2 ± 0.5 | 97.4 ± 0.2 | 96.7 ± 0.4 | 0.0 ± 0.0 |
| 0.45 | 98.7 ± 0.4 | 97.9 ± 0.1 | 96.6 ± 0.4 | 0.0 ± 0.0 |
| 0.50 **←** | 98.1 ± 0.4 | 98.5 ± 0.2 | 96.6 ± 0.4 | 0.0 ± 0.0 |
| 0.55 | 97.4 ± 0.3 | 99.0 ± 0.5 | 96.4 ± 0.5 | 0.0 ± 0.0 |
| 0.60 | 97.1 ± 0.1 | 99.1 ± 0.5 | 96.2 ± 0.5 | 0.0 ± 0.0 |
| 0.65 | 96.6 ± 0.2 | 99.3 ± 0.3 | 96.0 ± 0.4 | 0.0 ± 0.0 |
| 0.70 | 96.3 ± 0.2 | 99.4 ± 0.2 | 95.8 ± 0.3 | 0.0 ± 0.0 |
| 0.75 | 95.9 ± 0.3 | 99.5 ± 0.2 | 95.5 ± 0.4 | 0.0 ± 0.0 |
| 0.80 | 95.3 ± 0.4 | 99.7 ± 0.0 | 95.0 ± 0.4 | 0.0 ± 0.0 |
| 0.85 | 94.7 ± 0.6 | 99.7 ± 0.1 | 94.4 ± 0.6 | 0.0 ± 0.0 |
| 0.90 | 94.1 ± 0.6 | 99.8 ± 0.0 | 93.9 ± 0.6 | 0.0 ± 0.0 |
| 0.95 | 93.2 ± 0.6 | 99.8 ± 0.1 | 93.1 ± 0.7 | 0.0 ± 0.0 |

## Com classe _none (`aug2_chroma_stft8192_scalerfeat_gapbn_none`) — teste (reportado)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite | Cobertura (degr.) | Acurácia (degr.) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 99.6 ± 0.1 | 96.8 ± 0.6 | 96.4 ± 0.6 | 1.8 ± 0.8 | 99.7 ± 0.1 | 96.1 ± 0.3 |
| 0.35 | 99.5 ± 0.2 | 96.9 ± 0.5 | 96.4 ± 0.7 | 1.8 ± 0.8 | 99.6 ± 0.1 | 96.2 ± 0.2 |
| 0.40 | 99.4 ± 0.1 | 97.0 ± 0.6 | 96.4 ± 0.6 | 1.8 ± 0.8 | 99.4 ± 0.0 | 96.3 ± 0.2 |
| 0.45 | 99.1 ± 0.3 | 97.1 ± 0.5 | 96.3 ± 0.7 | 1.6 ± 0.7 | 99.0 ± 0.3 | 96.6 ± 0.3 |
| 0.50 **←** | 98.7 ± 0.2 | 97.4 ± 0.6 | 96.0 ± 0.7 | 1.5 ± 0.9 | 98.2 ± 0.2 | 96.9 ± 0.4 |
| 0.55 | 97.6 ± 0.4 | 97.9 ± 0.6 | 95.5 ± 0.6 | 0.9 ± 0.4 | 97.2 ± 0.3 | 97.3 ± 0.4 |
| 0.60 | 96.9 ± 0.5 | 98.2 ± 0.7 | 95.2 ± 0.5 | 0.9 ± 0.4 | 96.5 ± 0.4 | 97.6 ± 0.4 |
| 0.65 | 96.4 ± 0.6 | 98.4 ± 0.7 | 94.9 ± 0.5 | 0.9 ± 0.4 | 95.6 ± 0.5 | 98.0 ± 0.6 |
| 0.70 | 95.7 ± 0.5 | 98.7 ± 0.5 | 94.4 ± 0.3 | 0.7 ± 0.5 | 95.0 ± 0.4 | 98.2 ± 0.6 |
| 0.75 | 95.3 ± 0.4 | 98.8 ± 0.4 | 94.1 ± 0.3 | 0.7 ± 0.5 | 94.1 ± 0.2 | 98.5 ± 0.5 |
| 0.80 | 94.5 ± 0.6 | 99.0 ± 0.4 | 93.6 ± 0.5 | 0.6 ± 0.7 | 93.4 ± 0.3 | 98.7 ± 0.5 |
| 0.85 | 93.9 ± 0.8 | 99.1 ± 0.3 | 93.1 ± 0.7 | 0.6 ± 0.7 | 92.5 ± 0.3 | 98.9 ± 0.4 |
| 0.90 | 92.7 ± 0.5 | 99.3 ± 0.4 | 92.0 ± 0.4 | 0.3 ± 0.5 | 91.1 ± 0.2 | 99.2 ± 0.3 |
| 0.95 | 91.0 ± 0.9 | 99.4 ± 0.4 | 90.4 ± 0.7 | 0.3 ± 0.5 | 89.2 ± 0.5 | 99.4 ± 0.3 |
