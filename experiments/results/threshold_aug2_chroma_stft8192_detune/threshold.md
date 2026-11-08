# Limiar de confiança — ChordAI (`aug2_chroma_stft8192_scalerfeat_detune_gapbn_none`)

- **Validação** (usada para ESCOLHER o limiar): 944 janelas de acorde (de 985; 41 com RMS < 0.003, para as quais o app já devolve "silence") + 133 negativas (low_volume: 45, noise: 44, percussion: 44).
- **Teste** (só REPORTADO, não participa da escolha): 1485 janelas de acorde (de 1545; 60 com RMS < 0.003, para as quais o app já devolve "silence") + 223 negativas (low_volume: 75, noise: 74, percussion: 74).
- Média ± desvio sobre as seeds 42, 43, 44. Valores em %.

"Útil" = janelas de acorde aceitas **e** certas (cobertura × acurácia). "Falso aceite" = negativas aceitas como algum acorde. Colunas "degr." = mesmo teste de acordes com ruído rosa a 15 dB + passa-banda 200 Hz–5 kHz.

**Limiar escolhido na validação: 0.65** — critério: maior cobertura com falso aceite <= 5% e acuracia nos aceitos >= 98% (modelo aug2_chroma_stft8192_scalerfeat_detune_gapbn_none, media das seeds [42, 43, 44]).

| No limiar escolhido | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---|---:|---:|---:|---:|
| Validação (escolha) | 96.5 ± 0.3 | 98.1 ± 0.9 | 94.6 ± 1.2 | 0.3 ± 0.4 |
| Teste (reportado) | 96.4 ± 0.2 | 97.3 ± 0.5 | 93.8 ± 0.4 | 2.2 ± 1.2 |

No teste degradado, com o mesmo limiar: cobertura 95.4%, acurácia nos aceitos 96.9%.

**Limiar de cada seed** (escolhido na validação da própria seed, mesmo critério; é assim que o modelo de produção é calibrado): seed 42: 0.75, seed 43: 0.70, seed 44: 0.50. No teste, cada seed no seu limiar: cobertura 95.7 ± 2.2, acurácia nos aceitos 97.4 ± 0.6, falso aceite 2.1 ± 1.6.

![curva](threshold.png)

## Com classe _none (`aug2_chroma_stft8192_scalerfeat_detune_gapbn_none`) — validação (escolha)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite |
|---:|---:|---:|---:|---:|
| 0.30 | 99.5 ± 0.2 | 96.3 ± 0.6 | 95.8 ± 0.7 | 0.8 ± 0.8 |
| 0.35 | 99.5 ± 0.2 | 96.3 ± 0.6 | 95.8 ± 0.8 | 0.8 ± 0.8 |
| 0.40 | 99.4 ± 0.2 | 96.4 ± 0.7 | 95.8 ± 0.8 | 0.8 ± 0.8 |
| 0.45 | 98.9 ± 0.2 | 96.8 ± 1.0 | 95.7 ± 0.9 | 0.8 ± 0.8 |
| 0.50 | 98.7 ± 0.2 | 97.0 ± 1.0 | 95.7 ± 0.9 | 0.3 ± 0.4 |
| 0.55 | 98.0 ± 0.2 | 97.4 ± 1.0 | 95.4 ± 0.9 | 0.3 ± 0.4 |
| 0.60 | 97.3 ± 0.3 | 97.8 ± 1.1 | 95.2 ± 0.8 | 0.3 ± 0.4 |
| 0.65 **←** | 96.5 ± 0.3 | 98.1 ± 0.9 | 94.6 ± 1.2 | 0.3 ± 0.4 |
| 0.70 | 95.2 ± 1.2 | 98.7 ± 0.8 | 94.0 ± 1.4 | 0.3 ± 0.4 |
| 0.75 | 94.2 ± 1.2 | 98.9 ± 0.5 | 93.2 ± 1.5 | 0.3 ± 0.4 |
| 0.80 | 93.0 ± 1.3 | 99.4 ± 0.2 | 92.4 ± 1.4 | 0.3 ± 0.4 |
| 0.85 | 91.8 ± 1.4 | 99.6 ± 0.1 | 91.4 ± 1.5 | 0.3 ± 0.4 |
| 0.90 | 90.4 ± 1.4 | 99.6 ± 0.0 | 90.0 ± 1.4 | 0.0 ± 0.0 |
| 0.95 | 88.4 ± 1.5 | 99.7 ± 0.1 | 88.1 ± 1.5 | 0.0 ± 0.0 |

## Com classe _none (`aug2_chroma_stft8192_scalerfeat_detune_gapbn_none`) — teste (reportado)

| Limiar | Cobertura | Acurácia nos aceitos | Útil | Falso aceite | Cobertura (degr.) | Acurácia (degr.) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 99.6 ± 0.0 | 95.9 ± 0.4 | 95.5 ± 0.4 | 3.1 ± 0.8 | 99.6 ± 0.1 | 95.1 ± 0.3 |
| 0.35 | 99.6 ± 0.1 | 95.9 ± 0.4 | 95.4 ± 0.4 | 3.1 ± 0.8 | 99.6 ± 0.1 | 95.2 ± 0.2 |
| 0.40 | 99.3 ± 0.0 | 95.9 ± 0.4 | 95.3 ± 0.4 | 3.1 ± 0.8 | 99.4 ± 0.1 | 95.2 ± 0.3 |
| 0.45 | 99.1 ± 0.3 | 96.0 ± 0.5 | 95.1 ± 0.2 | 3.1 ± 0.8 | 99.0 ± 0.3 | 95.4 ± 0.4 |
| 0.50 | 98.6 ± 0.3 | 96.3 ± 0.6 | 94.9 ± 0.4 | 3.1 ± 0.8 | 98.2 ± 0.2 | 95.8 ± 0.4 |
| 0.55 | 97.9 ± 0.4 | 96.6 ± 0.6 | 94.6 ± 0.3 | 2.5 ± 1.1 | 97.4 ± 0.4 | 96.1 ± 0.5 |
| 0.60 | 97.1 ± 0.4 | 96.9 ± 0.6 | 94.1 ± 0.5 | 2.4 ± 1.4 | 96.3 ± 0.3 | 96.5 ± 0.4 |
| 0.65 **←** | 96.4 ± 0.2 | 97.3 ± 0.5 | 93.8 ± 0.4 | 2.2 ± 1.2 | 95.4 ± 0.2 | 96.9 ± 0.5 |
| 0.70 | 95.3 ± 0.4 | 97.7 ± 0.5 | 93.1 ± 0.7 | 1.8 ± 1.2 | 94.4 ± 0.5 | 97.4 ± 0.5 |
| 0.75 | 94.1 ± 0.5 | 97.9 ± 0.5 | 92.2 ± 0.8 | 1.3 ± 0.9 | 93.4 ± 0.4 | 97.7 ± 0.5 |
| 0.80 | 93.2 ± 0.9 | 98.1 ± 0.4 | 91.5 ± 1.0 | 1.2 ± 0.9 | 92.2 ± 1.0 | 98.0 ± 0.4 |
| 0.85 | 91.8 ± 1.1 | 98.6 ± 0.3 | 90.5 ± 1.1 | 1.2 ± 0.9 | 90.8 ± 1.4 | 98.4 ± 0.5 |
| 0.90 | 89.8 ± 1.3 | 98.8 ± 0.2 | 88.7 ± 1.3 | 1.2 ± 0.9 | 88.8 ± 1.9 | 98.6 ± 0.4 |
| 0.95 | 86.9 ± 2.2 | 99.0 ± 0.2 | 86.1 ± 2.0 | 1.0 ± 1.0 | 86.1 ± 2.3 | 99.0 ± 0.3 |
