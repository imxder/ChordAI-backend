# Comparativo: modelo atual × treino com as duas fontes

Gerado por `python -m scripts.comparar_modelos` a partir dos `metrics.json` desta pasta.
Média das seeds, com o menor e o maior entre parênteses. O modelo 0 é um só (não tem seed).

| Modelo | Seeds | Teste v3, 8 acordes | take02, 8 acordes | take02, extras (D, A, E) | Falso aceite, gravado | Falso aceite, sintético | Limiar |
|---|---|---|---|---|---|---|---|
| 0 | - | 90.7% | 99.2% | - | 87.5% | 2.7% | 0.5 |
| R | 42, 43, 44 | 90.4% (90.2–90.7) | 98.6% (97.3–99.2) | - | 75.7% (68.8–87.5) | 2.4% (1.9–2.7) | 0.3, 0.5, 0.55 |
| A | 42, 43, 44 | 76.2% (72.8–78.8) | 99.7% (99.2–100.0) | - | 18.1% (10.4–29.2) | 4.2% (2.7–5.0) | 0.3 |
| B8 | 42, 43, 44 | 89.3% (88.6–90.1) | 99.5% (98.8–100.0) | - | 26.4% (22.9–31.2) | 1.5% (0.8–2.3) | 0.5, 0.75 |
| B8v | 42, 43, 44 | 90.3% (89.2–91.0) | 100.0% (100.0–100.0) | - | 30.6% (27.1–33.3) | 0.9% (0.8–1.2) | 0.35, 0.45, 0.55 |
| B11 | 42, 43, 44 | 88.3% (87.7–88.6) | 98.6% (97.7–100.0) | 98.6% (96.9–100.0) | 35.4% (29.2–39.6) | 1.3% (1.2–1.5) | 0.6, 0.65, 0.7 |
| B11v | 42, 43, 44 | 89.2% (88.7–90.3) | 99.0% (98.0–100.0) | 96.9% (92.7–100.0) | 36.8% (33.3–41.7) | 0.9% (0.4–1.2) | 0.3 |
| B11vs | 42, 43, 44 | 90.3% (90.1–90.5) | 99.9% (99.6–100.0) | 93.1% (91.7–94.8) | 39.6% (37.5–41.7) | 1.0% (0.8–1.2) | 0.3 |
| F | 42, 43, 44 | 89.4% (89.2–89.5) | - | - | 36.1% (35.4–37.5) | 0.1% (0.0–0.4) | 0.3, 0.55 |
| Fa | 42, 43, 44 | 89.2% (89.0–89.4) | - | - | 18.8% (14.6–27.1) | 0.4% (0.0–0.9) | 0.3, 0.35 |

Tamanho dos testes (janelas de 1,5 s):
- Teste v3: 1508 janelas dos 8 acordes
- take02: 256 janelas dos 8 acordes e 96 dos extras
- Sem acorde: gravado_abafadas 24, gravado_afinando 24, low_volume 112, noise 74, percussion 74

## B11: confusão entre acordes que diferem por uma nota (take02)

| Seed | A->Am | Am->A | E->Em | Em->E | D->Dm | Dm->D |
|---|---|---|---|---|---|---|
| 42 | 0 de 32 | 0 de 32 | 0 de 32 | 0 de 32 | 0 de 32 | 5 de 32 |
| 43 | 0 de 32 | 0 de 32 | 1 de 32 | 0 de 32 | 2 de 32 | 0 de 32 |
| 44 | 0 de 32 | 0 de 32 | 1 de 32 | 0 de 32 | 0 de 32 | 5 de 32 |

## Como ler

- Acurácia: argmax em todas as janelas, sem limiar (a mesma conta do `accuracy_chords` do `train_model.py`).
- Falso aceite: fração das janelas sem acorde que o app mostraria como acorde (passa do filtro de silêncio, a classe não é `_none` e a confiança chega no limiar do modelo).
- Os take02 são do mesmo dia, cômodo e microfone do treino: o número é otimista e serve para comparar modelos, não para prometer acerto.
- O modelo 0 e o A nunca viram o Test do v3; o 0 também nunca viu a s1 nem a g1.
- O B11 muda duas coisas em relação ao B8 (os acordes extras e o peso por classe). Se ele perder acurácia, não dá para dizer qual das duas causou.
