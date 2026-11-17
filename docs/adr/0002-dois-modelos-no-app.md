# ADR 0002: o app traz dois modelos e o padrão é o v3

- **Status:** aceita
- **Data:** 15/11/2026
- **Repositórios:** [ChordAI-backend](https://github.com/imxder/ChordAI-backend) e [ChordAI](https://github.com/imxder/ChordAI)
- **Caminhos de arquivo:** os que começam com `lib/` ou `assets/` são do app. Os outros são do backend.

## Contexto

Depois de treinar com o meu dataset e o Guitar Chords V3 juntos ([`docs/experimentos.md`](https://github.com/imxder/ChordAI-backend/blob/main/docs/experimentos.md)), fiquei com dois modelos bons em coisas diferentes:

| | **v3** (C1, seed 44) | **v3_meu** (F9, seed 43) |
|---|---|---|
| Treino | Training do V3 + 42 gravações antigas minhas | Training do V3 + o meu dataset (s1 e g1) |
| Acordes | 8 | 9 (os 8 + o Lá maior) |
| Test do V3 (outras pessoas e violões) | **87,4%** | 85,7% |
| Falso aceite com sem acorde gravado (48 janelas) | 81,2% | **22,9%** |

Os números são de `experiments/results_duas_fontes/0c/metrics.json` (v3) e `experiments/results_duas_fontes/F9_stft8192_aug2_s43/metrics.json` (v3_meu). Na média das 3 seeds, o F9 dá 29,9% de falso aceite gravado (`experiments/results_duas_fontes/comparativo.md`).

## Decisão

O app traz os dois modelos, e eu escolho qual usar em **Perfil → Reconhecimento → Modelo no aparelho** ("Só v3" ou "v3 + meu dataset"). **O padrão é o v3.**

- **Por que dois:** pra eu poder comparar os dois no uso real.
- **Por que o v3 é o padrão:** ele teve a maior acurácia no teste independente, 87,4% na pasta Test do V3 contra 85,7% do v3_meu. Esse teste usa outras pessoas e outros violões, então é o que mais se parece com um usuário novo.
- **Por que o v3_meu fica como opção:** ele reconhece o Lá maior e erra bem menos com barulho gravado (22,9% de falso aceite contra 81,2%), mas foi treinado e testado por menos tempo.

Os dois modelos usam o Guitar Chords V3, que a autora me autorizou por escrito, [num comentário do dataset](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3/discussion/757671#3552939), a usar pra treinar e distribuir no app, com crédito e link. Isso vale pros dois e não é o motivo da escolha.

## Limitações

- **O teste de barulho gravado é pequeno:** 48 janelas (24 de cordas abafadas e 24 de afinando), gravadas na mesma sessão do resto do meu dataset. E, nele, o v3 aceita como acorde a maioria das janelas sem acorde (81,2%).
- **Não tenho um teste do meu violão em outras condições.** Não gravei a `s3`. Os `take02` são da mesma sessão do treino, então não servem pra decidir entre os dois.
- **A diferença no Test do V3 é de 1,7 ponto,** com uma seed de cada modelo.

## Consequências

- O app fica uns 420 KB maior por ter o segundo modelo (`export/export_report.json`, `size_kb_float32`).
- O app precisa mostrar o Lá maior só quando o v3_meu está escolhido.
- Cada modelo tem a sua GitHub Release no backend (`model-v3` e `model-v3-meu`), e o app confere o hash dos dois.
