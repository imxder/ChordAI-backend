# Experimentos com as duas fontes (15/11/2026)

Aqui eu explico as siglas de `experiments/results_duas_fontes/`, a ordem em que testei, o que cada rodada mostrou e por que o **F9** virou o modelo `v3_meu` do app.

Todos os números de acurácia e falso aceite vêm de [`experiments/results_duas_fontes/comparativo.md`](../experiments/results_duas_fontes/comparativo.md), que é a média das seeds 42, 43 e 44. Quando o número é de uma seed só, eu digo de qual `metrics.json` ele veio. O script que treina e avalia tudo é o `scripts/comparar_modelos.py`, e a divisão dos dados é o `scripts/dividir_treino.py`.

Os experimentos mais antigos, de 08/11 (MFCC × chroma, scaler, GAP+BN, classe `_none`, chroma_stft × chroma_cqt), estão em [`experiments/results/comparativo.md`](../experiments/results/comparativo.md) e no README.

## Os testes

Os mesmos para todos os modelos (tamanhos do `comparativo.md`, em janelas de 1,5 s):

| Teste | O que é | Janelas |
|---|---|---|
| **Test do V3** | a pasta `dataset_kaggle/Test`: 10 pessoas/instrumentos que não aparecem no treino. É o teste independente. | 1508 |
| **take02** | o `take02` de cada acorde da `s1` e da `g1`. É da **mesma sessão** do treino (mesmo dia, cômodo e microfone), então o número sai otimista. | 256 dos 8 acordes + 96 dos extras |
| **Sem acorde gravado** | os meus takes de cordas abafadas e afinando, que não entram no treino. Também são da mesma sessão. | 48 (24 + 24) |
| **Sem acorde sintético** | ruído, percussão e acorde em volume baixo, gerados pelo `pipeline.py` | 260 |

- **Acurácia:** a classe de maior probabilidade em todas as janelas, sem limiar.
- **Falso aceite:** a fração das janelas sem acorde que o app mostraria como acorde, ou seja, passa do filtro de silêncio, não é `_none` e tem confiança acima do limiar do modelo.

## Como ler os nomes

`<sigla>_<feature>_<aug>_s<seed>`, por exemplo `F9_stft8192_aug2_s43`:

- **sem sufixo de feature:** `chroma`, a chroma_cqt que a API Flask usa (`trained_model/`);
- **`_stft8192`:** a chroma_stft com n_fft 8192, a feature que roda dentro do `.tflite` do celular;
- **`_aug2`:** duas versões aumentadas de cada janela de acorde no treino (ruído, ganho, passa-banda e deslocamento, de `experiments/augment.py`);
- **`_s42`, `_s43`, `_s44`:** a seed;
- **`_ens`:** a média das probabilidades das três seeds (não treina nada novo).

## Glossário

| Sigla | O que muda, em uma linha | Test do V3 | Falso aceite gravado |
|---|---|---|---|
| `0` | o `trained_model/` da API Flask, só avaliado (seed 42, chroma_cqt) | 90,7% | 87,5% |
| `0c` | o `.tflite` do app, o modelo **v3** (C1, seed 44), só avaliado | 87,4% | 81,2% |
| `R` | a receita do `0` retreinada nas 3 seeds, com o `dataset/` e o split sorteado do `train_model.py` | 90,4% | 75,7% |
| `R_stft8192_aug2` | a receita do C1 (o v3 do app) retreinada nas 3 seeds | 87,4% | 88,9% |
| `A` | só o meu dataset (s1 + g1 + antiga), 8 acordes | 76,2% | 18,1% |
| `B8` | o meu + o Training do V3, 8 acordes, validação separada por pessoa | 89,3% | 26,4% |
| `B8v` | o B8 com a validação sorteada por arquivo, como o `train_model.py` | 90,3% | 30,6% |
| `B11` | o B8 + D, A e E, com peso por classe | 88,3% | 35,4% |
| `B11v` | o B11 com a validação sorteada | 89,2% | 36,8% |
| `B11vs` | o B11v sem o peso por classe | 90,3% | 39,6% |
| `B11vs_ens` | a média das 3 seeds do B11vs | 90,6% | 37,5% |
| `F` | o B11vs treinado com tudo, inclusive os take02 (o primeiro candidato a modelo final) | 89,4% | 36,1% |
| `Fa` | o F com negativos sintéticos de afinação no `_none` | 89,2% | 18,8% |
| `B11vs_stft8192` | o B11vs com a feature do celular | 84,3% | 41,7% |
| `B11vs_stft8192_aug2` | o mesmo, com o aug2 | 85,7% | 45,1% |
| `B11vsaf_stft8192_aug2` | o anterior com a s1 e a g1 levadas ao Lá 440 no treino | 85,9% | 47,2% |
| `B8v_stft8192_aug2` | o B8v com a feature do celular e o aug2 (sem os extras) | 86,3% | 41,7% |
| `B8vp_stft8192_aug2` | o anterior com peso 0,5 nos meus cortes | 86,5% | 42,4% |
| `Fc_stft8192_aug2` | modelo final do celular: 11 acordes, peso 0,5 nos meus cortes, afinação sintética, com os take02 no treino | 84,2% | 16,7% |
| `Fd_stft8192_aug2` | o Fc sem os take02 no treino | 85,4% | 31,2% |
| `F9_stft8192_aug2` | o Fd só com o A de acorde extra (8 + A): o **v3_meu** do app | 85,7% | 29,9% |
| `V3_stft8192_aug2` | só os acordes do V3, sem nenhum áudio meu, receita do celular | 87,4% | 79,2% |
| `V3N_stft8192_aug2` | o V3 mais o meu sem acorde gravado | 86,3% | 43,8% |

O F, o Fa e o Fc treinam com os take02, então a coluna take02 deles não vale. A prova deles é o Test do V3 e o sem acorde gravado.

## Em que ordem eu testei e o que cada rodada mostrou

A ordem e os horários são dos commits de 15/11.

1. **0, A, B8 e B11** (12:40). Nenhum modelo novo passou o atual no Test do V3: o 0 deu 90,7%, o B8 89,3% e o B11 88,3%. O meu dataset sozinho (A) deu 76,2%, ou seja, ele não basta pra reconhecer o violão de outras pessoas. O ganho apareceu no sem acorde gravado: o 0 mostra acorde em 87,5% dessas janelas e o B8 em 26,4%.
2. **R** (13:13). Retreinei a receita do 0 nas 3 seeds, pra não comparar o B8 com uma seed só. Deu 90,4% (de 90,2% a 90,7%). O 90,7% do 0 está no topo dessa faixa.
3. **B8v** (13:23). Fiz o B8 com a validação sorteada por arquivo, como o `train_model.py`, pra ver se a diferença vinha do jeito de validar. O B8v deu 90,3%, quase igual ao R. Quase toda a diferença do B8 vinha da validação.
4. **B11v e B11vs** (13:40). O B11 mudava duas coisas de uma vez, os extras e o peso por classe. Sem o peso (B11vs), ele voltou a 90,3% e acertou 93,1% dos extras nos take02. O peso por classe é que custava ponto.
5. **F e Fa** (13:55 e 14:11). O F é o B11vs treinado com tudo, e deu 89,4%. No Fa coloquei negativos sintéticos de afinação, uma corda com a nota deslizando até afinar. O falso aceite gravado caiu de 36,1% pra 18,8%, e o Test do V3 ficou em 89,2%.
6. **A feature do celular** (15:06). Aqui descobri que o app não usa o `trained_model/`, e sim o C1, que é o `.tflite` com a chroma_stft dentro. Avaliei o `.tflite` do app (0c), que deu 87,4% no Test do V3, e retreinei a receita dele (R_stft8192_aug2), que deu 87,4% também. Com a feature do celular, o B11vs caiu pra 84,3%, e com o aug2 subiu pra 85,7%. Na mesma rodada testei o ensemble (B11vs_ens), explicado abaixo.
7. **Corrigir a afinação** (15:26). A s1 e a g1 ficaram uns 10 cents abaixo do Lá 440, e a chroma_stft do celular usa `tuning=0`, então não compensa isso (a chroma_cqt do servidor compensa). Levei os meus cortes ao Lá 440 no treino (B11vsaf): 85,9% contra 85,7%. Não mudou nada.
8. **Sem os extras** (15:42). O B8v com a feature do celular deu 86,3%, uns 0,6 ponto acima do B11vs (85,7%). Quase todo esse ponto estava no Dm (descrição do PR #4).
9. **Peso 0,5 nos meus cortes** (16:29). Testei pra ver se o modelo estava se especializando no meu violão. O B8vp deu 86,5% contra 86,3%. Não mudou nada.
10. **Fc e Fd** (17:25). O Fc juntou o que deu certo pro celular: 11 acordes, afinação sintética, peso 0,5 e os take02 no treino. Deu 84,2%. O Fd é o mesmo sem os take02 no treino, e deu 85,4%. Colocar os take02 no treino custou 1,2 ponto nas médias, então exportei o Fd.
11. **F9** (18:25). Instalei o Fd no iPhone, e o D e o E confundiam com Dm e Em. O A funcionou. Treinei o F9, que é o Fd só com o A de extra. Deu 85,7% no Test do V3, 100% de acerto no A nos take02 (nas 3 seeds) e 29,9% de falso aceite gravado.
12. **V3 e V3N** (19:11). Explicado abaixo.

## O ensemble (B11vs_ens)

Testei o ensemble pra ver se a média das 3 seeds acertava mais. Ele ganhou só 0,3 ponto no Test do V3 (90,6% contra 90,3% do B11vs), perdeu nos acordes extras (91,7% contra 93,1%) e custaria o triplo de tempo e de tamanho no celular, porque seriam três modelos rodando. Não compensou.

## V3 e V3N

Com o V3 e o V3N eu queria separar o efeito dos meus acordes gravados do efeito do meu sem acorde gravado. A única diferença entre os dois é o sem acorde gravado:

- **O sem acorde gravado custa uns 1,1 ponto no Test do V3** (87,4% → 86,3%) e corta o falso aceite gravado de 79,2% pra 43,8%. Vale a pena.
- **Do V3N (86,3%) pro F9 (85,7%) cai mais um pouco.** Essa diferença não é só dos meus acordes gravados: o F9 também tem o A e a afinação sintética, e o V3N não tem nenhum dos dois. Então não dá pra dizer quanto é de cada um.

## Por que o F9 virou o v3_meu

- Ele tem o A, que funcionou no iPhone, e não tem o D e o E, que não funcionaram.
- Contra o Fd, ele ficou um pouco melhor nos três números: 85,7% contra 85,4% no Test do V3, 100% contra 96,9% nos extras dos take02 e 29,9% contra 31,2% de falso aceite gravado.
- Contra o v3 do app (0c), ele perde no Test do V3 (85,7% contra 87,4%), mas mostra acorde bem menos vezes quando não tem acorde (29,9% contra 81,2%).

A seed foi escolhida pela **validação**, nunca pelo teste: a maior acurácia nos acordes da validação. A 43 deu 94,92%, contra 94,58% da 42 e 94,50% da 44 (`export/export_report.json`, em `seed_selection`). O limiar é 0,55, o que a validação calibrou nas três seeds (`export/model_info.json`).

A seed 43, que está no app, tem números próprios no `experiments/results_duas_fontes/F9_stft8192_aug2_s43/metrics.json`: 85,68% no Test do V3 e 22,9% de falso aceite gravado.

Por que o app continua com o v3 como padrão está no [ADR 0002](adr/0002-dois-modelos-no-app.md).

## Cuidados com estes números

- **O teste de sem acorde gravado é pequeno:** 48 janelas, da mesma sessão em que gravei o resto. Ele mostra uma tendência, não uma taxa que vale pra qualquer lugar.
- **O take02 é da mesma sessão do treino.** Quase todo modelo dá entre 98% e 100% nele, então ele quase não separa um modelo do outro.
- **O teste independente é o Test do V3.** Não gravei a `s3`, que seria o teste com o meu violão em outro dia, cômodo e microfone.
