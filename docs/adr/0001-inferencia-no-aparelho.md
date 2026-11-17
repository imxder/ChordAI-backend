# ADR 0001: o reconhecimento de acordes roda no aparelho

- **Status:** aceita
- **Data:** 08/11/2026 (exportação do modelo para TFLite) e 10/11/2026 (o app passa a usar o aparelho por padrão)
- **Repositórios:** [ChordAI-backend](https://github.com/imxder/ChordAI-backend) e [ChordAI](https://github.com/imxder/ChordAI)
- **Caminhos de arquivo:** os que começam com `lib/` ou `assets/` são do app. Os outros são do backend.

## Contexto

No começo, o app gravava 1,5 s de áudio e mandava para a API Flask do backend (`POST /predict_chord`), que devolvia o acorde. Isso tinha quatro problemas:

- **Latência.** Cada janela ia e voltava pela rede. No servidor, só o `model.predict` da CNN levava uns 109 ms por janela no meu PC, sem contar a extração do chroma nem a rede ([`experiments/results/comparativo.md`](https://github.com/imxder/ChordAI-backend/blob/main/experiments/results/comparativo.md), coluna "ms predict" do `feat_chroma_scalerfeat_gapbn`).
- **Precisava de rede e de um computador ligado.** Sem o meu PC na mesma Wi-Fi do celular, o reconhecimento não funcionava.
- **Privacidade.** O áudio do microfone saía do celular.
- **A regra do Playbook.** O projeto tem que estar disponível "sem depender de notebook pessoal ou ambiente local" e "não apenas rodando em *localhost*" ([`Portfolio.md`](https://github.com/CatolicaSC-Portfolio/The-Portfolio-Playbook/blob/main/Portfolio.md), linhas 21 e 43). Hospedar a API na nuvem resolveria isso, mas com custo mensal.

## Decisão

O modelo roda no próprio celular, em TFLite (`OnDeviceChordService` no app), e é o modo padrão. O modo servidor continua como opção avançada, em Perfil → Reconhecimento. Os dois implementam a mesma interface, `ChordRecognizer`.

A extração da feature fica **dentro** do `.tflite`: o app manda o áudio bruto (33 075 amostras, 1,5 s a 22 050 Hz) e recebe as probabilidades. Pra isso troquei a chroma_cqt da API pela chroma_stft com n_fft 8192. O STFT vira a operação `RFFT2D`, que está na lista de operações do modelo (`export/export_report.json`, `ops_float32`), e o CQT não tem equivalente pronto no TensorFlow. Assim o app não precisa calcular chroma em Dart. Essa troca custou uns 0,3 ponto no teste sorteado da pasta antiga (o C1 contra a produção na Parte 7a de [`experiments/results/comparativo.md`](https://github.com/imxder/ChordAI-backend/blob/main/experiments/results/comparativo.md): −0,28 ± 0,88 p.p.).

## Alternativas que considerei

- **Manter a API Flask no meu PC.** Não cumpre o Playbook e não funciona fora de casa.
- **Hospedar a API na nuvem.** Resolve a disponibilidade, mas continua precisando de internet, mandando o áudio pra fora do celular e custando dinheiro todo mês.
- **Rodar no aparelho, mas calcular o chroma em Dart.** Exigiria reimplementar e manter a feature igual à do librosa em outra linguagem. Com a feature dentro do `.tflite`, o mesmo grafo serve pro backend conferir a equivalência (`export/build_tflite.py` compara o TFLite com o librosa em 200 janelas).

## Consequências

- **Boas:**
  - funciona offline;
  - o áudio não sai do celular;
  - não tem custo de nuvem;
  - a inferência leva uns 9 ms por janela no meu PC com 1 thread (`export/export_report.json`, `latency_ms_tflite_1thread`). No celular eu ainda não medi.
- **O custo:** **atualizar o modelo exige uma versão nova do app**, porque o `.tflite` vai dentro dele, em `assets/models/`. Pra saber qual arquivo está em cada versão, cada modelo fica numa GitHub Release do backend, e o app confere o hash do `.tflite` contra o `model_info.json`.
- **O app fica maior.** Cada modelo float32 tem uns 420 KB (`export/export_report.json`, `size_kb_float32`).
- **A API Flask vira ferramenta de desenvolvimento.** Ela continua usando o modelo antigo dela (`trained_model/`, chroma_cqt), que não é nenhum dos modelos do app.
