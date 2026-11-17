# Roteiro de gravação do meu dataset

Gravei no sábado, 14/11/2026, pra ter dados meus ao lado do [Guitar Chords V3](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3). O plano era gravar quatro sessões no fim de semana, mas gravei só duas. Este arquivo mostra primeiro o que gravei de verdade e depois o roteiro que segui pra gravar e cortar.

## O que gravei de verdade

| Sessão | Instrumento | Microfone | Takes | Cortes |
| --- | --- | --- | --- | --- |
| `s1` | violão | Gravador do iPhone 11, a uns 30 cm da boca do violão, no quarto | 2 por acorde nos 11 acordes + 6 de sem acorde | 176 de acorde + 74 sem acorde |
| `g1` | Tagima TW-61 (estilo Jazzmaster), som limpo | Boss GT-1 sem efeitos, gravado no Ableton | 2 por acorde nos 11 acordes | 176 |

Os números de cortes são as linhas da `s1` e da `g1` no `data/sources.csv`. Nos arquivos, a Tagima aparece como `jazzmaster`.

**Não gravei** a `s2` (violão em outro cômodo), a `s3` (a sessão de teste) nem a semi-acústica da `g1`. Elas aparecem mais abaixo só porque eram parte do plano.

### Como isso vira teste

Sem a `s3`, não tenho um teste do meu violão em outro dia, cômodo e microfone. No treino com as duas fontes (`scripts/dividir_treino.py`):

- **O teste independente é a pasta Test do V3**: 320 arquivos de pessoas e violões que não aparecem no treino.
- **O `take02` de cada acorde é teste, mas da mesma sessão do treino.** Ele foi gravado no mesmo dia, cômodo e microfone do `take01`, então o número sai otimista. Serve pra comparar modelos entre si, não pra prometer acerto com um usuário novo.
- **Sem acorde:** os takes de cordas abafadas (`take04`) e afinando (`take06`) ficam de fora do treino e viram teste. São só 48 janelas, também da mesma sessão.

O meu dataset está no Kaggle: [ChordAI Guitar Chords](https://www.kaggle.com/datasets/imxder/chordai-guitar-chords-v1).

## O roteiro que eu segui

### Antes de começar

- **Metrônomo só no fone.** Se o clique vazar no microfone, o script de corte acha que é uma batida. No Ableton, o metrônomo não entra no áudio gravado, mas o som da caixa de som entra.
- **Sem efeito nenhum** na faixa: nada de EQ, compressor, reverb, chorus ou drive. Eles mudam o chroma, que é o que o modelo olha.
- **Volume:** os picos entre −12 e −6 dB no medidor do Ableton. Nunca no vermelho.
- **Preencher `data/sessoes.csv`** com a data, o cômodo e o microfone de cada sessão. Sem o microfone preenchido, o script de corte para e avisa.
- **Nome de cada take:** `<acorde>_<instrumento>_<sessao>_take<NN>.wav` (ou `.m4a`, do iPhone), por exemplo `Am_violao_s1_take01.wav`. Instrumento e sessão em minúsculas, sem acento. "Sem acorde" usa `none` no lugar do acorde.

### Taxa de amostragem

O plano era gravar e exportar em **44.100 Hz**. No fim, a `s1` saiu do iPhone e a `g1` do Ableton, as duas em 48.000 Hz. Não muda nada: o pipeline converte tudo pra 22.050 Hz ao carregar (`SAMPLE_RATE` em `pipeline.py`), que é a mesma taxa em que o app grava no celular.

O V3 está em 16.000 Hz, então não tem nada acima de 8 kHz. Mesmo assim não vale gravar em 16 kHz: o chroma pesa principalmente as notas no meio do violão (a ponderação de `CHROMA_STFT_CTROCT` e `CHROMA_STFT_OCTWIDTH` em `pipeline.py`), e acima de 8 kHz quase não entra. Gravar mais alto não estraga nada e deixa o arquivo original mais fiel.

### Como tocar em cada take

- Metrônomo a **60 BPM**, compasso 4/4: um compasso dura 4 segundos.
- **Uma batida por compasso, no tempo 1**, e deixo soar até o próximo. O script corta 2,5 s depois de cada batida, então sobra 1,5 s de folga.
- Se errar, não paro: toco o próximo compasso. O script descarta o compasso com duas batidas, e eu confiro no relatório.
- O **arpejo** tem que ser rápido, as seis cordas em menos de meio tempo (uns 0,3 s). Arpejo lento vira "duas batidas" e é descartado.

Num take de 8 compassos, eu vario assim:

| Compassos | Como |
| --- | --- |
| 1 e 2 | rasgueado para baixo, forte |
| 3 e 4 | rasgueado para baixo, fraco |
| 5 e 6 | rasgueado para cima |
| 7 e 8 | arpejo rápido |

Cada acorde tem dois takes por sessão (`take01` e `take02`). Cortes do mesmo take saem quase iguais, então nenhum take pode ter cortes no treino e no teste ao mesmo tempo: o `take02` inteiro fica de teste.

### Sessões que eu planejei

| Sessão | Instrumento | Uso | Gravei? |
| --- | --- | --- | --- |
| `s1` | violão | treino | sim, no iPhone 11 |
| `s2` | violão, em outro cômodo e a uns 1 m | treino | não |
| `g1` | semi-acústica desplugada e Jazzmaster limpa | treino | só a Tagima (Jazzmaster) |
| `s3` | violão em outro dia, cômodo e microfone, com 12 compassos por take | **teste** | não |

### Acordes

Am, Bb, Bdim, C, Dm, Em, F e G, e também D, A e E. No modelo que foi pro app (`v3_meu`), só o A entrou de extra: o D e o E confundiam com Dm e Em no iPhone (`export/README.md`).

### Sem acorde

Cada take de "sem acorde" tem uns 30 segundos e vira uns 12 pedaços de 2,5 s (o script não procura batida nesses, só corta em sequência e pula o que for silêncio).

| Take | O que gravar |
| --- | --- |
| fala | eu falando perto do violão |
| casa | ventilador, TV baixa, barulho da rua |
| nota solta | uma corda por vez, devagar, sem formar acorde |
| abafado | batendo nas cordas com a mão em cima, sem soar nota |
| corpo | batida no corpo do violão, palheta raspando |
| afinando | girando a tarraxa com a corda soando |

Na `s1` gravei os seis, nessa ordem (`take01` a `take06`), e eles viraram 74 pedaços. O abafado (`take04`) e o afinando (`take06`) ficam de teste.

Exemplo de nome: `none_violao_s1_take02.m4a`.

### Exportar do Ableton

Arquivo → Exportar Áudio/Vídeo:

- **Faixa renderizada:** só a faixa do instrumento (não a Master).
- **Início e duração:** o take inteiro.
- **Tipo:** WAV, **44.100 Hz**, **24 bits**.
- **Converter para mono:** ligado.
- **Normalizar:** desligado.
- **Dither:** nenhum.

Um arquivo por take, direto na pasta `gravacoes/` (ela não vai pro Git).

Antes de exportar, confiro se o **Warp** do clipe está desligado. Com o Warp ligado, o Ableton estica o áudio no tempo sem mudar a afinação: o primeiro take saiu com o dobro da duração (64 s em vez de 32 s), o som ficou estranho e o script errou as batidas. Depois de exportar, confiro se a duração bate com a do take gravado.

### Gravar no iPhone

O `.m4a` que o Gravador exporta vai direto pro script, sem passar pelo Ableton. Renomeio no padrão (`C_violao_s1_take01.m4a`) e rodo o script apontando pro arquivo, onde ele estiver:

```bash
python -m scripts.cortar_takes ~/Downloads/C_violao_s1_take01.m4a --conferir
```

O script converte com o `afconvert` do macOS para `gravacoes/<take>.wav` (mesma taxa de amostragem, 24 bits) e corta a partir desse `.wav`. O `.m4a` não é alterado. No `--conferir`, a conversão vai para uma pasta temporária e nada fica no projeto.

### Depois de cada take

```bash
python -m scripts.cortar_takes gravacoes/Am_violao_s1_take01.wav --conferir
```

O relatório mostra quantos cortes saíram e por que os outros foram descartados (`duas batidas`, `silencio`, `fraco` ou `curto`). O esperado é um corte por compasso, 8 por take. Se vier muito menos, ouço o take antes de seguir. Se as batidas fracas caírem como `fraco`, me aviso antes de gravar o resto.

Quando estiver tudo certo, corto de verdade:

```bash
python -m scripts.cortar_takes gravacoes/
```

Os cortes vão pra `dataset_proprio/<acorde>/` e as linhas entram no `data/sources.csv`. O take original nunca é alterado. Se precisar cortar um take de novo, uso `--refazer`.
