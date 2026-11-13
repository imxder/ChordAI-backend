# Roteiro de gravação do meu dataset

Gravo no fim de semana de 14 e 15/11/2026 pra ter dados meus ao lado do [Guitar Chords V3](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3). Uma sessão inteira de violão fica só pra teste, pra medir o modelo com gravações que ele nunca ouviu.

## Antes de começar

- **Metrônomo só no fone.** Se o clique vazar no microfone, o script de corte acha que é uma batida. No Ableton, o metrônomo não entra no áudio gravado, mas o som da caixa de som entra.
- **Sem efeito nenhum** na faixa: nada de EQ, compressor, reverb, chorus ou drive. Eles mudam o chroma, que é o que o modelo olha.
- **Volume:** os picos entre −12 e −6 dB no medidor do Ableton. Nunca no vermelho.
- **Preencher `data/sessoes.csv`** com a data, o cômodo e o microfone de cada sessão. Sem o microfone preenchido, o script de corte para e avisa.
- **Nome de cada take:** `<acorde>_<instrumento>_<sessao>_take<NN>.wav`, por exemplo `Am_violao_s1_take01.wav`. Instrumento e sessão em minúsculas, sem acento. "Sem acorde" usa `none` no lugar do acorde.

## Taxa de amostragem

Gravo e exporto em **44.100 Hz**. O pipeline converte tudo pra 22.050 Hz ao carregar (`SAMPLE_RATE` em `pipeline.py`), que é a mesma taxa em que o app grava no celular.

O V3 está em 16.000 Hz, então não tem nada acima de 8 kHz. Mesmo assim não vale gravar em 16 kHz: o chroma pesa principalmente as notas no meio do violão (a ponderação de `CHROMA_STFT_CTROCT` e `CHROMA_STFT_OCTWIDTH` em `pipeline.py`), e acima de 8 kHz quase não entra. Gravar mais alto não estraga nada e deixa o arquivo original mais fiel.

## Como tocar em cada take

- Metrônomo a **60 BPM**, compasso 4/4: um compasso dura 4 segundos.
- **Uma batida por compasso, no tempo 1**, e deixo soar até o próximo. O script corta 2,5 s depois de cada batida, então sobra 1,5 s de folga.
- Se errar, não paro: toco o próximo compasso. O script descarta o compasso com duas batidas, e eu confiro no relatório.
- O **arpejo** tem que ser rápido, as seis cordas em menos de meio tempo (uns 0,3 s). Arpejo lento vira "duas batidas" e é descartado.

Num take de 8 compassos (s1, s2 e g1), eu vario assim:

| Compassos | Como |
| --- | --- |
| 1 e 2 | rasgueado para baixo, forte |
| 3 e 4 | rasgueado para baixo, fraco |
| 5 e 6 | rasgueado para cima |
| 7 e 8 | arpejo rápido |

Nos takes de 12 compassos (s3), faço 3 compassos de cada jeito, na mesma ordem.

Cada acorde tem dois takes por sessão (`take01` e `take02`). Cortes do mesmo take saem quase iguais, então na hora de treinar a validação separa por take, não por arquivo; com dois takes por acorde em cada sessão, sobra take pra validação.

## Sessões

| Sessão | Quando | Instrumento | Microfone e lugar | Takes por acorde | Uso |
| --- | --- | --- | --- | --- | --- |
| `s1` | sábado | violão | interface, microfone a uns 30 cm da boca do violão | 2 de 8 compassos | treino |
| `s2` | sábado | violão | celular a uns 1 m, em outro cômodo | 2 de 8 compassos | treino |
| `g1` | sábado | semi-acústica desplugada (`semi`) e Jazzmaster limpa pela GT-1 ou direto (`jazzmaster`) | interface | 2 de 8 compassos em cada guitarra | treino |
| `s3` | domingo | violão | outro celular ou outra posição, outro cômodo | 2 de 12 compassos | **teste** |

Por acorde, isso dá:

| Instrumento | Treino | Teste |
| --- | --- | --- |
| violão | 32 (16 da `s1` + 16 da `s2`) | 24 (`s3`) |
| semi | 16 | – |
| Jazzmaster | 16 | – |
| **total** | **64** | **24** |

São 11 acordes (os 8 do app mais D, A e E), então no fim de semana são 88 takes de acorde: 22 na `s1`, 22 na `s2`, 22 por guitarra na `g1` e 22 na `s3`. Os números de cortes contam um corte por compasso; os descartados pelo script saem dessa conta.

A `s3` é a sessão de teste: tudo nela tem que ser diferente das outras (outro dia, outro cômodo, outro microfone ou posição). Se ela parecer demais com o treino, o número do teste fica otimista.

### Acordes

Am, Bb, Bdim, C, Dm, Em, F e G, e também D, A e E, em **todas** as sessões, inclusive na `s3`. Sem a `s3`, os extras não teriam teste. Eles ficam na pasta, mas só entram num treino separado, depois do modelo dos 8 acordes.

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

Na `s1` e na `s2` gravo seis desses (uns 72 pedaços pra treino). Na `s3` gravo três (uns 36 pra teste), pra medir quantas vezes o modelo "acha" um acorde onde não tem.

Exemplo de nome: `none_violao_s3_take02.wav`.

## Exportar do Ableton

Arquivo → Exportar Áudio/Vídeo:

- **Faixa renderizada:** só a faixa do instrumento (não a Master).
- **Início e duração:** o take inteiro.
- **Tipo:** WAV, **44.100 Hz**, **24 bits**.
- **Converter para mono:** ligado.
- **Normalizar:** desligado.
- **Dither:** nenhum.

Um arquivo por take, direto na pasta `gravacoes/` (ela não vai pro Git).

## Depois de cada take

```bash
python -m scripts.cortar_takes gravacoes/Am_violao_s1_take01.wav --conferir
```

O relatório mostra quantos cortes saíram e por que os outros foram descartados (`duas batidas`, `silencio`, `fraco` ou `curto`). O esperado é um corte por compasso: 8 num take da `s1`, `s2` ou `g1` e 12 num take da `s3`. Se vier muito menos, ouço o take antes de seguir. Se as batidas fracas caírem como `fraco`, me aviso antes de gravar o resto.

Quando estiver tudo certo, corto de verdade:

```bash
python -m scripts.cortar_takes gravacoes/
```

Os cortes vão pra `dataset_proprio/<acorde>/` e as linhas entram no `data/sources.csv`. O take original nunca é alterado. Se precisar cortar um take de novo, uso `--refazer`.
