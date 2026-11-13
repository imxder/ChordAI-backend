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

Num take de 16 compassos, eu vario assim:

| Compassos | Como |
| --- | --- |
| 1 a 4 | rasgueado para baixo, forte |
| 5 a 8 | rasgueado para baixo, fraco |
| 9 a 12 | rasgueado para cima |
| 13 a 16 | arpejo rápido |

Nos takes de 12 compassos, faço 3 compassos de cada.

## Sessões

| Sessão | Quando | Instrumento | Microfone e lugar | Compassos por acorde | Uso |
| --- | --- | --- | --- | --- | --- |
| `s1` | sábado | violão | interface, microfone a uns 30 cm da boca do violão | 16 | treino |
| `s2` | sábado | violão | celular a uns 1 m, em outro cômodo | 16 | treino |
| `g1` | sábado | semi-acústica desplugada (`semi`) e Jazzmaster limpa pela GT-1 ou direto (`jazzmaster`) | interface | 12 de cada | treino |
| `s3` | domingo | violão | outro celular ou outra posição, outro cômodo | 12 | **teste** |

Isso dá 44 cortes de violão por acorde (16 + 16 + 12), 12 de semi e 12 de Jazzmaster.

A `s3` é a sessão de teste: tudo nela tem que ser diferente das outras (outro dia, outro cômodo, outro microfone ou posição). Se ela parecer demais com o treino, o número do teste fica otimista.

### Acordes

Am, Bb, Bdim, C, Dm, Em, F e G em todas as sessões. No sábado gravo também D, A e E do mesmo jeito; eles ficam na pasta, mas só entram num treino separado, depois.

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

O relatório mostra quantos cortes saíram e por que os outros foram descartados (`duas batidas`, `silencio`, `fraco` ou `curto`). Num take de 16 compassos o esperado é perto de 16 cortes. Se vier muito menos, ouço o take antes de seguir.

Quando estiver tudo certo, corto de verdade:

```bash
python -m scripts.cortar_takes gravacoes/
```

Os cortes vão pra `dataset_proprio/<acorde>/` e as linhas entram no `data/sources.csv`. O take original nunca é alterado. Se precisar cortar um take de novo, uso `--refazer`.
