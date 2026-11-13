
# ChordAI - Backend

Este projeto treina um dataset de audios de acordes musicais com o metodo CNN 1D e disponibiliza uma API Flask para visualizar uma interface onde o usuário podera ver o reconhecimento do acorde tocado.

App Flutter: [github.com/imxder/ChordAI](https://github.com/imxder/ChordAI)

---

## Requisitos

- Python 3.11 ou superior. Os resultados foram gerados com o Python 3.12.10.
- Versões usadas nos experimentos e no modelo em produção (lista completa em `requirements.lock.txt`):
  - **librosa 1.0.0**
  - **TensorFlow 2.21.0** (Keras 3.15.1)
  - NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1
- O `requirements.lock.txt` fixa essas versões. Ele importa porque a Parte 7b replica detalhes internos do librosa (a janela e o padding da STFT e o banco de filtros chroma), que mudam entre versões.

---

## 1. Instalação

### Passo 1: Clone o repositório

```bash
git clone https://github.com/imxder/ChordAI-backend.git
cd ChordAI-backend
```

### Passo 2: Crie e ative o ambiente virtual

No Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Passo 3: Instale as dependências

```bash
pip install -r requirements.txt
```

Para reproduzir os resultados com as mesmas versões exatas, instale o lock:

```bash
pip install -r requirements.lock.txt
```

## 2. Baixe o dataset

Uso duas fontes, em pastas separadas que nunca se misturam em disco:

- **[Guitar Chords V3](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3)**, da Fabiana Vinci (CC BY-SA 4.0). Ela autorizou por escrito, nos comentários do dataset, o uso pra treinar, avaliar e distribuir o modelo no app gratuito, com crédito e link. Fica em `dataset_kaggle/Training` e `dataset_kaggle/Test`.
- **Minhas gravações**, em `dataset_proprio/<acorde>/`. Como gravei está em [docs/roteiro-gravacao.md](docs/roteiro-gravacao.md).

```bash
python baixar_dataset.py              # V3 -> dataset_kaggle/ (o Kaggle entrega sem login)
python baixar_dataset.py --proprio    # meu dataset -> dataset_proprio/, depois que eu publicar
python -m scripts.montar_sources      # gera data/sources.csv com as duas fontes
```

O `data/sources.csv` tem uma linha por arquivo: caminho, acorde, fonte, instrumento, sessão, take, microfone e uso (treino ou teste). As sessões das minhas gravações estão em `data/sessoes.csv`.

O modelo que está no app hoje foi treinado com a pasta antiga `dataset/` (a Training do V3 mais 42 gravações minhas). Pra reproduzir ele, `python baixar_dataset.py --antigo` baixa esse zip do Google Drive. Essa opção sai quando o modelo com as duas fontes for aprovado.

## 3. Modelo de reconhecimento

### Pipeline em produção

O modelo em `trained_model/` (o mesmo de `candidates/candidate_chroma_gapbn_none/`) usa:

1. **Janela de 1,5 s** a 22 050 Hz, o mesmo tamanho do trecho que o app grava.
2. **Chroma** (`librosa.feature.chroma_cqt`, 12 bins × 65 frames, `hop_length=512`) no lugar dos 40 MFCC do baseline.
3. **StandardScaler por feature**: média e desvio de cada bin de chroma, calculados no treino.
4. **CNN 1D com BatchNorm + GlobalAveragePooling**: 47 mil parâmetros e 188 KB em TFLite, contra 268 mil parâmetros do baseline.
5. **Classe `_none`** ("nenhum acorde"), treinada com negativos sintéticos: áudio do dataset abafado em ruído, ruído branco, rosa, marrom ou de rede elétrica, e batidas percussivas.
6. **Limiar de confiança 0,50.** Previsões `_none`, ou com confiança abaixo do limiar, viram `status: "unknown"`.
   - O limiar foi calibrado **na validação** para este modelo (seed 42), com o critério da Parte 5: maior cobertura com falso aceite ≤ 5% e acurácia nos aceitos ≥ 98%.
   - O 0,45 que aparece em `experiments/results/threshold/threshold.md` era a **média das 3 seeds** da Parte 5.
   - O 0,50 é o limiar do modelo em produção. É também o valor que `train_model.py` calibra sozinho ao treinar com a seed 42.
7. **Silêncio:** janelas com RMS < 0,003 continuam sendo devolvidas como `status: "silence"`, antes de passar pelo modelo.

**`model_config.json`:**
- Cada modelo salvo tem esse arquivo, com feature, scaler, arquitetura, classes, rótulo `_none`, limiar, taxa de amostragem, janela e hop.
- `load_trained_model()` lê o arquivo, e a inferência extrai a feature e aplica o scaler exatamente como no treino.
- Uma pasta **sem** `model_config.json` (por exemplo, `candidates/candidate_91_81/`) é tratada como o pipeline antigo: 40 MFCC + scaler por frame, sem `_none` e sem limiar.

O código comum a produção e experimentos fica em `pipeline.py`: features, split, scalers, arquiteturas, negativos, limiar e `model_config.json`. `train_model.py` e `app.py` não dependem da pasta `experiments/`.

O contrato da API não mudou (`{status, chord, confidence}`), e o app Flutter não precisa de alterações.

### Resultados

O teste tem 297 arquivos, cortados em 1545 janelas de 1,5 s.
- O split é por arquivo: 20% de teste, `random_state=42`, sobre a lista de arquivos ordenada. Por isso é igual em qualquer sistema operacional.
- A validação usa 15% dos arquivos de treino. O teste só entra na avaliação final.

A tabela abaixo mostra médias ± desvio sobre as seeds 42, 43 e 44, sem limiar (argmax):

| Configuração | Acurácia (%) | Macro-F1 (%) | Acurácia com ruído (%) | Parâmetros |
|---|---:|---:|---:|---:|
| Baseline: MFCC 40 + scaler por frame + CNN 1D (Flatten) | 88,00 ± 0,94 | 88,00 ± 0,93 | 76,12 ± 0,73 | 268 104 |
| Chroma + scaler por frame + CNN 1D (Flatten) | 95,75 ± 0,54 | 95,73 ± 0,57 | 95,12 ± 0,42 | 259 144 |
| Chroma + scaler por feature + CNN 1D (Flatten) | 96,03 ± 0,63 | 96,00 ± 0,65 | 94,80 ± 0,93 | 259 144 |
| Chroma + scaler por feature + GAP+BN | 95,43 ± 0,36 | 95,40 ± 0,33 | 94,69 ± 0,51 | 46 920 |
| **Chroma + scaler por feature + GAP+BN + `_none`** (produção) | 95,32 ± 0,26 | — | 93,14 ± 0,45 | 47 049 |

- A coluna "Acurácia com ruído" usa o mesmo teste, com ruído rosa a 15 dB e passa-banda de 200 Hz a 5 kHz.
- **O que cada mudança fez:**
  - A troca de MFCC por chroma explica quase todo o ganho (+7,75 p.p.).
  - O scaler por feature, o augmentation e a CNN 2D não tiveram efeito significativo.
  - O GAP+BN custa ~0,6 p.p. e deixa o modelo 5,5× menor.
- A tabela completa, com efeitos pareados, F1 por classe, pares mais confundidos e latência, está em [`experiments/results/comparativo.md`](experiments/results/comparativo.md).

**Modelo em produção (seed 42, limiar 0,50).** Os números abaixo usam o mesmo caminho do app: `load_trained_model` + `recognize_chord` sobre o áudio bruto. O resultado completo está em [`experiments/results/parte6/equivalence.json`](experiments/results/parte6/equivalence.json).

| Janelas de acorde do teste | Janelas | % do total |
|---|---:|---:|
| Acorde certo | 1444 | 93,46 |
| Acorde errado | 26 | 1,68 |
| `unknown` (`_none` ou confiança < 0,50) | 15 | 0,97 |
| `silence` (RMS < 0,003, tratado à parte pelo app) | 60 | 3,88 |
| **Total** | **1545** | 100 |

- **A acurácia nos acordes, sem as janelas de silêncio, é 97,24%** (1444 de 1485).
  - O total de 93,46% conta como erro as 60 janelas silenciosas.
  - O app responde a essas janelas como `silence`, e não como um acorde.
- **Acurácia nos aceitos** (`status: "ok"`): 98,23%. **Cobertura**: 98,99% das janelas não silenciosas.
- **Negativos do teste** (223 janelas sem acorde): 98,21% viram `unknown`, ou seja, 1,79% de falso aceite.
- **Latência no PC:** ~95 ms para extrair o chroma + ~105 ms de `model.predict` ≈ 200 ms por janela, dentro do orçamento de 1,5 s.

**Modelo anterior** (`candidates/candidate_91_81/`, MFCC):
- No `evaluate_realtime.py`, ele dá 96,89% neste mesmo teste.
- Esse número não é comparável com os de cima. O modelo foi treinado com um split antigo, que dependia da ordem do `os.listdir()` e colocava no treino arquivos que hoje estão no teste.
- Com o split corrigido e a validação separada, o mesmo pipeline dá 88%.

### Treinar o modelo

Ao clonar o repositório, o modelo já vem treinado. Para treinar de novo, apague o conteúdo da pasta `trained_model` (ou mude `MODEL_DIR`) e execute:

```bash
python train_model.py
```

O que este script faz:

- Carrega os áudios da pasta `dataset` e corta janelas de 1,5 s.
- Separa treino, validação (15% dos arquivos de treino) e teste (20% dos arquivos).
- Gera os negativos da classe `_none`.
- Treina com EarlyStopping e ReduceLROnPlateau monitorando a **validação**.
- Calibra o limiar de confiança na validação.
- Salva `model.h5`, `encoder.joblib`, `scaler.joblib`, `model_config.json` e `metrics.json`.

A configuração fica nas constantes do topo de `train_model.py`. Os valores padrão são os da produção:

```python
FEATURE_TYPE = "chroma"   # "mfcc", "chroma", "cqt" ou "mfcc_chroma"
SCALER_TYPE = "feature"   # "feature" (por feature) ou "frame" (o do baseline)
ARCH = "gap_bn"           # "gap_bn", "flatten" (o do baseline) ou "cnn2d"
WITH_NONE = True          # classe "_none" + limiar de confiança
SEED = 42
```

Com a seed 42, o treino é determinístico e reproduz o modelo de produção: 95,60% nos acordes sem limiar, 98,21% de recall `_none` e limiar 0,50.

### Reproduzir o baseline

Para treinar o pipeline original do TCC (40 MFCC + scaler por frame + CNN 1D com Flatten, sem `_none`) sem mexer em `trained_model/`:

```bash
python -c "import train_model as t; t.train_model(out_dir='trained_model_baseline', feature='mfcc', scaler='frame', arch='flatten', with_none=False)"
python evaluate_realtime.py --model-dir trained_model_baseline
```

- **Resultado esperado:** 88,67% no teste com a seed 42. A média das 3 seeds é 88,00%.
- **Alternativa:** trocar as constantes do topo de `train_model.py` para `"mfcc"`, `"frame"`, `"flatten"` e `False`.
- **Pelos scripts de experimento:**

```bash
python -m experiments.exp_features --feature mfcc --seed 42        # uma seed
python -m experiments.run_seeds --only feat_mfcc                   # seeds 42, 43 e 44
```

### Avaliação e experimentos

```bash
python evaluate_realtime.py [--model-dir trained_model]   # acurácia, taxa de unknown e latência por janela
python confusion_report.py [--model-dir trained_model]    # matriz de confusão e relatório por acorde
python -m experiments.equivalence_check                   # produção x experimento (a diferença deve ser 0)
python -m experiments.split_hash --check                  # confere se o split de teste é o esperado
python -m experiments.run_seeds                           # todos os experimentos, 3 seeds
python -m experiments.threshold                           # curva do limiar (Parte 5)
python -m experiments.compare                             # gera experiments/results/comparativo.md
python -m pytest                                          # testes
```

Os pesos (`.h5`) dos modelos de experimento não ficam no Git (`experiments/**/*.h5` está no `.gitignore`).
- **O que fica versionado em `experiments/results/`:** `metrics.json`, matrizes de confusão, relatórios, `train.log`, `comparativo.md` e `equivalence.json`.
- **Os modelos que importam** ficam em `trained_model/` e `candidates/`.
- **Para regerar os modelos dos experimentos:** rode `python -m experiments.run_seeds`. O treino é determinístico por seed, então reproduz os mesmos pesos e métricas. Também dá para regerar só alguns grupos, com `--only feat_mfcc feat_chroma ...`.
- Scripts que carregam o `model.h5` de um experimento precisam dessa regeneração antes: `bench_latency.py`, `threshold.py`, `equivalence_check.py`, `promote_candidate.py`, `eval_degraded.py`, `eval_saved_model.py` e `eval_realtime_exp.py`.

## 4. Rodar a API Flask

Este servidor é o backend do app de reconhecimento de acordes. O app grava trechos de áudio de ~1,5s e envia para a API, que retorna o acorde reconhecido.

Para iniciar a API:

```bash
python app.py
```

A API fica disponível em:

```
http://localhost:5000
```

### Endpoint principal: `POST /predict_chord`

Recebe um arquivo WAV no campo `audio` (multipart/form-data) e retorna:

```json
{ "status": "ok", "chord": "C", "confidence": 0.98 }
```

Valores possíveis de `status`: `ok`, `silence`, `too_short` e `unknown` (nos três últimos, `chord` vem `null`).

A rota `/` abre uma interface web simples, usada para testar o reconhecimento direto pelo microfone do computador.

## 5. Deixar o servidor rodando (Windows)

Para deixar a API sempre ligada, use o `serve.py`, que sobe o servidor com waitress, sem janela, e grava a saída em `server.log`:

```bash
python serve.py
```

Para iniciar automaticamente ao fazer login no Windows, crie uma tarefa no Agendador de Tarefas (PowerShell):

```powershell
$dir = "C:\caminho\para\ChordAI-backend"
$action = New-ScheduledTaskAction -Execute "$dir\.venv\Scripts\pythonw.exe" -Argument "`"$dir\serve.py`"" -WorkingDirectory $dir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0 -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "VIOLAO_AI Server" -Action $action -Trigger $trigger -Settings $settings -Force
Start-ScheduledTask -TaskName "VIOLAO_AI Server"
```

Para acessar de outro dispositivo (ex.: celular), use o IP da máquina na rede (ou do Tailscale) na porta 5000.

---

## Estrutura dos arquivos
```
/
├── app.py                     # API Flask (Aplicação Principal)
├── serve.py                   # Sobe a API com waitress (servidor para deixar rodando no Windows)
├── baixar_dataset.py          # Baixa o V3 do Kaggle (e o meu dataset, depois de publicado)
├── scripts/
│   ├── cortar_takes.py        # Corta os takes longos em um arquivo por batida
│   ├── montar_sources.py      # Monta o data/sources.csv com as duas fontes
│   └── fontes.py              # Lista de acordes, pastas e formato do sources.csv
├── data/
│   ├── sessoes.csv            # Data, cômodo, microfone e uso de cada sessão de gravação
│   └── sources.csv            # Uma linha por arquivo de áudio das duas fontes
├── docs/
│   └── roteiro-gravacao.md    # Como gravei o meu dataset
├── pipeline.py                # Código comum a produção e experimentos (features, split, modelos, model_config)
├── train_model.py             # Treino e inferência (recognize_chord / predict_note)
├── audio_capture.py           # Script para captura de áudio do usuário (Microfone ou Interface)
|
├── evaluate_realtime.py       # Mede acurácia e latência em janelas de captura ao vivo
├── confusion_report.py        # Matriz de confusão e relatório por acorde
├── train_experiment_class_weight.py  # Experimento de treino com pesos por classe
├── train_ensemble_experiment.py      # Experimento de ensemble com vários modelos
|
├── requirements.txt           # Dependências do projeto
|
├── static/                    # Arquivos estáticos (CSS, JS)
│   ├── css/
│   │   └── style.css          # Arquivo CSS para estilização
│   └── js/
│       └── main.js            # Arquivo javascript
│
├── templates/                 # Templates HTML do Flask
│   └── index.html             # Página principal da aplicação
|
├── tests/                     # Testes (pytest)
│   ├── conftest.py            # Configuração compartilhada dos testes
│   ├── unit/
│   │   ├── test_app_logic.py      # Testes da lógica de estabilização do acorde
│   │   ├── test_unknown_chord.py  # Classe _none e limiar de confiança -> "unknown"
│   │   ├── test_model_config.py   # model_config.json e compatibilidade com o pipeline antigo
│   │   └── test_cortar_takes.py   # Script de corte do meu dataset (takes sintéticos)
│   └── integration/
│       └── test_api.py        # Testes dos endpoints da API
|
├── experiments/               # Experimentos do TCC (cada um salva em experiments/results/<nome>/)
│   └── results/comparativo.md # Tabela comparativa de todos os experimentos
|
├── candidates/                # Modelos candidatos
│   ├── candidate_90_61/
│   ├── candidate_90_67/
│   ├── candidate_91_81/       # Modelo antigo (MFCC, pipeline legado, sem model_config.json)
│   └── candidate_chroma_gapbn_none/  # Modelo atual (o mesmo de trained_model/)
|
└── trained_model/             # Modelo em uso pela API
    ├── model.h5
    ├── encoder.joblib
    ├── scaler.joblib
    ├── model_config.json      # Feature, scaler, arquitetura, classes, _none e limiar
    └── metrics.json
```
