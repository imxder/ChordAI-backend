
# VIOLÃO AI

Este projeto treina um dataset de audios de acordes musicais com o metodo CNN 1D e disponibiliza uma API Flask para visualizar uma interface onde o usuário podera ver o reconhecimento do acorde tocado.

---

## Requisitos

- Python 3.11 ou superior

---

## 1. Instalação

### Passo 1: Clone o repositório

```bash
git clone https://github.com/imxder/VIOLAO_AI.git
cd VIOLAO_AI
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

## 2. Baixe o DATASET.

- Execute o codigo de `baixar_bataset.py`.

```bash
python baixar_bataset.py
```

Este script deve:

- Baixar a pasta do dataset do Google Drive descomprimir.
- Descomprimir pasta `dataset.zip`.

## 3. Treinar o modelo CNN 1D

Ao clonar o repositório já vem o modelo treinado,
para treinar novamente basta apagar o conteúdo da pasta `trained_model`.

Execute o treinamento do modelo:

```bash
python train_model.py
```

O que este script faz:

- Carrega os audios da pasta `dataset`.
- Treina o modelo CNN.

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
$dir = "C:\caminho\para\VIOLAO_AI"
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
├── baixar_dataset.py          # Script para baixar o dataset (via gdown)
├── train_model.py             # Script para treinar modelo (CNN 1D)
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
│   │   └── test_app_logic.py  # Testes da lógica de estabilização do acorde
│   └── integration/
│       └── test_api.py        # Testes dos endpoints da API
|
├── candidates/                # Modelos candidatos gerados nos experimentos
│   ├── candidate_90_61/
│   ├── candidate_90_67/
│   └── candidate_91_81/       # (cada um com .h5, encoder e scaler)
|
└── trained_model/             # Modelo em uso pela API
    ├── chord_recognizer_cnn_model.h5
    ├── label_encoder_chords.joblib
    └── scaler_chords.joblib
```
