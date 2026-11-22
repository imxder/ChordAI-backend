import os
import sys

# roda o servidor sem janela (pythonw), entao a saida vai para um arquivo.
# BASE_DIR e a raiz do repositorio (a pasta acima de api/): o train_model.py
# procura o trained_model/ a partir dela.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
log = open(os.path.join(BASE_DIR, 'server.log'), 'a', buffering=1, encoding='utf-8')
sys.stdout = log
sys.stderr = log

from waitress import serve
from api.app import app

if __name__ == '__main__':
    print("Servidor iniciado na porta 5000")
    # so o proprio computador acessa (o app nao usa mais a API)
    serve(app, host='127.0.0.1', port=5000, threads=8)
