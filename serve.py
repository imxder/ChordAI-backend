import os
import sys

# roda o servidor sem janela (pythonw), entao a saida vai para um arquivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
log = open(os.path.join(BASE_DIR, 'server.log'), 'a', buffering=1, encoding='utf-8')
sys.stdout = log
sys.stderr = log

from waitress import serve
from app import app

if __name__ == '__main__':
    print("Servidor iniciado na porta 5000")
    serve(app, host='0.0.0.0', port=5000, threads=8)
