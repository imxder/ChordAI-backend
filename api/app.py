from flask import Flask, request, jsonify
import os
import io
import librosa

from train_model import load_trained_model, recognize_chord, SAMPLE_RATE, N_MFCC, MAX_PAD_LEN

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB por chunk de audio recebido

model_loaded = None
encoder_loaded = None
scaler_loaded = None


with app.app_context():
    print("Tentando carregar modelo de ACORDES na inicialização do Flask...")
    try:
        model_loaded, encoder_loaded, scaler_loaded = load_trained_model()
        if model_loaded is None:
            print("AVISO: Modelo de acordes não foi carregado. Verifique o dataset e o treinamento.")
            print("Para treinar o modelo, execute: python train_model.py")
        else:
            print("Modelo de acordes, encoder e scaler carregados com sucesso!")
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar o modelo de acordes na inicialização: {e}")
        print("Certifique-se de que o modelo de acordes foi treinado e os arquivos .joblib estão na pasta 'trained_model'.")
        model_loaded = None 

@app.route('/predict_chord', methods=['POST'])
def predict_chord():
    """Endpoint para o app Flutter: recebe um chunk de audio gravado no
    cliente (campo 'audio' em multipart/form-data) e retorna o acorde
    reconhecido. Grave chunks de ~RECORD_DURATION segundos em WAV -
    outros formatos (mp3/m4a) podem falhar aqui pois este ambiente nao
    tem ffmpeg instalado para decodifica-los."""
    if model_loaded is None or encoder_loaded is None or scaler_loaded is None:
        return jsonify(status='error', message='Modelo de reconhecimento não carregado.'), 503

    if 'audio' not in request.files:
        return jsonify(status='error', message="Campo 'audio' ausente na requisição (multipart/form-data)."), 400

    audio_file = request.files['audio']
    try:
        audio_np, _ = librosa.load(io.BytesIO(audio_file.read()), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return jsonify(status='error', message=f'Não foi possível decodificar o áudio: {e}'), 400

    result = recognize_chord(audio_np, model_loaded, encoder_loaded, scaler_loaded,
                              sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN)
    return jsonify(result)

if __name__ == '__main__':
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('dataset', exist_ok=True) 
    
    # debug so liga com FLASK_DEBUG=1: o modo debug mostra o codigo e deixa rodar
    # comandos pela pagina de erro.
    # host 127.0.0.1: a API so atende o proprio computador. O app nao usa mais
    # a API, entao ela nao precisa ficar aberta pra rede.
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, threaded=True, host='127.0.0.1')