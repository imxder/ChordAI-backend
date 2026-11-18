"""Áudio sintético pros testes: acordes feitos de senoides, sem precisar do dataset.

Cada nota tem a fundamental e dois harmônicos mais fracos, com um decaimento
leve, parecido com uma corda tocada. Dá pra conferir o chroma porque a gente sabe
exatamente quais notas estão no sinal.
"""
import os

import numpy as np
import soundfile as sf

# frequências (Hz) das notas de cada acorde, na região do violão
ACORDES = {
    'C': [130.81, 164.81, 196.00, 261.63, 329.63],   # C3 E3 G3 C4 E4
    'G': [98.00, 123.47, 146.83, 196.00, 246.94],    # G2 B2 D3 G3 B3
    'Am': [110.00, 164.81, 220.00, 261.63, 329.63],  # A2 E3 A3 C4 E4
    'Em': [82.41, 123.47, 164.81, 196.00, 246.94],   # E2 B2 E3 G3 B3
}

# índice no chroma (0 = C, 1 = C#, ..., 11 = B) das notas de cada acorde
NOTAS_CHROMA = {'C': {0, 4, 7}, 'G': {7, 11, 2}, 'Am': {9, 0, 4}, 'Em': {4, 7, 11}}


def acorde(nome, segundos=1.5, sr=22050, volume=0.3, seed=0):
    t = np.arange(int(sr * segundos)) / sr
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    for f in ACORDES[nome]:
        fase = rng.uniform(0, 2 * np.pi)
        for h, peso in ((1, 1.0), (2, 0.4), (3, 0.2)):
            x += peso * np.sin(2 * np.pi * f * h * t + fase)
    x *= np.exp(-t / 2.0)
    x += 0.01 * rng.standard_normal(len(t))  # um pouco de ruído, como numa gravação
    return (volume * x / np.max(np.abs(x))).astype(np.float32)


def gravar_dataset(pasta, acordes=('C', 'G', 'Am', 'Em'), por_acorde=10, segundos=3.0, sr=22050):
    """Monta uma pasta no formato do dataset/ (uma subpasta por acorde, .wav dentro)."""
    for nome in acordes:
        os.makedirs(os.path.join(pasta, nome), exist_ok=True)
        for i in range(por_acorde):
            x = acorde(nome, segundos, sr, volume=0.2 + 0.02 * i, seed=i)
            sf.write(os.path.join(pasta, nome, f'{nome}_{i:02d}.wav'), x, sr)
    return pasta
