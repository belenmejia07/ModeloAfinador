"""Genera WAV sinteticos (tonos tipo violin) SOLO para probar que el pipeline funciona.
No son datos reales: no los uses para los resultados del articulo."""
import os, sys
import numpy as np
from scipy.io import wavfile

SR = 44100
FREQ = {"Sol3": 196.00, "La3": 220.00, "Si3": 246.94, "Do4": 261.63, "Re4": 293.66, "Mi4": 329.63,
        "Fa4": 349.23, "Sol4": 392.00, "La4": 440.00, "Si4": 493.88, "Do5": 523.25, "Re5": 587.33, "Mi5": 659.26}


def tono(f0, rng, dur=3.0, cents=0.0, fund=None):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * 2 ** (cents / 1200)
    vib = 1 + (rng.uniform(0, 6) / 1200 * np.log(2)) * np.sin(2 * np.pi * rng.uniform(5, 6) * t)
    fase = 2 * np.pi * np.cumsum(f * vib) / SR
    p = rng.uniform(0.3, 1.5)
    g1 = rng.uniform(0.4, 1.0) if fund is None else fund
    x = np.zeros_like(t)
    for k in range(1, 7):
        a = k ** -p * rng.lognormal(0, 0.4) * (g1 if k == 1 else 1)
        x += a * np.sin(k * fase + rng.uniform(0, 2 * np.pi))
    env = np.minimum(1, t / 0.05) * np.minimum(1, (dur - t) / 0.2)
    x *= env
    snr = rng.uniform(10, 30)
    ruido = rng.standard_normal(len(t)); ruido = ruido + np.cumsum(rng.standard_normal(len(t))) * 0.0005
    ruido *= np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10)) / np.std(ruido)
    y = x + ruido
    return (y / np.max(np.abs(y)) * rng.uniform(0.2, 0.9)).astype(np.float32)


def ruido(rng, dur=3.0):
    n = int(dur * SR); y = rng.standard_normal(n)
    espectro = np.fft.rfft(y); f = np.fft.rfftfreq(n, 1 / SR); f[0] = 1
    y = np.fft.irfft(espectro / f ** rng.uniform(0, 1.0), n)
    if rng.random() < 0.5:
        hz = rng.choice([50, 60]); t = np.arange(n) / SR
        y = y / np.std(y) + rng.uniform(0.2, 1) * sum(np.sin(2 * np.pi * hz * k * t) / k for k in (1, 2, 3))
    return (y / np.max(np.abs(y)) * rng.uniform(0.02, 0.5)).astype(np.float32)


def guardar(ruta, x):
    wavfile.write(ruta, SR, (x * 32767).astype(np.int16))


if __name__ == "__main__":
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "Dataset_Sintetico"
    os.makedirs(carpeta, exist_ok=True)
    rng = np.random.default_rng(7)
    for nota, f0 in FREQ.items():
        for toma in range(15):                       # 3 condiciones x 5 tomas
            cents = [0.0, rng.uniform(-35, -20), rng.uniform(20, 35)][toma % 3]
            guardar(f"{carpeta}/{nota}_{int(cents):+d}_{toma + 1:02d}.wav", tono(f0, rng, cents=cents))
    for toma in range(30):
        guardar(f"{carpeta}/ruido_0_{toma + 1:02d}.wav", ruido(rng))
    print("WAV generados en", carpeta, "->", len(os.listdir(carpeta)))
