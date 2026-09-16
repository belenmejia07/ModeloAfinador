import aubio
import numpy as np
import cv2

# Prueba 1: aubio detecta pitch
p = aubio.pitch("yinfft", 4096, 512, 44100)
p.set_unit("Hz")
sr = 44100
t = np.arange(512) / sr
signal = (0.5 * np.sin(2*np.pi*440*t)).astype(np.float32)
pitch = p(signal)[0]
print(f"[OK] aubio detecto: {pitch:.2f} Hz (deberia ser cercano a 440)")

# Prueba 2: camara abre correctamente
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("[OK] camara detectada")
    cap.release()
else:
    print("[ERROR] no se pudo abrir la camara")

# Prueba 3: tensorflow carga
import tensorflow as tf
print(f"[OK] tensorflow version: {tf.__version__}")