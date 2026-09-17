import tensorflow as tf
import numpy as np
import pyaudio

# --- Cargar modelo ---
interpreter = tf.lite.Interpreter(model_path="Models/soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Cargar labels ---
with open("Models/labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# --- Configurar audio ---
SAMPLE_RATE = 44100
CHUNK = 44032  # el tamaño exacto que pide el modelo

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=SAMPLE_RATE,
                 input=True,
                 frames_per_buffer=CHUNK)

print("Escuchando... (Ctrl+C para detener)")

try:
    while True:
        raw = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.float32)

        # Dar forma [1, 44032] como pide el modelo
        audio_input = audio.reshape(1, CHUNK).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], audio_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = np.argmax(output)
        confianza = output[idx] * 100
        print(f"{labels[idx]}: {confianza:.1f}%")

except KeyboardInterrupt:
    print("\nDetenido.")
    stream.stop_stream()
    stream.close()
    p.terminate()