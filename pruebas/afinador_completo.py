import tensorflow as tf
import numpy as np
import pyaudio
# import aubio
import librosa

# --- Cargar modelo TFLite ---
interpreter = tf.lite.Interpreter(model_path="Models/soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("Models/labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# --- Configurar audio ---
SAMPLE_RATE = 44100
CHUNK = 44032

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=SAMPLE_RATE,
                 input=True,
                 frames_per_buffer=CHUNK)

# --- Configurar aubio (detector de pitch) ---
WIN_S = 4096
HOP_S = 512
# pitch_o = aubio.pitch("yinfft", WIN_S, HOP_S, SAMPLE_RATE)
# pitch_o.set_unit("Hz")
#pitch_o.set_tolerance(0.8)

REFERENCIA_HZ = 440.0  # cuerda La

def calcular_frecuencia(audio_buffer):
    """Detecta la frecuencia fundamental usando librosa.pyin sobre el buffer completo."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_buffer,
        fmin=librosa.note_to_hz('C3'),
        fmax=librosa.note_to_hz('C6'),
        sr=SAMPLE_RATE
    )
    f0_validas = f0[voiced_flag]
    if len(f0_validas) > 0:
        return float(np.nanmedian(f0_validas))
    return None

def calcular_cents(freq_detectada, freq_referencia):
    if freq_detectada is None or freq_detectada <= 0:
        return None
    return 1200 * np.log2(freq_detectada / freq_referencia)

try:
    while True:
        raw = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.float32)

        # --- Clasificación con Teachable Machine ---
        audio_input = audio.reshape(1, CHUNK).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], audio_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = np.argmax(output)
        clase = labels[idx]
        confianza = output[idx] * 100

        if clase == "Background Noise":
            print(f"{clase}: {confianza:.1f}%")
        else:
            # Hay nota sonando -> calcular cents con aubio
            freq = calcular_frecuencia(audio)
            cents = calcular_cents(freq, REFERENCIA_HZ)

            if freq is not None:
                print(f"{clase} ({confianza:.1f}%) | Frecuencia: {freq:.1f} Hz | Desviacion: {cents:+.1f} cents")
            else:
                print(f"{clase} ({confianza:.1f}%) | No se pudo detectar pitch claro")

except KeyboardInterrupt:
    print("\nDetenido.")
    stream.stop_stream()
    stream.close()
    p.terminate()