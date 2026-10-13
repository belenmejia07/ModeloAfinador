import tensorflow as tf
import numpy as np
import pyaudio
import librosa

# --- Cargar modelo TFLite ---
# Ajusta la ruta segun donde coloques este script (raiz del proyecto o carpeta pruebas/)
interpreter = tf.lite.Interpreter(model_path="../Model2/soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("../Model2/labels.txt", "r") as f:
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

# --- Referencias de las 4 cuerdas del violin ---
REFERENCIAS = {
    "Sol": 196.00,
    "Re": 293.66,
    "La": 440.00,
    "Mi": 659.25,
}

UMBRAL_PRECISION_AFINADO = 70  # equivale a 15 cents o menos de diferencia


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


def encontrar_cuerda_mas_cercana(freq_detectada):
    """Encuentra la cuerda cuya frecuencia esta mas cerca en escala de cents (no en Hz)."""
    mejor_cuerda = None
    menor_diferencia_cents = float("inf")
    for cuerda, freq_ref in REFERENCIAS.items():
        diferencia = abs(1200 * np.log2(freq_detectada / freq_ref))
        if diferencia < menor_diferencia_cents:
            menor_diferencia_cents = diferencia
            mejor_cuerda = cuerda
    return mejor_cuerda, REFERENCIAS[mejor_cuerda], menor_diferencia_cents


def corregir_error_de_octava(freq_detectada):
    """librosa.pyin a veces detecta el doble o la mitad de la frecuencia real
    (error de octava/subarmonico), mas comun en transiciones entre notas.
    Como solo hay 4 frecuencias validas (las cuerdas del violin), probamos
    freq, freq*2 y freq/2 contra las referencias y nos quedamos con la que
    mas se acerque a una cuerda conocida."""
    candidatos = [freq_detectada, freq_detectada * 2, freq_detectada / 2]
    mejor_freq = freq_detectada
    menor_diferencia = float("inf")
    for candidato in candidatos:
        _, _, diferencia = encontrar_cuerda_mas_cercana(candidato)
        if diferencia < menor_diferencia:
            menor_diferencia = diferencia
            mejor_freq = candidato
    return mejor_freq


def calcular_cents(freq_detectada, freq_referencia):
    if freq_detectada is None or freq_detectada <= 0:
        return None
    return 1200 * np.log2(freq_detectada / freq_referencia)


def calcular_precision(cents):
    """Convierte la desviacion en cents a un porcentaje de precision.
    100% = perfectamente afinado (0 cents), 0% = 50+ cents de diferencia."""
    if cents is None:
        return None
    abs_cents = min(abs(cents), 50)
    return max(0, 100 - abs_cents * 2)


def corregir_clasificacion(clase_tm, precision):
    """Corrige la clasificacion de Teachable Machine usando la precision de librosa.
    Si la precision (medicion exacta) es alta, se considera afinado sin importar
    lo que haya dicho el clasificador entrenado."""
    if precision is None:
        return clase_tm
    if precision >= UMBRAL_PRECISION_AFINADO:
        return "Afinada"
    else:
        return "Desafinada"


try:
    print("Escuchando... (Ctrl+C para detener)")
    while True:
        raw = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(raw, dtype=np.float32)

        # --- Clasificacion inicial con Teachable Machine ---
        audio_input = audio.reshape(1, CHUNK).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], audio_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = np.argmax(output)
        clase_tm = labels[idx]

        if clase_tm == "Background Noise":
            print(f"{clase_tm}")
        else:
            freq = calcular_frecuencia(audio)

            if freq is not None:
                freq = corregir_error_de_octava(freq)
                cuerda_detectada, referencia, _ = encontrar_cuerda_mas_cercana(freq)
                cents = calcular_cents(freq, referencia)
                precision = calcular_precision(cents)
                estado_final = corregir_clasificacion(clase_tm, precision)

                print(f"{estado_final} | Cuerda: {cuerda_detectada} | Frecuencia: {freq:.1f} Hz | "
                      f"Precision: {precision:.0f}% | Desviacion: {cents:+.1f} cents")
            else:
                print(f"{clase_tm} | No se pudo detectar pitch claro")

except KeyboardInterrupt:
    print("\nDetenido.")
    stream.stop_stream()
    stream.close()
    p.terminate()