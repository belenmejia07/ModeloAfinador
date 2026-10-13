import threading
import cv2
import numpy as np
import pyaudio
import librosa
import tensorflow as tf

# ---------- Configuracion ----------
SAMPLE_RATE = 44100
CHUNK = 44032

MODEL_PATH = "Model2/soundclassifier_with_metadata.tflite"
LABELS_PATH = "Model2/labels.txt"

# --- Referencias de las 4 cuerdas del violin ---
REFERENCIAS = {
    "Sol": 196.00,
    "Re": 293.66,
    "La": 440.00,
    "Mi": 659.25,
}

UMBRAL_PRECISION_AFINADO = 70  # equivale a 15 cents o menos de diferencia

# ---------- Cargar modelo ----------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# ---------- Estado compartido entre hilos ----------
estado_lock = threading.Lock()
estado = {
    "clase": "Esperando...",
    "cuerda": None,
    "frecuencia": None,
    "cents": None,
    "precision": None,
}
detener = threading.Event()


# ---------- Funciones de audio ----------
def calcular_frecuencia(audio_buffer):
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


def hilo_audio():
    """Corre en paralelo: escucha el microfono, clasifica y corrige con precision."""
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=SAMPLE_RATE,
                         input=True,
                         frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"ERROR: No se pudo abrir el microfono: {e}")
        detener.set()
        return

    while not detener.is_set():
        try:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(raw, dtype=np.float32)

            # --- Clasificacion inicial con Teachable Machine ---
            audio_input = audio.reshape(1, CHUNK).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], audio_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            idx = np.argmax(output)
            clase_tm = labels[idx]

            cuerda = None
            freq = None
            cents = None
            precision = None
            clase_final = clase_tm

            if clase_tm != "Background Noise":
                freq = calcular_frecuencia(audio)
                if freq is not None:
                    freq = corregir_error_de_octava(freq)
                    cuerda, referencia, _ = encontrar_cuerda_mas_cercana(freq)
                    cents = calcular_cents(freq, referencia)
                    precision = calcular_precision(cents)
                    clase_final = corregir_clasificacion(clase_tm, precision)

            with estado_lock:
                estado["clase"] = clase_final
                estado["cuerda"] = cuerda
                estado["frecuencia"] = freq
                estado["cents"] = cents
                estado["precision"] = precision

        except Exception as e:
            print(f"ADVERTENCIA en hilo de audio: {e}")
            continue

    stream.stop_stream()
    stream.close()
    p.terminate()


# ---------- Color segun clase y precision ----------
def color_por_clase_y_precision(clase, precision):
    if clase == "Background Noise" or precision is None:
        return (200, 200, 200)  # gris: sin nota sonando

    if clase == "Afinada":
        # Tonos de verde (BGR): mas brillante cuanto mayor la precision
        intensidad = int(100 + (precision / 100) * 155)
        return (0, intensidad, 0)
    else:
        # Desafinada -> tonos de rojo (BGR): mas intenso cuanto menor la precision
        intensidad_roja = int(100 + (100 - precision) / 100 * 155)
        return (0, 0, intensidad_roja)


# ---------- Hilo principal: camara ----------
def main():
    hilo = threading.Thread(target=hilo_audio, daemon=True)
    hilo.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la camara. Verifica que no este siendo usada por otra aplicacion.")
        detener.set()
        return

    print("Camara y microfono listos. Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ADVERTENCIA: No se pudo leer un frame de la camara, reintentando...")
                continue

            with estado_lock:
                clase = estado["clase"]
                cuerda = estado["cuerda"]
                freq = estado["frecuencia"]
                cents = estado["cents"]
                precision = estado["precision"]

            color = color_por_clase_y_precision(clase, precision)

            cv2.rectangle(frame, (30, frame.shape[0] - 80),
                          (frame.shape[1] - 30, frame.shape[0] - 30), color, -1)

            if cuerda is not None:
                texto_clase = f"{clase} - Cuerda {cuerda}"
            else:
                texto_clase = clase
            cv2.putText(frame, texto_clase, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if freq is not None and cents is not None and precision is not None:
                texto_info = f"{freq:.1f} Hz | {cents:+.1f} cents | Precision: {precision:.0f}%"
                cv2.putText(frame, texto_info, (30, frame.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imshow("Afinador de Violin", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    except Exception as e:
        print(f"ERROR inesperado: {e}")
    finally:
        print("Cerrando camara y microfono...")
        detener.set()
        hilo.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()