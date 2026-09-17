import threading
import cv2
import numpy as np
import pyaudio
import librosa
import tensorflow as tf

# ---------- Configuracion ----------
SAMPLE_RATE = 44100
CHUNK = 44032
REFERENCIA_HZ = 440.0  # cuerda La

MODEL_PATH = "Models/soundclassifier_with_metadata.tflite"
LABELS_PATH = "Models/labels.txt"

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
    "confianza": 0.0,
    "frecuencia": None,
    "cents": None,
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


def calcular_cents(freq_detectada, freq_referencia):
    if freq_detectada is None or freq_detectada <= 0:
        return None
    return 1200 * np.log2(freq_detectada / freq_referencia)


def hilo_audio():
    """Corre en paralelo: escucha el microfono, clasifica y calcula cents."""
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

            # --- Clasificacion con Teachable Machine ---
            audio_input = audio.reshape(1, CHUNK).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], audio_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            idx = np.argmax(output)
            clase = labels[idx]
            confianza = output[idx] * 100

            # --- Calculo de cents (solo si hay nota sonando) ---
            freq = None
            cents = None
            if clase != "Background Noise":
                freq = calcular_frecuencia(audio)
                cents = calcular_cents(freq, REFERENCIA_HZ)

            with estado_lock:
                estado["clase"] = clase
                estado["confianza"] = confianza
                estado["frecuencia"] = freq
                estado["cents"] = cents

        except Exception as e:
            print(f"ADVERTENCIA en hilo de audio: {e}")
            continue

    stream.stop_stream()
    stream.close()
    p.terminate()


# ---------- Color segun clase y cents ----------
def color_por_clase_y_cents(clase, cents):
    if clase == "Background Noise" or cents is None:
        return (200, 200, 200)  # gris: sin nota sonando

    abs_c = min(abs(cents), 50)

    if clase == "BienAfinada":
        # Tonos de verde (BGR): mas brillante cuanto mas cerca de 0 cents
        intensidad = int(255 - (abs_c / 50) * 155)
        return (0, intensidad, 0)
    else:
        # MalAfinada -> tonos de rojo (BGR): mas intenso cuanto mas lejos de 0 cents
        intensidad_roja = int(100 + (abs_c / 50) * 155)
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
                confianza = estado["confianza"]
                freq = estado["frecuencia"]
                cents = estado["cents"]

            color = color_por_clase_y_cents(clase, cents)

            cv2.rectangle(frame, (30, frame.shape[0] - 80),
                          (frame.shape[1] - 30, frame.shape[0] - 30), color, -1)

            texto_clase = f"{clase} ({confianza:.0f}%)"
            cv2.putText(frame, texto_clase, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if freq is not None and cents is not None:
                texto_info = f"{freq:.1f} Hz | {cents:+.1f} cents"
                cv2.putText(frame, texto_info, (30, frame.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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