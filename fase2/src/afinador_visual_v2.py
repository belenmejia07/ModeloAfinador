"""
Afinador de violin v2 (fase 2): modelo PROPIO (MLP) + FFT con interpolacion parabolica.

El MLP identifica la NOTA; la FFT mide los CENTS contra la frecuencia de esa nota.
Estado: Afinada si precision >= 70 (|cents| <= 15).

Uso:
  python afinador_visual_v2.py --dispositivo 1       # tiempo real con camara
  python afinador_visual_v2.py --sin-camara          # solo consola
  python afinador_visual_v2.py --wav toma.wav        # analizar un archivo
"""
import os, sys, time, argparse, threading
import numpy as np
import entrenar_modelo_notas as E

SR = E.SR
HOP = SR // 2
UMBRAL_PROB = 0.60
UMBRAL_PRECISION = 70
UMBRAL_ENERGIA = 0.003   # RMS minimo para considerar que hay sonido

HZ = {"Sol3": 196.00, "La3": 220.00, "Si3": 246.94, "Do4": 261.63, "Re4": 293.66,
      "Mi4": 329.63, "Fa4": 349.23, "Sol4": 392.00, "La4": 440.00, "Si4": 493.88,
      "Do5": 523.25, "Re5": 587.33, "Mi5": 659.25}


def cargar_modelo(ruta=None):
    import tensorflow as tf
    for r in ([ruta] if ruta else []) + ["modelos/modelo_notas.keras", "salida_modelo/modelo_notas.keras"]:
        if r and os.path.exists(r):
            return tf.keras.models.load_model(r)
    sys.exit("No encuentro modelo_notas.keras (usa --modelo ruta)")


def cents_fft(x, f_ref, armonicos=(1, 2, 3, 4), ancho_cents=100, claridad=3.0):
    """Cents respecto a f_ref usando FFT (Hann, zero-padding x4, interpolacion parabolica)
    sobre varios armonicos; promedio ponderado por magnitud. None si no hay pico claro."""
    x = np.asarray(x[:E.VENTANA], np.float64)
    x = x - x.mean()
    N = len(x); P = 4 * N
    mag = np.abs(np.fft.rfft(x * np.hanning(N), P))
    fr = np.fft.rfftfreq(P, 1 / SR)
    vals, pesos = [], []
    for k in armonicos:
        fk = f_ref * k
        if fk > SR / 2 - 200:
            break
        lo = np.searchsorted(fr, fk * 2 ** (-ancho_cents / 1200))
        hi = np.searchsorted(fr, fk * 2 ** (ancho_cents / 1200))
        seg = mag[lo:hi]
        if len(seg) < 3:
            continue
        i = int(seg.argmax()) + lo
        if i < 1 or i >= len(mag) - 1 or mag[i] < claridad * np.median(seg):
            continue
        a, b, c = np.log(mag[i - 1] + 1e-12), np.log(mag[i] + 1e-12), np.log(mag[i + 1] + 1e-12)
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
        f = (i + d) * SR / P
        vals.append(1200 * np.log2(f / fk)); pesos.append(mag[i])
    if not vals:
        return None, None
    c = float(np.average(vals, weights=pesos))
    return c, f_ref * 2 ** (c / 1200)


def precision(c):
    return max(0.0, 100 - 2 * min(abs(c), 50))


def analizar_ventana(x, modelo):
    r = {"nota": None, "prob": 0.0, "cents": None, "precision": 0.0,
         "estado": "Sin nota", "f_est": None}
    if np.sqrt(np.mean(np.asarray(x) ** 2)) < UMBRAL_ENERGIA:
        return r
    p = modelo.predict(E.caracteristicas(x)[None], verbose=0)[0]
    i = int(p.argmax())
    r["prob"] = float(p[i])
    if E.CLASES[i] == "ruido" or p[i] < UMBRAL_PROB:
        r["estado"] = "No reconocida"
        return r
    nota = E.CLASES[i]
    c, f = cents_fft(x, HZ[nota])
    r["nota"] = nota
    if c is None:
        r["estado"] = "No reconocida"
        return r
    r.update(cents=c, precision=precision(c), f_est=f)
    r["estado"] = "Afinada" if r["precision"] >= UMBRAL_PRECISION else "Desafinada"
    return r


def linea(r):
    if r["cents"] is None:
        return f"{r['estado']}"
    s = "+" if r["cents"] >= 0 else ""
    return f"{r['nota']}  {s}{r['cents']:.1f} cents  precision {r['precision']:.0f}%  -> {r['estado']}"


def procesar_wav(ruta, modelo):
    x = E.cargar_wav(ruta)
    for i in range(0, max(1, len(x) - E.VENTANA + 1), HOP):
        t = i / SR
        print(f"[{t:5.1f}s] {linea(analizar_ventana(x[i:i + E.VENTANA], modelo))}")


def color(r):
    """BGR: verde = afinada, rojo = muy desafinada, degradado por precision."""
    if r["cents"] is None:
        return (128, 128, 128)
    p = r["precision"] / 100
    return (0, int(255 * p), int(255 * (1 - p)))


def tiempo_real(modelo, con_camara=True, dispositivo=1):
    import pyaudio
    buf = np.zeros(E.VENTANA, np.float32)
    lock = threading.Lock()
    estado = {"r": {"nota": None, "prob": 0.0, "cents": None, "precision": 0.0,
                    "estado": "Escuchando...", "f_est": None}, "t": 0.0}
    parar = threading.Event()

    def hilo_audio():
        nonlocal buf
        pa = pyaudio.PyAudio()
        try:
            s = pa.open(format=pyaudio.paInt16, channels=1, rate=SR, input=True,
                        input_device_index=dispositivo, frames_per_buffer=2048)
        except Exception as e:
            print("No pude abrir el microfono:", e, "(prueba --dispositivo N)")
            parar.set(); return
        nuevas = 0
        while not parar.is_set():
            x = np.frombuffer(s.read(2048, exception_on_overflow=False), np.int16).astype(np.float32) / 32768
            with lock:
                buf = np.concatenate([buf, x])[-E.VENTANA:]
            nuevas += len(x)
            if nuevas >= HOP:
                nuevas = 0
                with lock:
                    w = buf.copy()
                t0 = time.time()
                r = analizar_ventana(w, modelo)
                estado["r"], estado["t"] = r, (time.time() - t0) * 1000
                print(linea(r), f"({estado['t']:.0f} ms)")
        s.stop_stream(); s.close(); pa.terminate()

    th = threading.Thread(target=hilo_audio, daemon=True)
    th.start()
    try:
        if not con_camara:
            while not parar.is_set():
                time.sleep(0.5)
        else:
            import cv2
            cap = cv2.VideoCapture(0)
            while not parar.is_set():
                ok, fr = cap.read()
                if not ok:
                    break
                fr = cv2.flip(fr, 1)
                r = estado["r"]
                col = color(r)
                cv2.rectangle(fr, (0, 0), (fr.shape[1] - 1, fr.shape[0] - 1), col, 14)
                cv2.putText(fr, r["nota"] or "--", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, col, 5)
                txt = r["estado"] if r["cents"] is None else f"{r['cents']:+.1f} cents | {r['precision']:.0f}% | {r['estado']}"
                cv2.putText(fr, txt, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                cv2.imshow("Afinador de violin v2 (q = salir)", fr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release(); cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
    parar.set()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo"); ap.add_argument("--wav")
    ap.add_argument("--sin-camara", action="store_true")
    ap.add_argument("--dispositivo", type=int, default=1)
    a = ap.parse_args()
    m = cargar_modelo(a.modelo)
    if a.wav:
        procesar_wav(a.wav, m)
    else:
        tiempo_real(m, not a.sin_camara, a.dispositivo)
