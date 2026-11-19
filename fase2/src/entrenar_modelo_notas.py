"""
Entrena un clasificador propio de notas del violin a partir de archivos WAV.

Carpeta de datos (un WAV por toma, nombre: NOTA_CENTS_TOMA.wav):
    Sol3_0_01.wav   La4_-20_03.wav   Mi5_+25_02.wav   ruido_0_05.wav
Clases: Sol3 La3 Si3 Do4 Re4 Mi4 Fa4 Sol4 La4 Si4 Do5 Re5 Mi5 y "ruido" (sin nota).

Uso:  python entrenar_modelo_notas.py Dataset_Notas
"""
import os, sys, json, glob, time
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ---------------- Configuracion ----------------
SR = 44100                    # frecuencia de muestreo (Hz)
VENTANA = SR                  # 1 segundo = 44100 muestras
SALTO = SR // 2               # solapamiento del 50 % entre ventanas
RECORTE = int(0.25 * SR)      # se descartan 0,25 s del ataque y de la cola de cada toma
F_MIN, F_MAX = 180.0, 3000.0  # rango de frecuencias que ve el modelo (fundamental + armonicos)
CENTS_POR_BIN = 25            # resolucion de las bandas logaritmicas
SEMILLA = 42
NOTAS = ["Sol3", "La3", "Si3", "Do4", "Re4", "Mi4", "Fa4", "Sol4", "La4", "Si4", "Do5", "Re5", "Mi5"]
CLASES = NOTAS + ["ruido"]

N_BINS = int(np.ceil(1200 * np.log2(F_MAX / F_MIN) / CENTS_POR_BIN))
_BORDES = F_MIN * 2.0 ** (np.arange(N_BINS + 1) * CENTS_POR_BIN / 1200)
_FREQS = np.fft.rfftfreq(VENTANA, 1 / SR)
_IDX = np.searchsorted(_FREQS, _BORDES)
_HANN = np.hanning(VENTANA)


def caracteristicas(x):
    """Ventana de audio (VENTANA muestras) -> vector de N_BINS valores entre 0 y 1."""
    mag = np.abs(np.fft.rfft(x * _HANN))                      # Hann + FFT
    banda = np.maximum.reduceat(mag[:_IDX[-1]], _IDX[:-1])    # maximo por banda logaritmica
    db = 20 * np.log10(banda + 1e-9)                          # a decibelios
    db = np.clip(db - db.max(), -80, 0)                       # el pico mas fuerte = 0 dB
    return ((db + 80) / 80).astype(np.float32)                # escala 0..1 (independiente del volumen)


def cargar_wav(ruta):
    sr, x = wavfile.read(ruta)
    if sr != SR:
        raise ValueError(f"{ruta}: frecuencia {sr} Hz; se esperaban {SR} Hz")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768
    if x.dtype == np.int32:
        return x.astype(np.float32) / 2 ** 31
    return x.astype(np.float32)


def ventanas_de_toma(x, es_ruido):
    if not es_ruido:
        x = x[RECORTE:-RECORTE]
    for ini in range(0, len(x) - VENTANA + 1, SALTO):
        yield x[ini:ini + VENTANA]


def clase_de(ruta):
    nombre = os.path.basename(ruta).split("_")[0]
    return nombre if nombre in CLASES else None


def dividir_por_toma(rutas, rng):
    """60/20/20 por toma (no por ventana) y por clase, para evitar fuga de datos."""
    por_clase = {c: [] for c in CLASES}
    for r in rutas:
        c = clase_de(r)
        if c is None:
            print("AVISO: nombre de clase desconocido, se ignora:", r)
        else:
            por_clase[c].append(r)
    reparto = {"train": [], "val": [], "test": []}
    for c, lista in por_clase.items():
        if len(lista) < 3:
            raise SystemExit(f"La clase {c} tiene {len(lista)} tomas; se necesitan al menos 3 (ideal: 10 o mas).")
        rng.shuffle(lista)
        n_t = max(1, round(0.2 * len(lista)))
        n_v = max(1, round(0.2 * len(lista)))
        reparto["test"] += lista[:n_t]
        reparto["val"] += lista[n_t:n_t + n_v]
        reparto["train"] += lista[n_t + n_v:]
    return reparto


def construir_conjunto(rutas):
    X, y = [], []
    for r in rutas:
        c = clase_de(r)
        for v in ventanas_de_toma(cargar_wav(r), c == "ruido"):
            X.append(caracteristicas(v))
            y.append(CLASES.index(c))
    return np.array(X), np.array(y)


def con_ruido(x, rng):
    """Aumento de datos: ruido blanco con relacion senal/ruido aleatoria (10 a 35 dB)."""
    snr = rng.uniform(10, 35)
    ruido = rng.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))
    return (x + ruido).astype(np.float32)


def construir_conjunto_train(rutas, rng, copias=2):
    X, y = [], []
    for r in rutas:
        c = clase_de(r)
        for v in ventanas_de_toma(cargar_wav(r), c == "ruido"):
            X.append(caracteristicas(v)); y.append(CLASES.index(c))
            for _ in range(copias):
                X.append(caracteristicas(con_ruido(v, rng))); y.append(CLASES.index(c))
    return np.array(X), np.array(y)


def mlp(n_entrada, n_clases, l2=1e-4):
    reg = keras.regularizers.l2(l2)
    return keras.Sequential([
        keras.layers.Input((n_entrada,)),
        keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(n_clases, activation="softmax"),
    ])


def regresion_softmax(n_entrada, n_clases):
    """Linea base sin capas ocultas (regresion logistica multiclase)."""
    return keras.Sequential([keras.layers.Input((n_entrada,)),
                             keras.layers.Dense(n_clases, activation="softmax")])


def entrenar(modelo, Xtr, ytr, Xva, yva):
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    parar = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    return modelo.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=300, batch_size=32,
                      callbacks=[parar], verbose=0)


def evaluar(nombre, modelo, X, y):
    pred = np.argmax(modelo.predict(X, verbose=0), axis=1)
    acc = float(np.mean(pred == y))
    f1 = float(f1_score(y, pred, average="macro"))
    print(f"{nombre:<22} exactitud = {acc:.3f} | F1 macro = {f1:.3f}")
    return pred, acc, f1


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "Dataset_Notas"
    os.makedirs("salida_modelo", exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    tf.random.set_seed(SEMILLA)

    rutas = sorted(glob.glob(os.path.join(carpeta, "*.wav")))
    if not rutas:
        raise SystemExit(f"No hay archivos .wav en {carpeta}")
    reparto = dividir_por_toma(rutas, rng)
    print({k: len(v) for k, v in reparto.items()}, "tomas por conjunto")

    Xtr, ytr = construir_conjunto_train(reparto["train"], rng)
    Xva, yva = construir_conjunto(reparto["val"])
    Xte, yte = construir_conjunto(reparto["test"])
    print(f"ventanas -> train {len(Xtr)} (con aumento) | val {len(Xva)} | test {len(Xte)} | entradas por ventana {N_BINS}")

    base = regresion_softmax(N_BINS, len(CLASES)); entrenar(base, Xtr, ytr, Xva, yva)
    red = mlp(N_BINS, len(CLASES)); h = entrenar(red, Xtr, ytr, Xva, yva)

    print("\n--- Resultados en TEST (se mira una sola vez) ---")
    evaluar("Regresion softmax", base, Xte, yte)
    pred, acc, f1 = evaluar("MLP (128-64)", red, Xte, yte)
    print(f"MLP: perdida train = {h.history['loss'][-1]:.3f} | perdida val = {min(h.history['val_loss']):.3f} "
          f"| epocas usadas = {len(h.history['loss'])}")

    # matriz de confusion y confusiones principales
    cm = confusion_matrix(yte, pred, labels=range(len(CLASES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(len(CLASES))); ax.set_yticks(range(len(CLASES)))
    ax.set_xticklabels(CLASES, rotation=60); ax.set_yticklabels(CLASES)
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusion (test)")
    fig.tight_layout(); fig.savefig("salida_modelo/matriz_confusion.png", dpi=170); plt.close(fig)
    conf = [(CLASES[i], CLASES[j], int(cm[i, j])) for i in range(len(CLASES)) for j in range(len(CLASES)) if i != j and cm[i, j]]
    print("Confusiones (real -> predicha, n):", sorted(conf, key=lambda t: -t[2])[:8] or "ninguna")

    # curvas de aprendizaje (diagnostico de sesgo y varianza)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(h.history["loss"], label="entrenamiento"); ax.plot(h.history["val_loss"], label="validacion")
    ax.set_xlabel("epoca"); ax.set_ylabel("perdida"); ax.legend(); ax.set_title("Curvas de aprendizaje")
    fig.tight_layout(); fig.savefig("salida_modelo/curvas_aprendizaje.png", dpi=170); plt.close(fig)

    # latencia
    x = np.random.default_rng(0).standard_normal(VENTANA).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(50):
        f = caracteristicas(x); red(f[None, :], training=False)
    print(f"Latencia (caracteristicas + prediccion): {(time.perf_counter() - t0) / 50 * 1000:.1f} ms por ventana")

    red.save("salida_modelo/modelo_notas.keras")
    json.dump({"clases": CLASES, "sr": SR, "ventana": VENTANA, "f_min": F_MIN, "f_max": F_MAX,
               "cents_por_bin": CENTS_POR_BIN},
              open("salida_modelo/config_modelo.json", "w"), indent=1)
    with open("salida_modelo/reporte_test.txt", "w") as fh:
        fh.write(classification_report(yte, pred, labels=range(len(CLASES)), target_names=CLASES, zero_division=0))
    print("Guardado en salida_modelo/: modelo_notas.keras, config_modelo.json, matriz_confusion.png, curvas_aprendizaje.png, reporte_test.txt")


if __name__ == "__main__":
    main()
