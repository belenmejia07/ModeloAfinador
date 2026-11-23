"""
Comparacion con otros modelos (NO modifica el modelo final ni los resultados de la fase 2).
Misma particion por toma (semilla 42), mismo aumento de datos (2 copias con ruido, SNR 10 a 35 dB).
  python modelos_alternativos.py datos/Dataset_Notas
Salida: resultados/modelos_alternativos.csv
"""
import os, sys, glob, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import entrenar_modelo_notas as E
from sklearn.svm import SVC
from tensorflow import keras
import tensorflow as tf

SALIDA = "resultados"; os.makedirs(SALIDA, exist_ok=True)


def ventanas(rutas, rng, aumento=0):
    X, y = [], []
    for r in rutas:
        c = E.clase_de(r)
        for v in E.ventanas_de_toma(E.cargar_wav(r), c == "ruido"):
            X.append(v); y.append(E.CLASES.index(c))
            for _ in range(aumento):
                X.append(E.con_ruido(v, rng)); y.append(E.CLASES.index(c))
    return np.array(X, np.float32), np.array(y)


def onda(x):
    x = x[: len(x) // 4 * 4].reshape(-1, 4).mean(1)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def cnn_bandas(n, k):
    return keras.Sequential([keras.layers.Input((n, 1)),
        keras.layers.Conv1D(16, 5, activation="relu", padding="same"), keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 5, activation="relu", padding="same"), keras.layers.MaxPooling1D(2),
        keras.layers.GlobalMaxPooling1D(), keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3), keras.layers.Dense(k, activation="softmax")])


def rnn_bandas(n, k):
    return keras.Sequential([keras.layers.Input((n, 1)), keras.layers.GRU(32),
        keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(k, activation="softmax")])


def cnn_onda(n, k):
    return keras.Sequential([keras.layers.Input((n, 1)),
        keras.layers.Conv1D(8, 64, strides=8, activation="relu"), keras.layers.MaxPooling1D(4),
        keras.layers.Conv1D(16, 16, strides=2, activation="relu"), keras.layers.MaxPooling1D(4),
        keras.layers.GlobalMaxPooling1D(), keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3), keras.layers.Dense(k, activation="softmax")])


def main(carpeta):
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.wav")))
    rng = np.random.default_rng(E.SEMILLA)
    rep = E.dividir_por_toma(rutas, rng)
    W = {}
    W["tr"] = ventanas(rep["train"], rng, 2); W["va"] = ventanas(rep["val"], rng); W["te"] = ventanas(rep["test"], rng)
    ytr, yva, yte = W["tr"][1], W["va"][1], W["te"][1]
    print("ventanas", len(ytr), len(yva), len(yte), flush=True)
    F = {k: np.array([E.caracteristicas(v) for v in W[k][0]], np.float32) for k in W}
    filas = []

    def reg(nombre, entrada, accs, accva, params, t_ent, t_inf):
        filas.append(dict(modelo=nombre, entrada=entrada, acc_prueba=np.mean(accs), sd=np.std(accs),
                          minimo=min(accs), acc_val=np.mean(accva), parametros=params,
                          seg_entrenamiento=t_ent, ms_por_ventana=t_inf))
        print(filas[-1], flush=True)

    # SVM (sin semilla: determinista)
    sc = StandardScaler().fit(F["tr"])
    for nombre, kw in [("SVM lineal", dict(kernel="linear", C=1.0)), ("SVM RBF", dict(kernel="rbf", C=10.0, gamma="scale"))]:
        t = time.time(); m = SVC(**kw).fit(sc.transform(F["tr"]), ytr); te = time.time() - t
        t = time.time(); p = m.predict(sc.transform(F["te"])); ti = (time.time() - t) / len(yte) * 1000
        reg(nombre, "195 bandas", [float((p == yte).mean())], [float((m.predict(sc.transform(F["va"])) == yva).mean())],
            int(m.support_vectors_.shape[0] * F["tr"].shape[1]), te, ti)

    # Redes (3 semillas)
    def redes(nombre, entrada, f, X):
        accs, accva, ts, tis, params = [], [], [], [], 0
        for s in (1, 2, 3):
            tf.keras.utils.set_random_seed(s)
            m = f(X["tr"].shape[1], len(E.CLASES)); params = m.count_params()
            t = time.time(); E.entrenar(m, X["tr"][..., None], ytr, X["va"][..., None], yva); ts.append(time.time() - t)
            t = time.time(); p = m.predict(X["te"][..., None], verbose=0).argmax(1); tis.append((time.time() - t) / len(yte) * 1000)
            accs.append(float((p == yte).mean())); accva.append(float((m.predict(X["va"][..., None], verbose=0).argmax(1) == yva).mean()))
        reg(nombre, entrada, accs, accva, params, np.mean(ts), np.mean(tis))

    redes("CNN 1D", "195 bandas", cnn_bandas, F)
    redes("RNN (GRU)", "195 bandas", rnn_bandas, F)
    O = {k: np.array([onda(v) for v in W[k][0]], np.float32) for k in W}
    redes("CNN 1D", "forma de onda (11 025)", cnn_onda, O)
    # MLP de referencia con el mismo protocolo
    accs, accva, ts, tis, params = [], [], [], [], 0
    for s in (1, 2, 3):
        tf.keras.utils.set_random_seed(s); m = E.mlp(195, len(E.CLASES)); params = m.count_params()
        t = time.time(); E.entrenar(m, F["tr"], ytr, F["va"], yva); ts.append(time.time() - t)
        t = time.time(); p = m.predict(F["te"], verbose=0).argmax(1); tis.append((time.time() - t) / len(yte) * 1000)
        accs.append(float((p == yte).mean())); accva.append(float((m.predict(F["va"], verbose=0).argmax(1) == yva).mean()))
    reg("MLP (modelo final)", "195 bandas", accs, accva, params, np.mean(ts), np.mean(tis))
    pd.DataFrame(filas).to_csv(f"{SALIDA}/modelos_alternativos.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Dataset_Notas")
