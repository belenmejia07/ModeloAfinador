"""
Experimentos complementarios de la fase 2 (para el articulo).
  python experimentos_fase2.py Dataset_Notas ablacion   # compara representaciones de entrada
  python experimentos_fase2.py Dataset_Notas ruido      # robustez del MLP y pYIN ante ruido anadido
  python experimentos_fase2.py Dataset_Notas vibrato    # estabilidad del tono por toma (pYIN)
  python experimentos_fase2.py Dataset_Notas aumento    # rango de SNR del aumento de datos y robustez
  python experimentos_fase2.py Dataset_Notas sistema    # prueba de extremo a extremo con el codigo de la demo
  python experimentos_fase2.py Dataset_Notas acuerdo    # acuerdo entre estimadores de cents (usa resultados/comparacion_cents_detalle.csv)
Salidas en resultados/.
"""
import os, sys, glob, re
import numpy as np
import pandas as pd
import entrenar_modelo_notas as E

SALIDA = "resultados"
os.makedirs(SALIDA, exist_ok=True)


def con_ruido_rango(x, rng, lo, hi):
    snr = rng.uniform(lo, hi)
    ruido = rng.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))
    return (x + ruido).astype(np.float32)


def ventanas(rutas, rng, aumento=0, snr=(10, 35)):
    X, y = [], []
    for r in rutas:
        c = E.clase_de(r)
        for v in E.ventanas_de_toma(E.cargar_wav(r), c == "ruido"):
            X.append(v); y.append(E.CLASES.index(c))
            for _ in range(aumento):
                X.append(con_ruido_rango(v, rng, *snr)); y.append(E.CLASES.index(c))
    return np.array(X, np.float32), np.array(y)


_FREQS = np.fft.rfftfreq(E.VENTANA, 1 / E.SR)
_SEL = (_FREQS >= E.F_MIN) & (_FREQS <= E.F_MAX)
_RECT = np.ones(E.VENTANA)


def f_onda(x):
    x = x[: len(x) // 4 * 4].reshape(-1, 4).mean(1)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def f_fft_lineal(x):
    mag = np.abs(np.fft.rfft(x * E._HANN))[_SEL]
    db = np.clip(20 * np.log10(mag + 1e-9) - 20 * np.log10(mag.max() + 1e-9), -80, 0)
    return ((db + 80) / 80).astype(np.float32)


def f_bandas_sin_hann(x):
    mag = np.abs(np.fft.rfft(x * _RECT))
    banda = np.maximum.reduceat(mag[:E._IDX[-1]], E._IDX[:-1])
    db = 20 * np.log10(banda + 1e-9)
    db = np.clip(db - db.max(), -80, 0)
    return ((db + 80) / 80).astype(np.float32)


REPRES = {
    "Onda temporal (11 025 muestras)": f_onda,
    "Espectro FFT lineal (1 Hz, 2 821 valores)": f_fft_lineal,
    "Bandas logaritmicas sin ventana de Hann (195)": f_bandas_sin_hann,
    "Bandas logaritmicas con ventana de Hann (195)": E.caracteristicas,
}


def aplicar(f, X):
    return np.array([f(v) for v in X], np.float32)


def preparar(carpeta):
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.wav")))
    rng = np.random.default_rng(E.SEMILLA)
    rep = E.dividir_por_toma(rutas, rng)
    Xtr, ytr = ventanas(rep["train"], rng, aumento=2)
    Xva, yva = ventanas(rep["val"], rng)
    Xte, yte = ventanas(rep["test"], rng)
    return rutas, rep, (Xtr, ytr), (Xva, yva), (Xte, yte)


def ablacion(carpeta):
    import tensorflow as tf
    _, _, (Xtr, ytr), (Xva, yva), (Xte, yte) = preparar(carpeta)
    filas = []
    for nombre, f in REPRES.items():
        A, B, C = aplicar(f, Xtr), aplicar(f, Xva), aplicar(f, Xte)
        accs, ms = [], []
        for semilla in (1, 2, 3):
            tf.keras.utils.set_random_seed(semilla)
            m = E.mlp(A.shape[1], len(E.CLASES)); h = E.entrenar(m, A, ytr, B, yva)
            accs.append(float((m.predict(C, verbose=0).argmax(1) == yte).mean()))
            ms.append(len(h.history["loss"]))
        filas.append(dict(representacion=nombre, entradas=A.shape[1], acc_media=np.mean(accs),
                          acc_sd=np.std(accs), acc_min=min(accs), epocas_media=np.mean(ms)))
        print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(f"{SALIDA}/ablacion_representaciones.csv", index=False)


def ruido(carpeta, modelo_ruta="salida_modelo/modelo_notas.keras"):
    import tensorflow as tf
    import comparar_estimadores as C
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.wav")))
    te = E.dividir_por_toma(rutas, np.random.default_rng(E.SEMILLA))["test"]
    modelo = tf.keras.models.load_model(modelo_ruta)
    W, y = [], []
    for r in te:
        c = E.clase_de(r)
        for v in list(E.ventanas_de_toma(E.cargar_wav(r), c == "ruido"))[:2]:
            W.append(v); y.append(c)
    rng = np.random.default_rng(0)
    filas = []
    for snr in (None, 30, 20, 10, 5, 0):
        Z = []
        for v in W:
            if snr is None:
                Z.append(v)
            else:
                p = np.mean(v ** 2) + 1e-12
                Z.append((v + rng.standard_normal(len(v)) * np.sqrt(p / 10 ** (snr / 10))).astype(np.float32))
        Xf = np.array([E.caracteristicas(v) for v in Z])
        pm = [E.CLASES[i] for i in modelo.predict(Xf, verbose=0).argmax(1)]
        pp = [C.nota_pyin(v) for v in Z]
        filas.append(dict(snr_db="sin ruido anadido" if snr is None else snr, n=len(W),
                          acc_mlp=np.mean([a == b for a, b in zip(pm, y)]),
                          acc_pyin=np.mean([a == b for a, b in zip(pp, y)])))
        print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(f"{SALIDA}/robustez_ruido.csv", index=False)


def vibrato(carpeta):
    import librosa
    filas = []
    for r in sorted(glob.glob(os.path.join(carpeta, "*.wav"))):
        c = E.clase_de(r)
        if c in (None, "ruido"):
            continue
        n = int(os.path.basename(r).split("_")[2].split(".")[0])
        x = E.cargar_wav(r)[E.RECORTE:-E.RECORTE]
        f0, vf, _ = librosa.pyin(x, fmin=170, fmax=1400, sr=E.SR, frame_length=4096, hop_length=441)
        f0 = f0[vf & ~np.isnan(f0)] if vf is not None else f0[~np.isnan(f0)]
        if len(f0) < 20:
            continue
        cents = 1200 * np.log2(f0 / np.median(f0))
        filas.append(dict(archivo=os.path.basename(r), nota=c, toma_global=n, toma=((n - 1) % 5) + 1,
                          sd_cents=float(np.std(cents)), rango_cents=float(np.percentile(cents, 95) - np.percentile(cents, 5))))
    d = pd.DataFrame(filas)
    d.to_csv(f"{SALIDA}/estabilidad_tono.csv", index=False)
    print(d.groupby("toma")[["sd_cents", "rango_cents"]].agg(["mean", "median"]).round(2))
    print("global: sd media", d.sd_cents.mean().round(2), "rango medio", d.rango_cents.mean().round(2))


def acuerdo(carpeta):
    d = pd.read_csv(f"{SALIDA}/comparacion_cents_detalle.csv")
    p = d.pivot_table(index=["archivo", "nota"], columns="metodo", values="estimado").reset_index()
    p["dif_fft_pyin"] = (p["FFT armonicos"] - p["pYIN"]).abs()
    p["dif_pico_arm"] = (p["FFT pico"] - p["FFT armonicos"]).abs()
    print("n", p.dif_fft_pyin.notna().sum(), "| |FFT arm - pYIN| media", round(p.dif_fft_pyin.mean(), 2),
          "mediana", round(p.dif_fft_pyin.median(), 2), "p95", round(p.dif_fft_pyin.quantile(.95), 2))
    print("|FFT pico - FFT arm| media", round(p.dif_pico_arm.mean(), 2))
    p.to_csv(f"{SALIDA}/acuerdo_estimadores.csv", index=False)


def aumento(carpeta):
    """Efecto del rango de SNR usado en el aumento de datos sobre la robustez ante ruido."""
    import tensorflow as tf
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.wav")))
    rng = np.random.default_rng(E.SEMILLA)
    rep = E.dividir_por_toma(rutas, rng)
    Xva, yva = ventanas(rep["val"], rng); Xte, yte = ventanas(rep["test"], rng)
    filas = []
    for rango in ((10, 35), (0, 35)):
        Xtr, ytr = ventanas(rep["train"], np.random.default_rng(E.SEMILLA), aumento=2, snr=rango)
        A, B = aplicar(E.caracteristicas, Xtr), aplicar(E.caracteristicas, Xva)
        tf.keras.utils.set_random_seed(1)
        m = E.mlp(A.shape[1], len(E.CLASES)); E.entrenar(m, A, ytr, B, yva)
        r2 = np.random.default_rng(0)
        for snr in (None, 20, 10, 5, 0):
            Z = Xte if snr is None else np.array([con_ruido_rango(v, r2, snr, snr) for v in Xte])
            F = aplicar(E.caracteristicas, Z)
            acc = float((m.predict(F, verbose=0).argmax(1) == yte).mean())
            filas.append(dict(aumento_snr=f"{rango[0]}-{rango[1]} dB", snr_prueba="ninguno" if snr is None else snr, acc=acc))
            print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(f"{SALIDA}/aumento_ruido.csv", index=False)


def sistema(carpeta, modelo_ruta="salida_modelo/modelo_notas.keras"):
    """Prueba de extremo a extremo con el mismo codigo de la demo (afinador_visual_v2)."""
    import time
    import afinador_visual_v2 as V
    modelo = V.cargar_modelo(modelo_ruta)
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.wav")))
    test = set(os.path.basename(r) for r in E.dividir_por_toma(rutas, np.random.default_rng(E.SEMILLA))["test"])
    filas = []
    for r in rutas:
        c = E.clase_de(r)
        if c is None:
            continue
        b = os.path.basename(r)
        cond = None
        if c != "ruido":
            n = int(b.split("_")[2].split(".")[0]); cond = ["afinada", "baja", "alta"][(n - 1) // 5]
        for w in E.ventanas_de_toma(E.cargar_wav(r), c == "ruido"):
            t0 = time.time(); a = V.analizar_ventana(w, modelo); ms = (time.time() - t0) * 1000
            filas.append(dict(archivo=b, clase=c, condicion=cond, test=b in test, nota=a["nota"], prob=a["prob"],
                              cents=a["cents"], estado=a["estado"], ms=ms))
    d = pd.DataFrame(filas); d.to_csv(f"{SALIDA}/sistema_extremo_a_extremo.csv", index=False)
    notas = d[d.clase != "ruido"]; ruido_ = d[d.clase == "ruido"]
    t = notas[notas.test]
    print("TEST notas: ventanas", len(t), "| nota correcta", round((t.nota == t.clase).mean(), 4),
          "| rechazadas", round(t.nota.isna().mean(), 4))
    tr = ruido_[ruido_.test]
    print("TEST ruido: ventanas", len(tr), "| rechazadas (Sin nota/No reconocida)", round(tr.nota.isna().mean(), 4))
    print("TODAS ruido: ventanas", len(ruido_), "| rechazadas", round(ruido_.nota.isna().mean(), 4))
    print("\nCENTS por condicion (todas las tomas de nota, ventanas con medicion):")
    g = notas.dropna(subset=["cents"]).groupby("condicion").cents.agg(["count", "mean", "std", "median"]).round(2)
    print(g)
    print("\nDecision Afinada/Desafinada vs condicion (ventanas con nota):")
    nn = notas.dropna(subset=["cents"]).copy(); nn["afinada_real"] = nn.condicion == "afinada"
    nn["afinada_pred"] = nn.estado == "Afinada"
    print("exactitud", round((nn.afinada_real == nn.afinada_pred).mean(), 4), "n", len(nn))
    print(pd.crosstab(nn.condicion, nn.estado))
    print("\nlatencia media ms", round(d.ms.mean(), 2), "p95", round(d.ms.quantile(.95), 2))


if __name__ == "__main__":
    carpeta, tarea = sys.argv[1], sys.argv[2]
    {"ablacion": ablacion, "ruido": ruido, "vibrato": vibrato, "acuerdo": acuerdo, "aumento": aumento, "sistema": sistema}[tarea](carpeta)
