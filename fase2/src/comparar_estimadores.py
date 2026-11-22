"""
Comparacion de estimadores (para el articulo):
 A) cents:  pYIN (resolution=0.01)  vs  FFT pico unico  vs  FFT con armonicos
 B) identificacion de nota (split TEST): pYIN->nota mas cercana  vs  MLP propio

Uso: python comparar_estimadores.py Dataset_Notas [--max-ventanas 2] [--salida resultados] [--modelo ruta]
Usa solo tomas de notas (no ruido) con cents nominal en el nombre NOTA_CENTS_TOMA.wav
"""
import os, sys, glob, time, argparse
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import entrenar_modelo_notas as E
import afinador_visual_v2 as V


def cents_pyin(x, f_ref):
    f0, vf, _ = librosa.pyin(x, fmin=f_ref * 0.7, fmax=f_ref * 1.4, sr=E.SR,
                             frame_length=4096, resolution=0.01)
    f0 = f0[vf & ~np.isnan(f0)] if vf is not None else f0[~np.isnan(f0)]
    if len(f0) == 0:
        return None
    return 1200 * np.log2(np.median(f0) / f_ref)


def cents_fft_pico(x, f_ref):
    c, _ = V.cents_fft(x, f_ref, armonicos=(1,))
    return c


def cents_fft_arm(x, f_ref):
    c, _ = V.cents_fft(x, f_ref)
    return c


def nota_pyin(x):
    f0, vf, _ = librosa.pyin(x, fmin=170, fmax=800, sr=E.SR, frame_length=4096)
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return "ruido"
    f = np.median(f0)
    return min(V.HZ, key=lambda n: abs(1200 * np.log2(f / V.HZ[n])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta"); ap.add_argument("--max-ventanas", type=int, default=2)
    ap.add_argument("--salida", default="resultados"); ap.add_argument("--modelo")
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    rutas = sorted(glob.glob(os.path.join(a.carpeta, "*.wav")))
    est = {"pYIN": cents_pyin, "FFT pico": cents_fft_pico, "FFT armonicos": cents_fft_arm}

    filas = []
    for r in rutas:
        c = E.clase_de(r)
        if c not in V.HZ:
            continue
        try:
            nom = float(os.path.basename(r).split("_")[1])
        except Exception:
            continue
        verdad = nom + 1200 * np.log2(1.0)  # cents nominales respecto a la nota exacta
        for k, w in enumerate(list(E.ventanas_de_toma(E.cargar_wav(r), False))[:a.max_ventanas]):
            for nombre, fn in est.items():
                t0 = time.time(); v = fn(w, V.HZ[c]); ms = (time.time() - t0) * 1000
                filas.append(dict(archivo=os.path.basename(r), nota=c, nominal=verdad,
                                  metodo=nombre, estimado=v, ms=ms))
    df = pd.DataFrame(filas)
    df["error"] = df.estimado - df.nominal
    df.to_csv(os.path.join(a.salida, "comparacion_cents_detalle.csv"), index=False)
    res = []
    for m, g in df.groupby("metodo"):
        ok = g.dropna(subset=["error"])
        res.append(dict(metodo=m, n=len(g), detectadas=len(ok),
                        MAE=ok.error.abs().mean(), RMSE=np.sqrt((ok.error ** 2).mean()),
                        pct_error_grueso_50=100 * (ok.error.abs() > 50).mean(),
                        ms_por_ventana=g.ms.mean()))
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(a.salida, "comparacion_cents_resumen.csv"), index=False)
    print(res.round(2).to_string(index=False))
    print("(El nominal es la desafinacion pedida al grabar; incluye la imprecision del"
          " violinista/tono, por eso el error absoluto no es cero ni con un estimador perfecto.)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(res.metodo, res.MAE); ax[0].set_ylabel("MAE (cents)"); ax[0].set_title("Error de cents")
    ax[1].bar(res.metodo, res.ms_por_ventana); ax[1].set_ylabel("ms/ventana"); ax[1].set_title("Tiempo")
    fig.tight_layout(); fig.savefig(os.path.join(a.salida, "comparacion_cents.png"), dpi=150)

    # B) identificacion en TEST
    te = E.dividir_por_toma(rutas, np.random.default_rng(E.SEMILLA))['test']
    modelo = V.cargar_modelo(a.modelo)
    ids = []
    for r in te:
        verdad = E.clase_de(r)
        for w in list(E.ventanas_de_toma(E.cargar_wav(r), verdad == 'ruido'))[:a.max_ventanas]:
            p = E.CLASES[int(modelo.predict(E.caracteristicas(w)[None], verbose=0)[0].argmax())]
            ids.append(dict(archivo=os.path.basename(r), real=verdad, pyin=nota_pyin(w), mlp=p))
    d = pd.DataFrame(ids)
    d.to_csv(os.path.join(a.salida, "comparacion_identificacion.csv"), index=False)
    print(f"\nIdentificacion (test, {len(d)} ventanas): "
          f"pYIN={100*(d.pyin==d.real).mean():.1f}%  MLP propio={100*(d.mlp==d.real).mean():.1f}%")


if __name__ == "__main__":
    main()
