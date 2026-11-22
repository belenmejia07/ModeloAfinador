"""
Grabacion guiada del dataset del afinador (fase 2).

Graba tomas con el microfono de la PC, las revisa (volumen, saturacion, nota escuchada)
y las guarda como WAV de 44.100 Hz, mono, 16 bits:

    Sol3_+0_01.wav     nota _ cents segun el afinador de referencia _ numero de toma
    La4_-22_07.wav
    ruido_0_03_sala.wav   (clase "ruido" = todo lo que NO es una de las 13 notas)

Uso (desde la carpeta fase2):
    python src\\grabar_dataset.py --listar                 # muestra los microfonos disponibles
    python src\\grabar_dataset.py --probar                 # medidor de nivel en vivo (10 s)
    python src\\grabar_dataset.py                          # protocolo completo (se puede reanudar)
    python src\\grabar_dataset.py --notas La4 Mi5          # solo esas notas
    python src\\grabar_dataset.py --solo-ruido             # solo la clase ruido
    python src\\grabar_dataset.py --dispositivo 1          # elegir microfono por numero
"""
import argparse, csv, os, sys, time, glob, datetime
from pathlib import Path
import numpy as np
from scipy.io import wavfile

try:
    import pyaudio
except ImportError:          # permite probar el resto del script sin microfono
    pyaudio = None
try:
    import librosa
except ImportError:
    librosa = None

SR = 44100
RECORTE = 0.25               # segundos descartados al inicio y al final (ataque y cola)
NOTAS = {"Sol3": 196.00, "La3": 220.00, "Si3": 246.94, "Do4": 261.63, "Re4": 293.66, "Mi4": 329.63,
         "Fa4": 349.23, "Sol4": 392.00, "La4": 440.00, "Si4": 493.88, "Do5": 523.25, "Re5": 587.33,
         "Mi5": 659.26}
AL_AIRE = {"Sol3", "Re4", "La4", "Mi5"}
CONDICIONES = [("afinada", 0), ("baja", -30), ("alta", +30)]
TIPOS_RUIDO = [
    ("silencio", "No toques nada. Ambiente en silencio."),
    ("sala", "Ruido normal de la sala: muevete, mueve la silla, respira, ruido de fondo."),
    ("roce", "Roce del arco o de las manos sobre el violin, SIN nota clara."),
    ("voz", "Habla o tararea cerca del microfono."),
    ("clavija", "Manipula el violin: gira una clavija, da golpecitos suaves al cuerpo."),
    ("alterada", "Toca UNA NOTA ALTERADA sostenida (Fa#, Do#, Sol#, Sib...). Cambia de nota en cada toma."),
    ("fuera_rango", "Toca una nota SOSTENIDA fuera del rango Sol3-Mi5 (por ejemplo Fa5 o Sol5)."),
]

# ------------------------------------------------------------------ audio
class GrabadoraPyAudio:
    def __init__(self, dispositivo=None):
        if pyaudio is None:
            raise SystemExit("PyAudio no esta instalado en este entorno.")
        self.pa = pyaudio.PyAudio()
        self.dispositivo = dispositivo

    def nombre(self):
        info = (self.pa.get_device_info_by_index(self.dispositivo) if self.dispositivo is not None
                else self.pa.get_default_input_device_info())
        return info["name"]

    def grabar(self, segundos):
        n = int(segundos * SR)
        try:
            s = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SR, input=True,
                             input_device_index=self.dispositivo, frames_per_buffer=1024)
        except Exception as e:
            raise SystemExit(f"No pude abrir el microfono a {SR} Hz: {e}\n"
                             "En Windows: Configuracion > Sonido > Propiedades del microfono > "
                             "Avanzado, y elige 44100 Hz.")
        bloques = []
        try:
            for _ in range(int(np.ceil(n / 1024))):
                bloques.append(s.read(1024, exception_on_overflow=False))
        finally:
            s.stop_stream(); s.close()
        return np.frombuffer(b"".join(bloques), dtype=np.int16)[:n]

    def listar(self):
        for i in range(self.pa.get_device_count()):
            d = self.pa.get_device_info_by_index(i)
            if d["maxInputChannels"] > 0:
                print(f"  [{i}] {d['name']}  (entradas: {d['maxInputChannels']}, "
                      f"frecuencia por defecto: {int(d['defaultSampleRate'])} Hz)")


def pitar(frecuencia=880, ms=180):
    try:
        import winsound
        winsound.Beep(frecuencia, ms)
    except Exception:
        print("\a", end="", flush=True)


def dbfs(valor):
    return 20 * np.log10(max(float(valor), 1e-9))


def medidor_en_vivo(grab, segundos=10):
    print("Toca una nota fuerte y otra suave. Objetivo: picos entre -20 y -6 dBFS, sin llegar a 0.\n")
    t0 = time.time()
    while time.time() - t0 < segundos:
        x = grab.grabar(0.25).astype(np.float32) / 32768
        pico, rms = dbfs(np.max(np.abs(x))), dbfs(np.sqrt(np.mean(x ** 2)))
        barra = "#" * int(max(0, (pico + 60)) / 60 * 40)
        estado = "SATURA" if pico > -1 else ("bajo" if pico < -30 else "ok")
        print(f"\rpico {pico:6.1f} dBFS | rms {rms:6.1f} | {barra:<40} {estado:<6}", end="", flush=True)
    print()


# ------------------------------------------------------------------ analisis
def analizar(x_int16, ruido_base_db, es_ruido):
    x = x_int16.astype(np.float32) / 32768
    c = int(RECORTE * SR)
    centro = x[c:-c] if len(x) > 2 * c + SR else x
    info = {"pico_dbfs": dbfs(np.max(np.abs(x))), "rms_dbfs": dbfs(np.sqrt(np.mean(centro ** 2))),
            "saturacion": float(np.mean(np.abs(x) >= 0.99))}
    info["snr_db"] = info["rms_dbfs"] - ruido_base_db if ruido_base_db is not None else float("nan")
    avisos = []
    if info["saturacion"] > 0:
        avisos.append("SATURACION: baja la ganancia del microfono o alejate un poco.")
    if not es_ruido:
        if info["pico_dbfs"] < -30:
            avisos.append("Volumen muy bajo (pico < -30 dBFS): acercate o sube la ganancia.")
        if ruido_base_db is not None and info["snr_db"] < 15:
            avisos.append(f"Poca senal frente al ruido de fondo (SNR {info['snr_db']:.0f} dB < 15).")
    return info, avisos, centro


def nota_mas_cercana(f):
    nombre = min(NOTAS, key=lambda n: abs(1200 * np.log2(f / NOTAS[n])))
    return nombre, 1200 * np.log2(f / NOTAS[nombre])


def estimar_nota(centro):
    """Verificacion orientativa con pYIN: que nota escucha el sistema."""
    if librosa is None:
        return None
    f0, voz, _ = librosa.pyin(centro, fmin=170, fmax=1400, sr=SR)
    f0 = f0[voz]
    if len(f0) < 5:
        return None
    f = float(np.nanmedian(f0))
    nombre, cents = nota_mas_cercana(f)
    return f, nombre, cents


# ------------------------------------------------------------------ plan
def construir_plan(notas, tomas, solo_ruido, sin_ruido, tomas_ruido):
    plan = []
    if not solo_ruido:
        for nota in notas:
            n = 0
            for cond, nominal in CONDICIONES:
                for k in range(1, tomas + 1):
                    n += 1
                    plan.append({"clase": nota, "cond": cond, "nominal": nominal, "toma": n, "tipo": "",
                                 "k": k, "n_cond": tomas})
    if not sin_ruido:
        n = 0
        for tipo, _ in TIPOS_RUIDO:
            for k in range(1, tomas_ruido + 1):
                n += 1
                plan.append({"clase": "ruido", "cond": "ruido", "nominal": 0, "toma": n, "tipo": tipo,
                             "k": k, "n_cond": tomas_ruido})
    return plan


def ya_existe(carpeta, it):
    if it["clase"] == "ruido":
        return bool(glob.glob(str(carpeta / f"ruido_0_{it['toma']:02d}_*.wav")))
    return bool(glob.glob(str(carpeta / f"{it['clase']}_*_{it['toma']:02d}.wav")))


def instruccion(it):
    if it["clase"] == "ruido":
        return dict(TIPOS_RUIDO)[it["tipo"]]
    base = {"afinada": "AFINADA (0 cents)", "baja": "BAJA: unos 30 cents mas GRAVE de lo correcto",
            "alta": "ALTA: unos 30 cents mas AGUDA de lo correcto"}[it["cond"]]
    extra = ""
    if it["clase"] in AL_AIRE and it["cond"] != "afinada":
        extra = "\n   (cuerda al aire: mueve la clavija y comprueba con el afinador de referencia)"
    return f"Nota {it['clase']} {base}. Sostenla todo el tiempo de la toma.{extra}"


# ------------------------------------------------------------------ principal
def main(argv=None, entrada=input, grabadora=None, base=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=None)
    ap.add_argument("--notas", nargs="*", default=list(NOTAS))
    ap.add_argument("--tomas", type=int, default=5, help="tomas por condicion (afinada/baja/alta)")
    ap.add_argument("--tomas-ruido", type=int, default=5, help="tomas por tipo de ruido")
    ap.add_argument("--duracion", type=float, default=4.0)
    ap.add_argument("--dispositivo", type=int, default=None)
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--probar", action="store_true")
    ap.add_argument("--solo-ruido", action="store_true")
    ap.add_argument("--sin-ruido", action="store_true")
    ap.add_argument("--sin-calibracion", action="store_true")
    a = ap.parse_args(argv)

    grab = grabadora or GrabadoraPyAudio(a.dispositivo)
    if a.listar:
        print("Microfonos disponibles:"); grab.listar(); return
    if a.probar:
        medidor_en_vivo(grab); return

    carpeta = Path(a.salida) if a.salida else (base or Path(__file__).resolve().parents[1]) / "datos" / "Dataset_Notas"
    carpeta.mkdir(parents=True, exist_ok=True)
    for n in a.notas:
        if n not in NOTAS:
            raise SystemExit(f"Nota desconocida: {n}. Validas: {', '.join(NOTAS)}")
    print(f"Microfono: {grab.nombre() if hasattr(grab, 'nombre') else 'simulado'} | carpeta: {carpeta}")

    ruido_base = None
    if not a.sin_calibracion:
        entrada("\nCalibracion: 3 s de SILENCIO total (no te muevas). Enter para empezar... ")
        pitar(660); x = grab.grabar(3.0); pitar(440)
        ruido_base = dbfs(np.sqrt(np.mean((x.astype(np.float32) / 32768) ** 2)))
        print(f"Ruido de fondo: {ruido_base:.1f} dBFS (referencia para la relacion senal/ruido)")

    plan = construir_plan(a.notas, a.tomas, a.solo_ruido, a.sin_ruido, a.tomas_ruido)
    registro = carpeta / "registro.csv"
    nuevo = not registro.exists()
    csvf = open(registro, "a", newline="", encoding="utf-8")
    w = csv.writer(csvf)
    if nuevo:
        w.writerow(["archivo", "clase", "condicion", "cents_nominal", "cents_referencia", "tipo_ruido",
                    "pico_dbfs", "rms_dbfs", "ruido_base_dbfs", "snr_db", "f_est_hz", "nota_est",
                    "cents_est", "fecha"])
    guardadas = 0
    i = 0
    ultima_clave = None
    try:
        while i < len(plan):
            it = plan[i]
            if ya_existe(carpeta, it):
                i += 1; continue
            clave = (it["clase"], it["cond"], it["tipo"])
            if clave != ultima_clave:
                print("\n" + "=" * 70); print(instruccion(it)); ultima_clave = clave
            hechas = sum(1 for p in plan if ya_existe(carpeta, p))
            pista = ""
            if it["clase"] != "ruido":
                pista = " - con vibrato natural" if it["k"] > int(np.ceil(0.6 * it["n_cond"])) else " - sin vibrato"
            r = entrada(f"\n[{hechas}/{len(plan)}] {it['clase']} - {it['tipo'] or it['cond']} - "
                        f"toma {it['k']}/{it['n_cond']}{pista}"
                        "  |  Enter = grabar, s = saltar esta nota/tipo, q = salir: ").strip().lower()
            if r == "q":
                break
            if r == "s":
                while i < len(plan) and (plan[i]["clase"], plan[i]["tipo"]) == (it["clase"], it["tipo"]):
                    i += 1
                ultima_clave = None; continue
            for k in (3, 2, 1):
                print(k, end=" ", flush=True); time.sleep(0.4 if grabadora is None else 0)
            pitar(880); print("GRABANDO...", flush=True)
            x = grab.grabar(a.duracion); pitar(440)

            es_ruido = it["clase"] == "ruido"
            info, avisos, centro = analizar(x, ruido_base, es_ruido)
            est = None
            if not es_ruido or it["tipo"] in ("alterada", "fuera_rango"):
                est = estimar_nota(centro)
            print(f"  pico {info['pico_dbfs']:.1f} dBFS | rms {info['rms_dbfs']:.1f} dBFS | "
                  f"SNR {info['snr_db']:.0f} dB" if ruido_base is not None else
                  f"  pico {info['pico_dbfs']:.1f} dBFS | rms {info['rms_dbfs']:.1f} dBFS")
            if est and not es_ruido:
                f, n_est, c_est = est
                print(f"  El sistema escucha: {n_est} ({c_est:+.0f} cents, {f:.1f} Hz)")
                if n_est != it["clase"]:
                    avisos.append(f"OJO: esperabas {it['clase']} y el sistema escucha {n_est}. "
                                  "Puede ser otra nota u otra octava.")
            for av in avisos:
                print("  !", av)
            r2 = entrada("  Enter = guardar, r = repetir, s = saltar, q = salir: ").strip().lower()
            if r2 == "r":
                continue
            if r2 == "q":
                break
            if r2 == "s":
                while i < len(plan) and (plan[i]["clase"], plan[i]["tipo"]) == (it["clase"], it["tipo"]):
                    i += 1
                ultima_clave = None; continue

            if es_ruido:
                nombre = f"ruido_0_{it['toma']:02d}_{it['tipo']}.wav"; cents_ref = 0.0
            else:
                while True:
                    txt = entrada(f"  Cents del afinador de referencia [Enter = {it['nominal']:+d}]: ").strip()
                    try:
                        cents_ref = float(txt.replace(",", ".")) if txt else float(it["nominal"])
                        if -100 <= cents_ref <= 100:
                            break
                    except ValueError:
                        pass
                    print("  Escribe un numero entre -100 y 100 (por ejemplo -27).")
                nombre = f"{it['clase']}_{int(round(cents_ref)):+d}_{it['toma']:02d}.wav"
            wavfile.write(str(carpeta / nombre), SR, x)
            w.writerow([nombre, it["clase"], it["cond"], it["nominal"], cents_ref, it["tipo"],
                        f"{info['pico_dbfs']:.1f}", f"{info['rms_dbfs']:.1f}",
                        "" if ruido_base is None else f"{ruido_base:.1f}", f"{info['snr_db']:.1f}",
                        "" if not est else f"{est[0]:.2f}", "" if not est else est[1],
                        "" if not est else f"{est[2]:.1f}", datetime.datetime.now().isoformat(timespec="seconds")])
            csvf.flush(); guardadas += 1
            print(f"  Guardado: {nombre}")
            i += 1
    finally:
        csvf.close()
    print(f"\nSesion terminada: {guardadas} tomas nuevas. Total en plan: "
          f"{sum(1 for p in plan if ya_existe(carpeta, p))}/{len(plan)}. Registro: {registro}")


if __name__ == "__main__":
    main()
