# Afinador de violín con inteligencia artificial

Afinador en tiempo real para violín. Escucha el micrófono, reconoce la nota que se toca, mide cuántos **cents** se aleja de la afinación correcta y muestra el resultado con un color (verde si está afinada, rojo si no).

Proyecto de la asignatura **Inteligencia Artificial**, Ingeniería en Inteligencia Artificial, Universidad Católica Boliviana «San Pablo», sede Santa Cruz.
Autora: Belen Mejia Medina.

## Cómo funciona

1. El micrófono captura el audio y lo corta en **ventanas de 1 s**.
2. **Identificar la nota.** En la fase 2, una red neuronal propia (MLP) recibe el espectro de Fourier de la ventana y decide cuál de las 13 notas suena, o si es ruido.
3. **Medir la afinación.** La FFT con interpolación parabólica estima la frecuencia y la convierte a cents respecto de la nota reconocida.
4. **Decidir y mostrar.** Si la precisión es de 70 o más (hasta 15 cents de desviación), la nota está afinada. La pantalla (OpenCV) muestra nota, cents y color.

## El proyecto en dos fases

| | Fase 1 | Fase 2 |
|---|---|---|
| Notas | 4 cuerdas al aire (Sol, Re, La, Mi) | 13 notas naturales (Sol3 a Mi5) y ruido |
| Modelo | Teachable Machine (TensorFlow Lite) | MLP propio, 34 254 parámetros, entrenado desde cero |
| Procesamiento | El de la herramienta; librosa (pYIN) para los cents | Ventana de Hann, FFT y 195 bandas logarítmicas; FFT para los cents |
| Datos | 150 muestras | 230 tomas de 4 s con control de pico y SNR |
| Resultado | 56,1 % (32 de 57) con el clasificador; 92,1 % (105 de 114) con la corrección por cents | 100,0 % en prueba; 97,4 ± 1,5 % en validación cruzada por toma |

Las métricas de las dos fases no son comparables: la tarea cambió.

La fase 1 resultó inconsistente al etiquetar afinada y desafinada con el clasificador. Por eso la decisión final se tomó con los cents medidos y el clasificador solo distinguía ruido de nota. En la fase 2 se ampliaron las notas, se evaluó el procesamiento con Fourier y se entrenó un modelo propio, sin herramientas preentrenadas.

## Resultados de la fase 2

- **Clasificación:** 100,0 % (283 de 283 ventanas de prueba). La cifra más realista es la validación cruzada por toma: **97,4 ± 1,5 %**.
- **Fourier es decisivo:** con la forma de onda la red acierta 41,5 %; con cualquier espectro, 99,4 % o más.
- **Cents:** error medio de 6,41 cents con la FFT, en 3,7 ms por ventana. pYIN da 5,54 cents pero tarda 395 ms.
- **Sistema completo:** 99,6 % de notas bien identificadas, 100,0 % de ruido rechazado y 98,5 % de decisiones correctas entre afinada y desafinada.
- **Otros modelos** (mismos datos): SVM RBF 100,0 %, SVM lineal 97,2 %, CNN 1D 92,5 %, RNN 90,3 %.

Detalle completo, fórmulas y experimentos en [`fase2/README_fase2.md`](fase2/README_fase2.md).

## Estructura del repositorio

```
ViolinTuner/
├── fase1/        # afinador con Teachable Machine y librosa
├── fase2/        # modelo propio (MLP), datos, experimentos y demostración
│   ├── src/        # scripts de grabación, entrenamiento, demo y experimentos
│   ├── modelos/    # modelo entrenado y su configuración
│   ├── resultados/ # reportes, figuras y tablas
│   └── datos/      # registro de las tomas (los audios no se suben)
├── docs/         # artículo, presentaciones y documentos del proyecto
├── Pruebas/      # scripts exploratorios
└── README.md
```

El entorno virtual (`venvVioTun`) no se sube; está excluido con `.gitignore`.

## Instalación

Requisitos: Windows, Python 3.11 y un micrófono.

```powershell
python -m venv venvVioTun
venvVioTun\Scripts\activate
pip install tensorflow numpy scipy scikit-learn pandas matplotlib soundfile librosa pyaudio opencv-python
```

El modelo de la fase 2 se entrenó con TensorFlow 2.21. Con otra versión puede ser necesario reentrenarlo (ver abajo).

## Uso rápido (fase 2)

Desde la carpeta `fase2`:

```powershell
# Demostración en tiempo real (en mi equipo el micrófono es el dispositivo 1)
python src\afinador_visual_v2.py --modelo modelos\modelo_notas.keras --dispositivo 1

# Analizar un archivo sin micrófono ni cámara
python src\afinador_visual_v2.py --modelo modelos\modelo_notas.keras --wav toma.wav --sin-camara

# Reentrenar el modelo con tus propias grabaciones
python src\entrenar_modelo_notas.py datos\Dataset_Notas
```

Para grabar un dataset propio: `python src\grabar_dataset.py --listar` muestra los micrófonos y `python src\grabar_dataset.py --dispositivo 1` inicia la grabación. Los experimentos están descritos en el README de la fase 2.

## Datos

230 tomas de 4 s (44,1 kHz, 16 bits, un canal) del micrófono de la laptop, grabadas en una sola sesión: 195 de notas (13 notas × 3 afinaciones nominales × 5 tomas) y 35 de ruido (7 tipos × 5 tomas), todas sin vibrato. Los audios no se suben por su tamaño; se sube `registro.csv` con el pico, el nivel y la relación señal-ruido de cada toma.

## Limitaciones

Un violín, una ejecutante, un micrófono y una sesión; sin vibrato; cents nominales sin afinador externo; solo notas naturales; ruido blanco como única perturbación evaluada; latencia medida fuera de la laptop de captura. Con otro violín o micrófono el acierto puede ser menor y los umbrales de decisión pueden necesitar ajuste.

## Documentación

- Artículo científico (en `docs/`).
- Presentación de la fase 2 (en `docs/`).
- [`fase2/README_fase2.md`](fase2/README_fase2.md): pipeline, fórmulas, experimentos y cómo reproducirlos.

## Referencias principales

Cooley y Tukey (1965); de Cheveigné y Kawahara (2002); Harris (1978); Kingma y Ba (2015); Mauch y Dixon (2014); Salamon y Gómez (2012); Srivastava et al. (2014). Referencias completas en el artículo.
