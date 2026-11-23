# Afinador de violín, fase 2

Afinador en tiempo real para **13 notas del violín** (Sol3 a Mi5) con una **red neuronal propia** entrenada desde cero, sin herramientas preentrenadas. La red identifica la nota a partir del espectro de Fourier y la FFT mide la desviación en cents.

Proyecto de la asignatura Inteligencia Artificial, Ingeniería en Inteligencia Artificial, Universidad Católica Boliviana «San Pablo», sede Santa Cruz.
Autora: Belen Mejia Medina.

## Qué cambió respecto de la fase 1

| Aspecto | Fase 1 | Fase 2 |
|---|---|---|
| Notas | 4 cuerdas al aire | 13 notas naturales (Sol3 a Mi5) y ruido |
| Modelo | Teachable Machine | MLP propio, 34 254 parámetros |
| Preprocesamiento | Interno de la herramienta | Hann, FFT y 195 bandas logarítmicas |
| Datos | 150 muestras | 230 tomas de 4 s con control de pico y SNR |
| Resultado principal | 56,1 % (32 de 57) con el clasificador; 92,1 % (105 de 114) con cents | 100,0 % (283 de 283) en prueba; 97,4 ± 1,5 % en validación cruzada |

Las métricas no son directamente comparables porque la tarea cambió.

## Cómo funciona

1. El micrófono captura audio a 44,1 kHz (un canal) y se corta en **ventanas de 1 s con salto de 0,5 s**.
2. **Rama de identificación:** ventana de Hann y FFT; el espectro se resume en **195 bandas de 25 cents** entre 180 y 3000 Hz, se pasa a decibelios (recorte en −80 dB) y se escala a 0–1. Esos 195 números entran a un MLP que da 14 probabilidades (13 notas y ruido).
3. **Rama de medición:** FFT con relleno de ceros ×4, búsqueda del pico alrededor de los armónicos 1 a 4 de la nota reconocida e interpolación parabólica. De ahí salen los **cents** y la **precisión P**.
4. **Decisión y pantalla:** una ventana de OpenCV muestra nota, cents y color (verde afinada, rojo desafinada).

Reglas de decisión:

- RMS < 0,003 (unos −50 dB): «Sin nota».
- Clase ruido o probabilidad < 0,60: «No reconocida».
- Afinada si P ≥ 70, es decir, |c| ≤ 15 cents; en otro caso, Desafinada.

Los umbrales son decisiones de diseño y pueden necesitar ajuste con otro micrófono.

## La red

Perceptrón multicapa (MLP): **195 → 128 → 64 → 14**.

| Capa | Entradas | Neuronas | Función | Parámetros |
|---|---|---|---|---|
| Entrada | — | 195 | Bandas normalizadas | 0 |
| Capa 1 densa | 195 | 128 | ReLU | 25 088 |
| Dropout 0,3 | 128 | — | Apaga 30 % al entrenar | 0 |
| Capa 2 densa | 128 | 64 | ReLU | 8 256 |
| Capa 3 densa | 64 | 14 | Softmax | 910 |
| **Total** | | | | **34 254** |

Parámetros = neuronas × entradas + un sesgo por neurona. Como referencia se entrena una regresión softmax de 2 744 parámetros.

Entrenamiento: entropía cruzada con regularización L2 (λ = 10⁻⁴), optimizador Adam (tasa 10⁻³, lotes de 32), parada temprana sobre la pérdida de validación (paciencia 15, restaura los mejores pesos) y semilla 42. El entrenamiento se detuvo entre las épocas 31 y 43.

## Fórmulas

| N.º | Fórmula | Qué hace | Dónde se usa |
|---|---|---|---|
| 1 | b_k = 180 · 2^(25k/1200) | Bordes de las 195 bandas | Preprocesamiento |
| 2 | x_k = (máx(L_k, −80) + 80) / 80 | Pasa de dB (respecto del máximo) a escala 0–1 | Preprocesamiento |
| — | h1 = ReLU(W1·x + b1); h2 = ReLU(W2·h1 + b2) | Capas ocultas | MLP |
| — | z = W3·h2 + b3; p_c = e^(z_c) / Σ_j e^(z_j) | Puntajes y softmax | MLP |
| 3 | L = −(1/N) Σ y·log p + λ Σ‖W‖² | Error que minimiza Adam | Entrenamiento |
| 4 | δ = (a − c) / (2(a − 2b + c)) | Posición real del pico (parábola en tres puntos) | Medición de cents |
| 5 | c = 1200 · log₂(f / f_ref) | Frecuencia a cents | Medición de cents |
| 6 | P = máx(0, 100 − 2 · mín(\|c\|, 50)) | Cents a precisión de 0 a 100 | Pantalla y decisión |

## Datos

230 tomas de 4 s, 44,1 kHz, 16 bits, un canal, micrófono de la laptop. Sesión del 22-11-2026.

- **195 tomas de notas:** 13 notas × 3 afinaciones nominales (0, −30 y +30 cents) × 5 tomas, sin vibrato.
- **35 tomas de ruido:** 7 tipos (silencio, sala, roce, voz, clavija, alterada, fuera de rango) × 5 tomas.
- Calidad de captura de las notas: pico medio −10,0 dBFS (DE 3,1), SNR media 27,8 dB (DE 4,1; mínimo 21,2).
- Los cents nominales son intencionales; no se contrastaron con un afinador externo.
- La partición es **por toma** (60/20/20: 138, 46 y 46 tomas) para que ventanas solapadas de una misma toma no queden en conjuntos distintos. Resultan 849 ventanas de entrenamiento originales, más 2 copias con ruido blanco (SNR 10 a 35 dB) = 2 547; 283 de validación y 283 de prueba.

Los audios no se suben al repositorio por su tamaño; se sube `registro.csv` (pico, RMS, SNR, nota estimada y cents de cada toma). Nombres de archivo: `La4_+0_01.wav` (nota, cents nominales, número de toma) y `ruido_0_01_silencio.wav`.

[Link del los datos en drive](https://drive.google.com/file/d/1Kf5LqhsUE8OeNpu5U1BqwfOHHnLjnqcC/view?usp=sharing)

## Resultados

**Clasificación (prueba):** MLP 100,0 % (283 de 283), F1 = 1,00, sin confusiones. Regresión softmax: 97,9 % (277 de 283).

**Experimento 1, representación de entrada** (prueba, 3 semillas):

| Entrada | Entradas | Exactitud | Mínimo |
|---|---|---|---|
| Forma de onda | 11 025 | 41,5 ± 1,1 % | 39,9 % |
| FFT lineal | 2 821 | 99,8 ± 0,2 % | 99,6 % |
| Bandas sin Hann | 195 | 100,0 ± 0,0 % | 100,0 % |
| Bandas con Hann | 195 | 99,4 ± 0,8 % | 98,2 % |

Sin Fourier la red acierta 41,5 %; con cualquier espectro, 99,4 % o más. Entre espectros la diferencia es de 0,6 puntos o menos (unas 2 ventanas de 283), sin ventaja demostrada.

**Experimento 2, validación cruzada de 5 pliegues por toma:** 97,2; 97,9; 97,9; 94,7 y 99,3 %. Media 97,4 ± 1,5 %. La prueba (100,0 %) es 2,6 puntos más optimista.

**Experimento 3, ruido blanco añadido (MLP frente a pYIN, 92 ventanas de prueba):**

| SNR | Sin ruido | 30 dB | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|---|
| MLP | 100,0 % | 98,9 % | 98,9 % | 97,8 % | 69,6 % | 23,9 % |
| pYIN | 88,0 % | 87,0 % | 88,0 % | 90,2 % | 87,0 % | 53,3 % |

**Experimento 4, rango de aumento con ruido (283 ventanas de prueba):**

| Entrenado con | Sin ruido | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| SNR 10 a 35 dB (modelo final) | 98,6 % | 98,9 % | 100,0 % | 95,1 % | 44,5 % |
| SNR 0 a 35 dB | 97,9 % | 97,2 % | 98,6 % | 98,9 % | 92,9 % |

Se reporta como experimento; el modelo final conserva el rango de 10 a 35 dB.

**Estimadores de cents (390 ventanas):**

| Estimador | Detectadas | EAM (cents) | RECM (cents) | Tiempo |
|---|---|---|---|---|
| FFT con armónicos | 100,0 % (390) | 6,41 | 8,43 | 3,7 ms |
| FFT con pico único | 99,7 % (389) | 6,75 | 9,30 | 3,8 ms |
| pYIN | 96,4 % (376) | 5,54 | 6,95 | 395 ms |

El error se mide frente a los cents nominales, por lo que incluye la desviación de la ejecutante (mediana de la desviación del tono por toma: 6,1 a 6,6 cents). pYIN tarda 107 veces más y consume 79 % del salto de 0,5 s.

**Sistema extremo a extremo** (mismo código de la demostración): 99,6 % (233 de 234) de notas bien identificadas en prueba, 100,0 % (245 de 245) de ventanas de ruido rechazadas y 98,5 % (1 149 de 1 166) de decisiones correctas entre afinada y desafinada. Cents medidos: afinadas +2,5 (DE 6,4), +30 nominales +36,1 (DE 7,0), −30 nominales −31,8 (DE 6,1).

> La latencia de 71,7 ms por ventana (percentil 95: 122 ms) se midió en el entorno de análisis, no en la laptop de captura. Para obtener la tuya, ejecuta `experimentos_fase2.py` con la tarea `sistema` (ver abajo).

## Límites

Un violín, una ejecutante, un micrófono y una sesión; sin vibrato; cents nominales sin afinador de referencia; solo notas naturales; ruido blanco como única perturbación; latencia medida fuera de la laptop de captura.

## Estructura

```
fase2/
├── src/
│   ├── grabar_dataset.py            # grabación guiada de las tomas y registro.csv
│   ├── entrenar_modelo_notas.py     # preprocesamiento, MLP, entrenamiento y evaluación
│   ├── afinador_visual_v2.py        # demostración en tiempo real (audio + OpenCV)
│   ├── comparar_estimadores.py      # FFT con armónicos, pico único y pYIN
│   ├── experimentos_fase2.py        # ablación, ruido, vibrato, acuerdo, aumento, sistema
│   └── kfold_tomas.py               # validación cruzada de 5 pliegues por toma
├── modelos/
│   ├── modelo_notas.keras           # modelo entrenado
│   └── config_modelo.json           # clases, frecuencias y parámetros del preprocesamiento
├── resultados/                      # reportes, matriz de confusión, curvas, tablas CSV
└── datos/Dataset_Notas/             # audios y registro.csv (los .wav no se suben)
```

## Cómo ejecutarlo

Requisitos: Python con TensorFlow 2.21 (Keras), NumPy, SciPy, scikit-learn, pandas, matplotlib, soundfile, librosa, PyAudio y OpenCV.

Todos los comandos se ejecutan desde `fase2`.

```powershell
# 1. Listar y probar el micrófono (en mi equipo funcionó el dispositivo 1, a 44 100 Hz)
python src\grabar_dataset.py --listar
python src\grabar_dataset.py --probar --dispositivo 1

# 2. Grabar el dataset
python src\grabar_dataset.py --dispositivo 1

# 3. Entrenar y evaluar el modelo
python src\entrenar_modelo_notas.py datos\Dataset_Notas

# 4. Demostración en tiempo real
python src\afinador_visual_v2.py --modelo modelos\modelo_notas.keras --dispositivo 1

# 5. Experimentos (tareas: ablacion, ruido, vibrato, acuerdo, aumento, sistema)
python src\experimentos_fase2.py datos\Dataset_Notas ablacion
python src\experimentos_fase2.py datos\Dataset_Notas sistema

# 6. Validación cruzada por toma (se ejecuta dentro de datos)
cd datos
python ..\src\kfold_tomas.py
```

Notas:

- Con `--wav archivo.wav` y `--sin-camara`, `afinador_visual_v2.py` analiza un archivo sin micrófono.
- El reentrenamiento no es idéntico entre ejecuciones (por ejemplo, 31 o 43 épocas con la misma semilla); los resultados se reportan como rango.
- Si Windows entrega silencio, revisa que el micrófono no esté silenciado y la privacidad del micrófono para aplicaciones de escritorio.
- El modelo se entrenó con TensorFlow 2.21; con otra versión puede ser necesario reentrenarlo.

## Referencias

Carney et al. (2020); Cooley y Tukey (1965); de Cheveigné y Kawahara (2002); Harris (1978); Jacobsen y Kootsookos (2007); Kim et al. (2018); Kingma y Ba (2015); Mauch y Dixon (2014); McFee et al. (2015); Mesaros et al. (2018); Moreno-Torres et al. (2012); Salamon y Gómez (2012); Srivastava et al. (2014); Woodhouse (2014). Referencias completas con DOI en el artículo para la Revista 3i.
