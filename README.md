# Laboratorio 3
## Análisis espectral de la voz

**Programa:** Ingeniería Biomédica  
**Asignatura:** Procesamiento Digital de Señales  
**Universidad:** Universidad Militar Nueva Granada  
**Estudiantes:** Danna Rivera, Duvan Paez

---

## Introducción
 
Las características espectrales desempeñan un papel fundamental en el análisis y comprensión de las señales de voz, ya que permiten identificar patrones fonéticos, rasgos del hablante y diferencias fisiológicas entre géneros. En esta práctica se aplican herramientas de procesamiento digital de señales para extraer y comparar parámetros espectrales de voces masculinas y femeninas.

Los parámetros analizados son:
 
| Parámetro | Descripción |
|---|---|
| **F0 — Frecuencia fundamental** | Frecuencia más baja de la señal; define la altura tonal de la voz |
| **Centroide espectral** | Centro de masa del espectro; indica el brillo o timbre de la voz |
| **Intensidad RMS** | Energía promedio de la señal de voz |
| **Jitter relativo** | Variación ciclo a ciclo en la frecuencia fundamental |
| **Shimmer relativo** | Variación ciclo a ciclo en la amplitud de la señal |
 
El **jitter** y el **shimmer** son indicadores de estabilidad vocal; en voces sanas sus valores típicos son ≤ 1% y ≤ 3–5% respectivamente. Valores elevados pueden estar asociados a ruido de grabación, condiciones acústicas deficientes o patologías vocales.
 
---

## PARTE A – Adquisición de las señales de voz

### Adquisición de señales
Se grabaron 6 señales de voz (3 hombres, 3 mujeres) pronunciando la misma frase corta (~5 segundos) en condiciones de bajo ruido ambiental, usando el micrófono de teléfonos inteligentes. Todos los archivos se guardaron en formato `.wav` con la misma frecuencia de muestreo.

 ### Procesamiento en Python
 
El análisis se realizó con las siguientes bibliotecas:
 
```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav
import pandas as pd
```

#### 1. Eliminación de silencios
Cada señal fue normalizada entre −1 y 1, y se eliminaron los silencios al inicio y al final mediante un umbral de amplitud:
 
```python
def eliminar_silencio(audio, threshold=0.02):
    idx = np.where(np.abs(audio) > threshold)[0]
    if len(idx) == 0:
        return audio
    return audio[idx[0]:idx[-1]]
```

#### 2. Transformada de Fourier (FFT) y PSD
Se calculó el espectro de magnitud mediante FFT y la densidad espectral de potencia con el método de Welch (`nperseg=1024`):
 
```python
fft_audio = np.fft.fft(audio)
freqs     = np.fft.fftfreq(N, 1/fs)
magnitud  = np.abs(fft_audio)
 
f_psd, psd = signal.welch(audio, fs, nperseg=1024)
```

#### 3. Frecuencia fundamental (F0)
Se estimó mediante autocorrelación de la señal:
 
```python
def calcular_f0(audio, fs):
    corr  = signal.correlate(audio, audio)
    corr  = corr[len(corr)//2:]
    d     = np.diff(corr)
    start = np.where(d > 0)[0][0]
    peak  = np.argmax(corr[start:]) + start
    return 1 / (peak / fs)
```

#### 4. Centroide espectral (Brillo)
Promedio ponderado de las frecuencias por su magnitud espectral, utilizado como estimador de la frecuencia media:
 
```python
centroide = np.sum(frec_pos * mag_pos) / np.sum(mag_pos)
```

#### 5. Intensidad RMS
```python
rms = np.sqrt(np.mean(audio**2))
```
 
## PARTE B – Medición de Jitter y Shimmer
 
 
