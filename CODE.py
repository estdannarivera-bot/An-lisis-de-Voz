import numpy as np                  # Biblioteca para cálculos numéricos y manejo de vectores
import matplotlib.pyplot as plt     # Biblioteca para generar gráficas
import scipy.signal as signal       # Herramientas de procesamiento de señales (filtros, correlación)
import scipy.io.wavfile as wav      # Permite leer archivos de audio .wav
import pandas as pd                 # Permite crear tablas de resultados

# ARCHIVOS DE AUDIO

hombre1 = "hombre1.wav"    
hombre2 = "hombre2.wav"
hombre3 = "hombre3.wav"

mujer1 = "mujer1.wav"
mujer2 = "mujer2.wav"
mujer3 = "mujer3.wav"


# FUNCION PARA ELIMINAR SILENCIOS

def eliminar_silencio(audio, threshold=0.02):

    # Se busca en qué posiciones del audio la amplitud es mayor al umbral
    # np.abs() toma el valor absoluto porque la señal puede ser positiva o negativa
    idx = np.where(np.abs(audio) > threshold)[0]

    # Si no se encuentra ningún punto que supere el umbral
    # se retorna el audio original
    if len(idx) == 0:
        return audio

    # Primer punto donde la señal deja de ser silencio
    inicio = idx[0]

    # Último punto donde la señal sigue siendo diferente de silencio
    fin = idx[-1]

    # Se recorta la señal entre esos dos puntos
    return audio[inicio:fin]


# FUNCION PARA CALCULAR FRECUENCIA FUNDAMENTAL (F0)

def calcular_f0(audio, fs):

    # Autocorrelación de la señal consigo misma
    # Esto permite detectar periodicidad (patrones repetidos)
    corr = signal.correlate(audio, audio)

    # La autocorrelación es simétrica
    # solo usamos la mitad positiva
    corr = corr[len(corr)//2:]

    # Diferencia entre puntos consecutivos
    # sirve para detectar donde la pendiente vuelve a crecer
    d = np.diff(corr)

    # Encontrar el primer punto donde la pendiente es positiva
    # ese punto indica el inicio del primer ciclo real
    start = np.where(d > 0)[0][0]

    # Buscar el máximo después de ese punto
    # ese máximo corresponde al periodo fundamental
    peak = np.argmax(corr[start:]) + start

    # Convertir ese índice en tiempo
    periodo = peak / fs

    # F0 = 1 / periodo
    f0 = 1 / periodo

    return f0


# FUNCION DE ANALISIS

def analizar_voz(nombre_archivo, nombre_persona, genero="M"):

    print("\nAnalizando:", nombre_archivo)

    # CARGAR AUDIO
    # fs = frecuencia de muestreo
    # audio = vector con las muestras del audio
    fs, audio = wav.read(nombre_archivo)

    # Convertir a tipo float para poder hacer operaciones matemáticas
    audio = audio.astype(float)

    # Normalizar la señal entre -1 y 1
    audio = audio / np.max(np.abs(audio))

    # Eliminar silencios al inicio y final
    audio = eliminar_silencio(audio)

    # Crear vector de tiempo
    t = np.arange(len(audio)) / fs
    
    
    # TRANSFORMADA DE FOURIER

    # Número de muestras de la señal
    N = len(audio)

    # Transformada rápida de Fourier
    fft_audio = np.fft.fft(audio)

    # Vector de frecuencias correspondiente
    freqs = np.fft.fftfreq(N, 1/fs)

    # Magnitud del espectro
    magnitud = np.abs(fft_audio)

    # Solo usamos frecuencias positivas
    frec_pos = freqs[:N//2]
    mag_pos = magnitud[:N//2]

    # PSD (método de Welch)
    f_psd, psd = signal.welch(audio, fs, nperseg=1024)


     # GRAFICAS (SEÑAL + FFT + PSD)

    plt.figure(figsize=(10,8))
    
    # Señal en el tiempo
    plt.subplot(3,1,1)
    plt.plot(t, audio)
    plt.title(f"Señal de voz - {nombre_persona}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid()
    
    # Espectro FFT
    plt.subplot(3,1,2)
    plt.semilogx(frec_pos, mag_pos)
    plt.title("Espectro de magnitud (FFT)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.xlim(80,4000)
    plt.grid()
    
    # PSD
    plt.subplot(3,1,3)
    plt.semilogx(f_psd, psd)
    plt.title("Densidad espectral de potencia (PSD)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Potencia")
    plt.xlim(80,4000)
    plt.grid()
    
    plt.tight_layout()
    plt.show()



    # FRECUENCIA FUNDAMENTAL

    f0 = calcular_f0(audio, fs)
    print("Frecuencia fundamental F0:", f0, "Hz")


    # CENTROIDE ESPECTRAL o BRILLO

    # El centroide indica donde se concentra la energía del espectro
    # se calcula como un promedio ponderado por magnitud
    centroide = np.sum(frec_pos * mag_pos) / np.sum(mag_pos)
    print("Centroide espectral:", centroide, "Hz")


    # INTENSIDAD RMS

    # RMS mide la energía promedio de la señal
    rms = np.sqrt(np.mean(audio**2))
    print("Intensidad RMS:", rms)

    # FILTRO PASA BANDA
 # rango de filtro según género 
    if genero == "M":
        low, high = 80, 400      # hombres: 80–400 Hz (guía)
    else:
        low, high = 150, 500     # mujeres: 150–500 Hz (guía)


    # Diseño de filtro Butterworth
    b, a = signal.butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    audio_filtrado = signal.filtfilt(b, a, audio)
    
    audio_env = np.abs(signal.hilbert(audio_filtrado))

    # DETECCION DE PICOS (CICLOS DE VOZ)

    # Distancia mínima entre picos
    # evita detectar demasiados picos
    # deteccion de ciclos
    distancia_picos = int(fs / f0 * 0.8)
    peaks, _ = signal.find_peaks(
        audio_env,
        distance=distancia_picos,
        prominence=0.01
    )



    # JITTER

    
    if len(peaks) > 2:
        # Periodos Ti = tiempo entre picos consecutivos
        periodos = np.diff(peaks) / fs

        # Filtro de outliers: descartar periodos > 20 % del promedio
        media_T = np.mean(periodos)
        periodos = periodos[np.abs(periodos - media_T) < 0.2 * media_T]

        if len(periodos) > 1:
            # Jitter_abs = (1/N-1) * Σ|Ti - Ti+1|
            jitter_abs = np.mean(np.abs(np.diff(periodos)))
            # Jitter_rel = Jitter_abs / mean(Ti) * 100
            jitter_rel = (jitter_abs / np.mean(periodos)) * 100
        else:
            jitter_rel = 0
    else:
        jitter_rel = 0

    print("Jitter relativo:", jitter_rel, "%")


    #  SHIMMER 
    if len(peaks) > 2:
        # ── CORRECCIÓN 4: amplitudes desde la ENVOLVENTE, no la señal filtrada
        amplitudes = audio_env[peaks]

        # Filtro de outliers
        media_A = np.mean(amplitudes)
        amplitudes = amplitudes[np.abs(amplitudes - media_A) < 0.5 * media_A]

        if len(amplitudes) > 1:
            # Shimmer_abs = (1/N-1) * Σ|Ai - Ai+1|
            shimmer_abs = np.mean(np.abs(np.diff(amplitudes)))
            # Shimmer_rel = Shimmer_abs / mean(Ai) * 100
            shimmer_rel = (shimmer_abs / np.mean(amplitudes)) * 100
        else:
            shimmer_rel = 0
    else:
        shimmer_rel = 0

    print("Shimmer relativo:", shimmer_rel, "%")


    # RESULTADOS
    resultados = {
        "F0": f0,
        "Centroide": centroide,
        "RMS": rms,
        "Jitter (%)": jitter_rel,
        "Shimmer (%)": shimmer_rel
    }

    return resultados


# ANALISIS DE LAS VOCES

resultados = []

res=analizar_voz(hombre1,"Hombre 1",genero="M")
resultados.append(["hombre1","M"]+list(res.values()))

res=analizar_voz(hombre2,"Hombre 2",genero="M")
resultados.append(["hombre2","M"]+list(res.values()))

res=analizar_voz(hombre3,"Hombre 3",genero="M")
resultados.append(["hombre3","M"]+list(res.values()))

res=analizar_voz(mujer1,"Mujer 1",genero="F")
resultados.append(["mujer1","F"]+list(res.values()))

res=analizar_voz(mujer2,"Mujer 2",genero="F")
resultados.append(["mujer2","F"]+list(res.values()))

res=analizar_voz(mujer3,"Mujer 3",genero="F")
resultados.append(["mujer3","F"]+list(res.values()))

# TABLA FINAL

columnas = [
    "Audio",
    "Genero",
    "F0",
    "Centroide",
    "RMS",
    "Jitter (%)",
    "Shimmer (%)"
]

tabla = pd.DataFrame(resultados, columns=columnas)

print("\nTabla de resultados")
print(tabla)

# GRÁFICO DE BARRAS CON PUNTOS - ESTILO MINIMALISTA

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("Comparación Hombres vs Mujeres", fontsize=14, fontweight='bold', y=1.02)

parametros = ["F0", "Centroide", "RMS", "Jitter (%)", "Shimmer (%)"]
labels     = ["F0 (Hz)", "Centroide (Hz)", "RMS", "Jitter (%)", "Shimmer (%)"]

color_H = "#4C72B0"   # azul sobrio
color_F = "#DD8452"   # naranja sobrio

for i, (param, label) in enumerate(zip(parametros, labels)):
    ax = axes[i]

    vals_H = tabla[tabla["Genero"] == "M"][param].values
    vals_F = tabla[tabla["Genero"] == "F"][param].values

    # Barras con la media
    ax.bar([0], [np.mean(vals_H)], color=color_H, alpha=0.6, width=0.5, label="Hombres")
    ax.bar([1], [np.mean(vals_F)], color=color_F, alpha=0.6, width=0.5, label="Mujeres")

    # Puntos individuales encima
    ax.scatter(np.zeros(len(vals_H)), vals_H, color=color_H, zorder=5, s=60, edgecolors='white', linewidths=0.8)
    ax.scatter(np.ones(len(vals_F)),  vals_F, color=color_F, zorder=5, s=60, edgecolors='white', linewidths=0.8)

    # Barra de error (desviación estándar)
    ax.errorbar(0, np.mean(vals_H), yerr=np.std(vals_H), fmt='none', color='black', capsize=5, linewidth=1.2)
    ax.errorbar(1, np.mean(vals_F), yerr=np.std(vals_F), fmt='none', color='black', capsize=5, linewidth=1.2)

    # Estilo minimalista
    ax.set_title(label, fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hombres", "Mujeres"], fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(left=True, bottom=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_xlim(-0.5, 1.5)

# Leyenda global
handles = [
    plt.Rectangle((0,0),1,1, color=color_H, alpha=0.6),
    plt.Rectangle((0,0),1,1, color=color_F, alpha=0.6)
]
fig.legend(handles, ["Hombres", "Mujeres"], loc="upper right",
           frameon=False, fontsize=10, bbox_to_anchor=(1.01, 1.01))

plt.tight_layout()
plt.show()