import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

resultados = []

for anio in range(2015, 2025):

    df = pd.read_csv(
        f"../FrecuenciaEHistograma/frecuencias_{anio}.csv"
    )

    distribucion = (
        df["count"]
        .value_counts()
        .sort_index()
    )

    # Analizar frecuencias de 1 a 20
    distribucion = distribucion[
        distribucion.index <= 20
    ]

    x = distribucion.index.to_numpy()
    y = distribucion.values

    # ---------------------------------------------
    # Normalización
    # ---------------------------------------------

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    punto_inicial = np.array([
        x_norm[0],
        y_norm[0]
    ])

    punto_final = np.array([
        x_norm[-1],
        y_norm[-1]
    ])

    vector_recta = punto_final - punto_inicial
    longitud_recta = np.linalg.norm(vector_recta)

    # ---------------------------------------------
    # Distancia de cada punto a la recta
    # ---------------------------------------------

    distancias = []

    for xi, yi in zip(x_norm, y_norm):

        punto = np.array([xi, yi])
        diferencia = punto - punto_inicial

        producto_cruzado = (
            vector_recta[0] * diferencia[1]
            - vector_recta[1] * diferencia[0]
        )

        distancia = (
            abs(producto_cruzado)
            / longitud_recta
        )

        distancias.append(distancia)

    # ---------------------------------------------
    # Detectar codo
    # ---------------------------------------------

    indice_codo = np.argmax(distancias)
    codo = int(x[indice_codo])

    resultados.append({
        "anio": anio,
        "codo": codo
    })

    print(
        f"{anio} -> codo = {codo}"
    )

    # ---------------------------------------------
    # Gráfica
    # ---------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        y,
        marker="o",
        label="Distribución"
    )

    plt.axvline(
        x=codo,
        linestyle="--",
        label=f"Codo = {codo}"
    )

    plt.scatter(
        codo,
        y[indice_codo],
        s=100,
        zorder=5
    )

    plt.xlabel(
        "Número de apariciones (count)"
    )

    plt.ylabel(
        "Número de palabras"
    )

    plt.title(
        f"Curva de distribución de frecuencias - {anio}"
    )

    plt.xticks(range(1, 21))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"../FrecuenciaEHistograma/curva_codo_{anio}.png",
        dpi=300
    )

    plt.close()


df_resultados = pd.DataFrame(resultados)

df_resultados.to_csv(
    "../FrecuenciaEHistograma/codos_por_anio.csv",
    index=False
)

print("\nResumen:")
print(df_resultados)