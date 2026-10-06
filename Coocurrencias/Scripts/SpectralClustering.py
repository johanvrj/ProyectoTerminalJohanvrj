import pandas as pd
import numpy as np
from scipy.linalg import eigh


# -------------------------------------------------
# Configuración
# -------------------------------------------------

ANIO = 2016

K_MIN = 2
K_MAX = 20

SEED = 42
N_INIT = 20
MAX_ITER = 300

def normalizar_filas(X):

    normas = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    normas[normas == 0] = 1

    return X / normas


def kmeans_numpy(
    X,
    k,
    seed=42,
    n_init=20,
    max_iter=300
):

    mejor_etiquetas = None
    mejor_inercia = np.inf

    for intento in range(n_init):

        rng = np.random.default_rng(
            seed + intento
        )

        indices = rng.choice(
            len(X),
            size=k,
            replace=False
        )

        centroides = X[indices].copy()

        etiquetas = None

        for _ in range(max_iter):

            distancias = np.sum(
                (
                    X[:, None, :]
                    -
                    centroides[None, :, :]
                ) ** 2,
                axis=2
            )

            nuevas_etiquetas = np.argmin(
                distancias,
                axis=1
            )

            if (
                etiquetas is not None
                and
                np.array_equal(
                    nuevas_etiquetas,
                    etiquetas
                )
            ):
                break

            etiquetas = nuevas_etiquetas

            nuevos_centroides = []

            for cluster in range(k):

                puntos = X[
                    etiquetas == cluster
                ]

                if len(puntos) == 0:

                    nuevo = X[
                        rng.integers(len(X))
                    ]

                else:

                    nuevo = puntos.mean(
                        axis=0
                    )

                nuevos_centroides.append(
                    nuevo
                )

            centroides = np.array(
                nuevos_centroides
            )

        distancias_finales = np.sum(
            (
                X
                -
                centroides[etiquetas]
            ) ** 2,
            axis=1
        )

        inercia = np.sum(
            distancias_finales
        )

        if inercia < mejor_inercia:

            mejor_inercia = inercia
            mejor_etiquetas = (
                etiquetas.copy()
            )

    return mejor_etiquetas


def silhouette_coseno(
    X,
    etiquetas
):

    # X debe estar normalizado.
    # Distancia coseno = 1 - similitud coseno

    similitud = X @ X.T

    distancias = 1 - similitud

    distancias = np.clip(
        distancias,
        0,
        2
    )

    valores = []

    clusters = np.unique(
        etiquetas
    )

    for i in range(len(X)):

        cluster_actual = etiquetas[i]

        mascara_mismo = (
            etiquetas == cluster_actual
        )

        # Excluir el propio punto
        mascara_mismo[i] = False

        if np.sum(mascara_mismo) == 0:

            valores.append(0.0)
            continue

        # Distancia media dentro del cluster
        a = np.mean(
            distancias[
                i,
                mascara_mismo
            ]
        )

        # Distancia media al cluster
        # externo más cercano
        b = np.inf

        for cluster in clusters:

            if cluster == cluster_actual:
                continue

            mascara_otro = (
                etiquetas == cluster
            )

            distancia_media = np.mean(
                distancias[
                    i,
                    mascara_otro
                ]
            )

            if distancia_media < b:
                b = distancia_media

        denominador = max(a, b)

        if denominador == 0:

            s = 0.0

        else:

            s = (
                (b - a)
                /
                denominador
            )

        valores.append(s)

    return np.mean(valores)


# -------------------------------------------------
# Cargar embeddings
# -------------------------------------------------

ruta_embeddings = (
    f"SkipGram/"
    f"embeddings_{ANIO}.csv"
)

print(
    f"Cargando embeddings de {ANIO}..."
)

df = pd.read_csv(
    ruta_embeddings
)

keywords = df[
    "keyword"
].tolist()

X = df.drop(
    columns=["keyword"]
).values.astype(float)


print(
    f"-> Keywords: {len(keywords)}"
)

print(
    f"-> Dimensiones: {X.shape[1]}"
)


# -------------------------------------------------
# Normalizar embeddings
# -------------------------------------------------

X_normalizado = normalizar_filas(
    X
)


# -------------------------------------------------
# Matriz de similitud coseno
# -------------------------------------------------

print(
    "\nConstruyendo matriz de similitud..."
)

W = (
    X_normalizado
    @
    X_normalizado.T
)

# Spectral Clustering necesita
# afinidades no negativas.

W[W < 0] = 0

W = np.clip(
    W,
    0,
    1
)

# No necesitamos auto-conexiones
# para construir el grafo.

np.fill_diagonal(
    W,
    0
)


print(
    f"-> Matriz: {W.shape}"
)


# -------------------------------------------------
# Laplaciano normalizado
# -------------------------------------------------

print(
    "\nConstruyendo Laplaciano normalizado..."
)

grados = W.sum(
    axis=1
)

grados[
    grados == 0
] = 1e-12

D_inv_sqrt = np.diag(
    1.0
    /
    np.sqrt(grados)
)

L = (
    np.eye(len(W))
    -
    D_inv_sqrt
    @ W
    @ D_inv_sqrt
)


# -------------------------------------------------
# Autovalores y autovectores
# -------------------------------------------------

print(
    "Calculando autovectores..."
)

# Solo necesitamos los primeros K_MAX
# autovectores asociados a los
# autovalores más pequeños.

autovalores, autovectores = eigh(
    L,
    subset_by_index=[
        0,
        K_MAX - 1
    ]
)


print(
    "\nPrimeros autovalores:"
)

for i, valor in enumerate(
    autovalores
):

    print(
        f"{i + 1:2d}: "
        f"{valor:.6f}"
    )


# -------------------------------------------------
# Evaluar distintos valores de k
# -------------------------------------------------

print(
    "\nProbando valores de k..."
)

resultados = []


for k in range(
    K_MIN,
    K_MAX + 1
):

    print(
        f"\nk = {k}"
    )

    # Primeros k autovectores

    U = autovectores[
        :,
        :k
    ]

    # Normalización por fila
    # de la representación espectral

    U = normalizar_filas(
        U
    )

    # K-Means implementado con NumPy

    etiquetas = kmeans_numpy(
        U,
        k,
        seed=SEED,
        n_init=N_INIT,
        max_iter=MAX_ITER
    )

    # Evaluamos los grupos sobre
    # los embeddings Skip-gram
    # mediante distancia coseno.

    score = silhouette_coseno(
        X_normalizado,
        etiquetas
    )

    print(
        f"Silhouette Score: "
        f"{score:.4f}"
    )

    resultados.append(
        {
            "k": k,
            "silhouette_score": score
        }
    )


df_resultados = pd.DataFrame(
    resultados
)

df_resultados = (
    df_resultados
    .sort_values(
        "silhouette_score",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\nResultados ordenados:"
)

print(
    df_resultados.to_string(
        index=False
    )
)


# -------------------------------------------------
# Mejor k
# -------------------------------------------------

mejor_k = int(
    df_resultados.loc[
        0,
        "k"
    ]
)

mejor_score = float(
    df_resultados.loc[
        0,
        "silhouette_score"
    ]
)


print(
    "\nMejor resultado:"
)

print(
    f"-> k = {mejor_k}"
)

print(
    f"-> Silhouette Score = "
    f"{mejor_score:.4f}"
)

ruta_salida = (
    f"SkipGram/"
    f"evaluacion_clusters_{ANIO}.csv"
)

df_resultados.to_csv(
    ruta_salida,
    index=False
)


print(
    f"\nResultados guardados en: "
    f"{ruta_salida}"
)

print(
    "\nProceso completado."
)