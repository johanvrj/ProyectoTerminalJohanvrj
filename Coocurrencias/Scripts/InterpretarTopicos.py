import pandas as pd
import numpy as np
from scipy.linalg import eigh
import os


# -------------------------------------------------
# Configuración
# -------------------------------------------------

ANIO = 2016

K_VALORES = [17, 18, 19]

TOP_N = 15

SEED = 42
N_INIT = 20
MAX_ITER = 300


# -------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------

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
                        rng.integers(
                            len(X)
                        )
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
                centroides[
                    etiquetas
                ]
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

keywords = np.array(
    df["keyword"].tolist()
)

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
# Matriz de afinidad
# -------------------------------------------------

print(
    "\nConstruyendo matriz de similitud..."
)

W = (
    X_normalizado
    @
    X_normalizado.T
)

W[W < 0] = 0

W = np.clip(
    W,
    0,
    1
)

np.fill_diagonal(
    W,
    0
)


# -------------------------------------------------
# Laplaciano normalizado
# -------------------------------------------------

print(
    "Construyendo Laplaciano..."
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
# Autovectores
# -------------------------------------------------

K_MAX = max(
    K_VALORES
)

print(
    f"Calculando los primeros "
    f"{K_MAX} autovectores..."
)

autovalores, autovectores = eigh(
    L,
    subset_by_index=[
        0,
        K_MAX - 1
    ]
)


os.makedirs(
    "SkipGram",
    exist_ok=True
)


# -------------------------------------------------
# Analizar cada valor de k
# -------------------------------------------------

for K in K_VALORES:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"ANÁLISIS PARA k = {K}"
    )

    print(
        "=" * 70
    )

    # ---------------------------------------------
    # Representación espectral
    # ---------------------------------------------

    U = autovectores[
        :,
        :K
    ]

    U = normalizar_filas(
        U
    )


    # ---------------------------------------------
    # K-Means
    # ---------------------------------------------

    etiquetas = kmeans_numpy(
        U,
        K,
        seed=SEED,
        n_init=N_INIT,
        max_iter=MAX_ITER
    )


    # ---------------------------------------------
    # Tabla de asignaciones
    # ---------------------------------------------

    df_clusters = pd.DataFrame(
        {
            "keyword": keywords,
            "cluster": etiquetas + 1
        }
    )


    # ---------------------------------------------
    # Keywords representativas
    # ---------------------------------------------

    resultados_top = []

    for cluster in range(K):

        indices = np.where(
            etiquetas == cluster
        )[0]

        vectores_cluster = (
            X_normalizado[
                indices
            ]
        )

        palabras_cluster = (
            keywords[
                indices
            ]
        )

        centroide = (
            vectores_cluster.mean(
                axis=0
            )
        )

        norma = np.linalg.norm(
            centroide
        )

        if norma != 0:

            centroide = (
                centroide
                /
                norma
            )

        similitudes = (
            vectores_cluster
            @
            centroide
        )

        orden = np.argsort(
            similitudes
        )[::-1]

        cantidad_top = min(
            TOP_N,
            len(orden)
        )


        print(
            f"\nCluster {cluster + 1}"
            f" ({len(indices)} keywords)"
        )

        print(
            "-" * 55
        )


        for posicion in orden[
            :cantidad_top
        ]:

            palabra = (
                palabras_cluster[
                    posicion
                ]
            )

            similitud = float(
                similitudes[
                    posicion
                ]
            )

            print(
                f"{palabra:<42} "
                f"{similitud:.4f}"
            )

            resultados_top.append(
                {
                    "k": K,
                    "cluster":
                        cluster + 1,

                    "keyword":
                        palabra,

                    "similitud_centro":
                        similitud,

                    "tamano_cluster":
                        len(indices)
                }
            )

    ruta_clusters = (
        f"SkipGram/"
        f"clusters_{ANIO}_k{K}.csv"
    )

    df_clusters.to_csv(
        ruta_clusters,
        index=False
    )

    df_top = pd.DataFrame(
        resultados_top
    )

    ruta_top = (
        f"SkipGram/"
        f"top_keywords_{ANIO}_k{K}.csv"
    )

    df_top.to_csv(
        ruta_top,
        index=False
    )

    print(
        "\nTamaños:"
    )

    tamanos = (
        df_clusters[
            "cluster"
        ]
        .value_counts()
        .sort_index()
    )

    for cluster, cantidad in tamanos.items():

        print(
            f"Cluster {cluster:2d}: "
            f"{cantidad:3d}"
        )


    print(
        f"\nGuardado:"
    )

    print(
        f"-> {ruta_clusters}"
    )

    print(
        f"-> {ruta_top}"
    )

print(
    "\n"
    + "=" * 70
)

print(
    "COMPARACIÓN COMPLETADA"
)

print(
    "=" * 70
)

print(
    "\nSe generaron resultados para:"
)

for K in K_VALORES:

    print(
        f"-> k = {K}"
    )