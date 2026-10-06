import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from gensim.models import Word2Vec

import nltk
from nltk.stem import WordNetLemmatizer


# =========================================================
# CONFIGURACIÓN
# =========================================================

ANIOS = range(2015, 2025)

FRECUENCIA_MINIMA = 3

VECTOR_SIZE = 100
EPOCHS = 100
SEED = 42

K_MIN = 2
K_MAX = 20
NUM_CANDIDATOS = 3

N_INIT = 20
MAX_ITER = 300

TOP_N = 15

CARPETA_FRECUENCIAS = "FrecuenciaEHistograma"
CARPETA_SKIPGRAM = "SkipGram"

os.makedirs(
    CARPETA_SKIPGRAM,
    exist_ok=True
)


# =========================================================
# NLTK
# =========================================================

nltk.download(
    "wordnet",
    quiet=True
)

nltk.download(
    "omw-1.4",
    quiet=True
)

lemmatizer = WordNetLemmatizer()


# =========================================================
# CARGAR DICCIONARIO DE LIMPIEZA
# =========================================================

print("Cargando diccionario de limpieza...")

df_words = pd.read_csv(
    "words_actualizado.csv"
)

df_words["words"] = (
    df_words["words"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df_words["action"] = (
    df_words["action"]
    .fillna("")
    .astype(str)
    .str.strip()
)

palabras_prohibidas = set(
    df_words.loc[
        df_words["action"].str.lower() == "x",
        "words"
    ]
)

sustituciones = {}

for _, fila in df_words.iterrows():

    palabra = fila["words"]
    accion = fila["action"]

    if (
        accion
        and
        accion.lower() != "x"
    ):
        sustituciones[palabra] = (
            accion.lower()
        )

# =========================================================
# LIMPIEZA DE KEYWORDS
# =========================================================

def limpiar_palabra(palabra):

    palabra = str(
        palabra
    ).lower().strip()

    palabra = re.sub(
        r"\([^)]*\)",
        "",
        palabra
    )

    palabra = palabra.replace(
        "(",
        ""
    ).replace(
        ")",
        ""
    )

    palabra = palabra.replace(
        "-",
        "_"
    )

    palabra = palabra.strip()

    if palabra in palabras_prohibidas:
        return None

    if palabra in sustituciones:
        palabra = sustituciones[
            palabra
        ]

    if re.search(
        r"\d",
        palabra
    ):
        return None

    palabra = palabra.replace(
        "&",
        "and"
    )

    palabra = re.sub(
        r'[<>"\']',
        "",
        palabra
    )

    partes = palabra.split()

    partes = [
        lemmatizer.lemmatize(
            parte
        )
        for parte in partes
    ]

    palabra = "_".join(
        partes
    )

    palabra = palabra.strip(
        "_"
    )

    if not palabra:
        return None

    if palabra in palabras_prohibidas:
        return None

    return palabra


# =========================================================
# FUNCIONES MATEMÁTICAS
# =========================================================

def normalizar_filas(X):

    normas = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    normas[
        normas == 0
    ] = 1

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

    for intento in range(
        n_init
    ):

        rng = np.random.default_rng(
            seed + intento
        )

        indices = rng.choice(
            len(X),
            size=k,
            replace=False
        )

        centroides = X[
            indices
        ].copy()

        etiquetas = None

        for _ in range(
            max_iter
        ):

            distancias = np.sum(
                (
                    X[:, None, :]
                    -
                    centroides[
                        None, :, :
                    ]
                ) ** 2,
                axis=2
            )

            nuevas_etiquetas = (
                np.argmin(
                    distancias,
                    axis=1
                )
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

            etiquetas = (
                nuevas_etiquetas
            )

            nuevos_centroides = []

            for cluster in range(
                k
            ):

                puntos = X[
                    etiquetas
                    ==
                    cluster
                ]

                if len(
                    puntos
                ) == 0:

                    nuevo = X[
                        rng.integers(
                            len(X)
                        )
                    ]

                else:

                    nuevo = (
                        puntos.mean(
                            axis=0
                        )
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

        if (
            inercia
            <
            mejor_inercia
        ):

            mejor_inercia = (
                inercia
            )

            mejor_etiquetas = (
                etiquetas.copy()
            )

    return mejor_etiquetas


def silhouette_coseno(
    X,
    etiquetas
):

    similitud = (
        X
        @
        X.T
    )

    distancias = (
        1
        -
        similitud
    )

    distancias = np.clip(
        distancias,
        0,
        2
    )

    clusters = np.unique(
        etiquetas
    )

    valores = []

    for i in range(
        len(X)
    ):

        cluster_actual = (
            etiquetas[i]
        )

        mascara_mismo = (
            etiquetas
            ==
            cluster_actual
        ).copy()

        mascara_mismo[i] = False

        if (
            np.sum(
                mascara_mismo
            )
            ==
            0
        ):
            valores.append(
                0.0
            )

            continue

        a = np.mean(
            distancias[
                i,
                mascara_mismo
            ]
        )

        b = np.inf

        for cluster in clusters:

            if (
                cluster
                ==
                cluster_actual
            ):
                continue

            mascara_otro = (
                etiquetas
                ==
                cluster
            )

            distancia_media = (
                np.mean(
                    distancias[
                        i,
                        mascara_otro
                    ]
                )
            )

            if (
                distancia_media
                <
                b
            ):
                b = (
                    distancia_media
                )

        denominador = max(
            a,
            b
        )

        if denominador == 0:

            s = 0.0

        else:

            s = (
                (b - a)
                /
                denominador
            )

        valores.append(
            s
        )

    return float(
        np.mean(
            valores
        )
    )


# =========================================================
# CARGAR PUBLICACIONES
# =========================================================

print(
    "Cargando publicaciones..."
)

df_publicaciones = pd.read_csv(
    "publicaciones_uam.csv",
    low_memory=False
)

df_publicaciones = (
    df_publicaciones[
        (
            df_publicaciones[
                "Document Type"
            ]
            ==
            "Article"
        )
        &
        (
            df_publicaciones[
                "Language of Original Document"
            ]
            ==
            "English"
        )
    ]
    .copy()
)

df_publicaciones["Year"] = (
    pd.to_numeric(
        df_publicaciones["Year"],
        errors="coerce"
    )
)

resumen_general = []

for ANIO in ANIOS:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"AÑO {ANIO}"
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # KEYWORDS VÁLIDAS DEL AÑO
    # -----------------------------------------------------

    ruta_frecuencias = (
        f"{CARPETA_FRECUENCIAS}/"
        f"frecuencias_{ANIO}.csv"
    )

    df_frecuencias = pd.read_csv(
        ruta_frecuencias
    )

    columna_keyword = (
        df_frecuencias.columns[0]
    )

    columna_frecuencia = (
        df_frecuencias.columns[1]
    )

    keywords_validas = set(
        df_frecuencias.loc[
            df_frecuencias[
                columna_frecuencia
            ]
            >=
            FRECUENCIA_MINIMA,
            columna_keyword
        ].astype(str)
    )

    print(
        f"Keywords válidas: "
        f"{len(keywords_validas)}"
    )


    # -----------------------------------------------------
    # EMBEDDINGS
    # -----------------------------------------------------

    ruta_embeddings = (
        f"{CARPETA_SKIPGRAM}/"
        f"embeddings_{ANIO}.csv"
    )

    ruta_modelo = (
        f"{CARPETA_SKIPGRAM}/"
        f"skipgram_{ANIO}.model"
    )


    if os.path.exists(
        ruta_embeddings
    ):

        print(
            "Embeddings existentes. "
            "Se reutilizan."
        )

        df_embeddings = pd.read_csv(
            ruta_embeddings
        )

    else:

        print(
            "Entrenando Skip-gram..."
        )

        df_anio = (
            df_publicaciones[
                df_publicaciones[
                    "Year"
                ]
                ==
                ANIO
            ]
            .dropna(
                subset=[
                    "Index Keywords"
                ]
            )
            .copy()
        )

        articulos = []

        for texto in df_anio[
            "Index Keywords"
        ]:

            palabras = []

            for palabra in str(
                texto
            ).split(";"):

                limpia = (
                    limpiar_palabra(
                        palabra
                    )
                )

                if (
                    limpia
                    and
                    limpia
                    in
                    keywords_validas
                ):
                    palabras.append(
                        limpia
                    )

            palabras = sorted(
                set(
                    palabras
                )
            )

            if len(
                palabras
            ) >= 2:

                articulos.append(
                    palabras
                )


        if not articulos:

            print(
                "No hay artículos "
                "suficientes."
            )

            continue


        max_keywords = max(
            len(
                articulo
            )
            for articulo
            in articulos
        )

        print(
            f"Artículos útiles: "
            f"{len(articulos)}"
        )

        print(
            f"Ventana: "
            f"{max_keywords}"
        )


        model = Word2Vec(
            sentences=articulos,
            vector_size=VECTOR_SIZE,
            window=max_keywords,
            min_count=1,
            sg=1,
            negative=5,
            epochs=EPOCHS,
            shrink_windows=False,
            seed=SEED,
            workers=1
        )


        model.save(
            ruta_modelo
        )


        vocabulario = (
            model.wv.index_to_key
        )

        vectores = np.array(
            [
                model.wv[
                    palabra
                ]
                for palabra
                in vocabulario
            ]
        )


        df_embeddings = (
            pd.DataFrame(
                vectores,
                columns=[
                    f"v{i + 1}"
                    for i in range(
                        VECTOR_SIZE
                    )
                ]
            )
        )

        df_embeddings.insert(
            0,
            "keyword",
            vocabulario
        )

        df_embeddings.to_csv(
            ruta_embeddings,
            index=False
        )

        print(
            "Embeddings guardados."
        )

    keywords = np.array(
        df_embeddings[
            "keyword"
        ].tolist()
    )

    X = (
        df_embeddings
        .drop(
            columns=[
                "keyword"
            ]
        )
        .values
        .astype(float)
    )

    X_normalizado = (
        normalizar_filas(
            X
        )
    )


    # -----------------------------------------------------
    # MATRIZ DE AFINIDAD
    # -----------------------------------------------------

    print(
        "Construyendo matriz "
        "de afinidad..."
    )

    W = (
        X_normalizado
        @
        X_normalizado.T
    )

    W[
        W < 0
    ] = 0

    W = np.clip(
        W,
        0,
        1
    )

    np.fill_diagonal(
        W,
        0
    )


    # -----------------------------------------------------
    # LAPLACIANO
    # -----------------------------------------------------

    grados = W.sum(
        axis=1
    )

    grados[
        grados == 0
    ] = 1e-12

    inv_sqrt = (
        1.0
        /
        np.sqrt(
            grados
        )
    )

    L = (
        np.eye(
            len(W)
        )
        -
        (
            inv_sqrt[:, None]
            *
            W
            *
            inv_sqrt[None, :]
        )
    )


    # -----------------------------------------------------
    # AUTOVECTORES
    # -----------------------------------------------------

    print(
        "Calculando "
        "autovectores..."
    )

    autovalores, autovectores = (
        eigh(
            L,
            subset_by_index=[
                0,
                K_MAX - 1
            ]
        )
    )


    # -----------------------------------------------------
    # EVALUAR K
    # -----------------------------------------------------

    resultados_k = []
    etiquetas_por_k = {}


    print(
        "Evaluando k=2...20..."
    )


    for k in range(
        K_MIN,
        K_MAX + 1
    ):

        U = autovectores[
            :,
            :k
        ]

        U = normalizar_filas(
            U
        )

        etiquetas = kmeans_numpy(
            U,
            k,
            seed=SEED,
            n_init=N_INIT,
            max_iter=MAX_ITER
        )

        score = silhouette_coseno(
            X_normalizado,
            etiquetas
        )

        etiquetas_por_k[
            k
        ] = etiquetas

        resultados_k.append(
            {
                "k": k,
                "silhouette_score":
                    score
            }
        )

        print(
            f"k={k:2d} -> "
            f"{score:.4f}"
        )


    df_evaluacion = (
        pd.DataFrame(
            resultados_k
        )
        .sort_values(
            "silhouette_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    ruta_evaluacion = (
        f"{CARPETA_SKIPGRAM}/"
        f"evaluacion_clusters_"
        f"{ANIO}.csv"
    )

    df_evaluacion.to_csv(
        ruta_evaluacion,
        index=False
    )


    # -----------------------------------------------------
    # TRES MEJORES CANDIDATOS
    # -----------------------------------------------------

    candidatos = (
        df_evaluacion
        .head(
            NUM_CANDIDATOS
        )[
            "k"
        ]
        .astype(int)
        .tolist()
    )


    print(
        f"Candidatos: "
        f"{candidatos}"
    )


    # -----------------------------------------------------
    # GENERAR CLUSTERS CANDIDATOS
    # -----------------------------------------------------

    for k in candidatos:

        etiquetas = (
            etiquetas_por_k[
                k
            ]
        )

        df_clusters = pd.DataFrame(
            {
                "keyword":
                    keywords,

                "cluster":
                    etiquetas + 1
            }
        )


        ruta_clusters = (
            f"{CARPETA_SKIPGRAM}/"
            f"clusters_{ANIO}_"
            f"k{k}.csv"
        )

        df_clusters.to_csv(
            ruta_clusters,
            index=False
        )


        # ---------------------------------------------
        # Keywords representativas
        # ---------------------------------------------

        resultados_top = []


        for cluster in range(
            k
        ):

            indices = np.where(
                etiquetas
                ==
                cluster
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
                vectores_cluster
                .mean(
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
                len(
                    orden
                )
            )


            for posicion in orden[
                :cantidad_top
            ]:

                resultados_top.append(
                    {
                        "k": k,

                        "cluster":
                            cluster + 1,

                        "keyword":
                            palabras_cluster[
                                posicion
                            ],

                        "similitud_centro":
                            float(
                                similitudes[
                                    posicion
                                ]
                            ),

                        "tamano_cluster":
                            len(
                                indices
                            )
                    }
                )


        df_top = pd.DataFrame(
            resultados_top
        )


        ruta_top = (
            f"{CARPETA_SKIPGRAM}/"
            f"top_keywords_"
            f"{ANIO}_k{k}.csv"
        )

        df_top.to_csv(
            ruta_top,
            index=False
        )


    # -----------------------------------------------------
    # RESUMEN DEL AÑO
    # -----------------------------------------------------

    mejor = (
        df_evaluacion.iloc[0]
    )

    resumen_general.append(
        {
            "anio": ANIO,

            "keywords":
                len(
                    keywords
                ),

            "mejor_k_silhouette":
                int(
                    mejor["k"]
                ),

            "mejor_silhouette":
                float(
                    mejor[
                        "silhouette_score"
                    ]
                ),

            "candidato_1":
                candidatos[0],

            "candidato_2":
                candidatos[1],

            "candidato_3":
                candidatos[2]
        }
    )


df_resumen = pd.DataFrame(
    resumen_general
)

ruta_resumen = (
    f"{CARPETA_SKIPGRAM}/"
    f"resumen_candidatos_"
    f"2015_2024.csv"
)

df_resumen.to_csv(
    ruta_resumen,
    index=False
)


print(
    "\n"
    + "=" * 70
)

print(
    "PIPELINE COMPLETADO"
)

print(
    "=" * 70
)

print(
    df_resumen.to_string(
        index=False
    )
)

print(
    f"\nResumen guardado en:"
    f"\n{ruta_resumen}"
)