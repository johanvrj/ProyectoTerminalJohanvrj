import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import re
import os

# -------------------------------------------------
# Configuración
# -------------------------------------------------

ANIO = 2016
FRECUENCIA_MINIMA = 3

VECTOR_SIZE = 100
EPOCHS = 100
SEED = 42

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()


# -------------------------------------------------
# Cargar diccionario de limpieza
# -------------------------------------------------

df_diccionario = pd.read_csv("words_actualizado.csv")

df_excluidas = df_diccionario[
    df_diccionario["action"]
    .astype(str)
    .str.lower()
    .str.strip() == "x"
]

palabras_prohibidas = set(
    df_excluidas["words"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df_sustitucion = df_diccionario[
    (
        df_diccionario["action"]
        .astype(str)
        .str.lower()
        .str.strip() != "x"
    )
    & (df_diccionario["action"].notna())
    & (
        df_diccionario["action"]
        .astype(str)
        .str.strip() != ""
    )
    & (
        df_diccionario["action"]
        .astype(str)
        .str.lower()
        .str.strip() != "nan"
    )
]

diccionario_sustitucion = dict(
    zip(
        df_sustitucion["words"]
        .astype(str)
        .str.strip()
        .str.lower(),

        df_sustitucion["action"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
)


def limpiar_palabra(texto):

    texto = str(texto).lower().strip()

    # Eliminar paréntesis y contenido
    texto = re.sub(r"\(.*?\)", "", texto).strip()

    # Eliminar paréntesis huérfanos
    texto = (
        texto
        .replace("(", "")
        .replace(")", "")
        .strip()
    )

    texto = texto.replace("-", "_")

    if texto in palabras_prohibidas:
        return ""

    if texto in diccionario_sustitucion:
        texto = diccionario_sustitucion[texto]

    if re.search(r"\d", texto):
        return ""

    texto = (
        texto
        .replace("&", "and")
        .replace("<", "")
        .replace(">", "")
        .replace('"', "")
        .replace("'", "")
    )

    partes = texto.split(" ")

    partes = [
        p for p in partes
        if p.strip() != ""
    ]

    partes_singulares = [
        lemmatizer.lemmatize(p)
        for p in partes
    ]

    texto_final = "_".join(partes_singulares)

    if texto_final in palabras_prohibidas:
        return ""

    return texto_final


print(f"Cargando keywords válidas de {ANIO}...")

df_frecuencias = pd.read_csv(
    f"FrecuenciaEHistograma/frecuencias_{ANIO}.csv"
)

keywords_validas = set(
    df_frecuencias[
        df_frecuencias["count"] >= FRECUENCIA_MINIMA
    ]["words"]
)

print(
    f"-> Keywords con frecuencia >= {FRECUENCIA_MINIMA}: "
    f"{len(keywords_validas)}"
)

print(f"\nCargando publicaciones de {ANIO}...")

df = pd.read_csv("publicaciones_uam.csv")

df = df[
    (df["Document Type"].astype(str).str.strip() == "Article")
    &
    (
        df["Language of Original Document"]
        .astype(str)
        .str.strip() == "English"
    )
    &
    (df["Year"] == ANIO)
]

df = df.dropna(
    subset=["Index Keywords"]
)

print(
    f"-> Publicaciones disponibles: {len(df)}"
)


# -------------------------------------------------
# Construir contextos por artículo
# -------------------------------------------------

articulos = []

for keywords_str in df["Index Keywords"]:

    palabras_crudas = str(
        keywords_str
    ).split(";")

    palabras = []

    for palabra in palabras_crudas:

        palabra_limpia = limpiar_palabra(
            palabra
        )

        if (
            palabra_limpia != ""
            and palabra_limpia in keywords_validas
        ):
            palabras.append(
                palabra_limpia
            )

    palabras = sorted(set(palabras))

    if len(palabras) >= 2:
        articulos.append(palabras)


print(
    f"-> Artículos con al menos 2 keywords válidas: "
    f"{len(articulos)}"
)


# -------------------------------------------------
# Determinar tamaño de ventana
# -------------------------------------------------

max_keywords = max(
    len(articulo)
    for articulo in articulos
)

print(
    f"-> Máximo de keywords válidas en un artículo: "
    f"{max_keywords}"
)

WINDOW = max_keywords


# -------------------------------------------------
# Entrenar Skip-gram
# -------------------------------------------------

print("\nEntrenando modelo Skip-gram...")

model = Word2Vec(
    sentences=articulos,

    # Dimensión de cada embedding
    vector_size=VECTOR_SIZE,

    # Toda la publicación entra en el contexto
    window=WINDOW,

    # Las keywords ya fueron filtradas por frecuencia anual
    min_count=1,

    # sg=1 -> Skip-gram
    sg=1,

    # Negative Sampling
    negative=5,

    # Número de épocas
    epochs=EPOCHS,

    # Evita reducir aleatoriamente la ventana
    shrink_windows=False,

    # Reproducibilidad
    seed=SEED,

    # Un solo worker para favorecer reproducibilidad
    workers=1
)


# -------------------------------------------------
# Información del modelo
# -------------------------------------------------

print("\nModelo entrenado.")

print(
    f"-> Keywords en el vocabulario: "
    f"{len(model.wv)}"
)

print(
    f"-> Dimensión de los embeddings: "
    f"{model.vector_size}"
)

print(
    f"-> Tamaño de ventana: "
    f"{WINDOW}"
)


print("\nEjemplos de vecinos aprendidos:")

ejemplos = [
    "controller",
    "design",
    "physiology"
]

for palabra in ejemplos:

    if palabra in model.wv:

        print(f"\n{palabra}:")

        similares = model.wv.most_similar(
            palabra,
            topn=10
        )

        for similar, similitud in similares:

            print(
                f"  {similar:<35} "
                f"{similitud:.4f}"
            )

os.makedirs(
    "SkipGram",
    exist_ok=True
)

ruta_modelo = (
    f"SkipGram/"
    f"skipgram_{ANIO}.model"
)

model.save(
    ruta_modelo
)

print(
    f"\nModelo guardado en: "
    f"{ruta_modelo}"
)

filas = []

for palabra in model.wv.index_to_key:

    vector = model.wv[palabra]

    fila = {
        "keyword": palabra
    }

    for i, valor in enumerate(vector):

        fila[f"v{i + 1}"] = valor

    filas.append(fila)


df_embeddings = pd.DataFrame(
    filas
)

ruta_embeddings = (
    f"SkipGram/"
    f"embeddings_{ANIO}.csv"
)

df_embeddings.to_csv(
    ruta_embeddings,
    index=False
)

print(
    f"Embeddings guardados en: "
    f"{ruta_embeddings}"
)

print(
    "\nProceso completado."
)