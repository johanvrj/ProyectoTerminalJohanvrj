import pandas as pd
import networkx as nx
from itertools import combinations
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import matplotlib.pyplot as plt

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()


df_diccionario = pd.read_csv('words_actualizado.csv')

df_excluidas = df_diccionario[
    df_diccionario['action'].astype(str).str.lower().str.strip() == 'x'
]
palabras_prohibidas = set(
    df_excluidas['words'].astype(str).str.strip().str.lower()
)

df_sustitucion = df_diccionario[
    (df_diccionario['action'].astype(str).str.lower().str.strip() != 'x') &
    (df_diccionario['action'].notna()) &
    (df_diccionario['action'].astype(str).str.strip() != '') &
    (df_diccionario['action'].astype(str).str.lower().str.strip() != 'nan')
]

diccionario_sustitucion = dict(zip(
    df_sustitucion['words'].astype(str).str.strip().str.lower(),
    df_sustitucion['action'].astype(str).str.strip().str.lower()
))


def limpiar_palabra(texto):
    texto = str(texto).lower().strip()

    # Eliminar paréntesis y todo lo que esté adentro
    texto = re.sub(r'\(.*?\)', '', texto).strip()

    # Eliminar paréntesis huérfanos
    texto = texto.replace('(', '').replace(')', '').strip()

    # Sustituir guion medio por guion bajo
    texto = texto.replace('-', '_')

    if texto in palabras_prohibidas:
        return ""

    if texto in diccionario_sustitucion:
        texto = diccionario_sustitucion[texto]

    if re.search(r'\d', texto):
        return "" 

    texto = (
        texto.replace("&", "and")
        .replace("<", "")
        .replace(">", "")
        .replace('"', '')
        .replace("'", "")
    )

    partes = texto.split(" ")
    partes = [p for p in partes if p.strip() != ""]
    partes_singulares = [lemmatizer.lemmatize(p) for p in partes]
    texto_final = "_".join(partes_singulares)

    if texto_final in palabras_prohibidas:
        return ""

    return texto_final


print("Cargando el archivo principal de Scopus...")
df = pd.read_csv('publicaciones_uam.csv')

print("\nAños presentes en la base original:")
print(sorted(df['Year'].dropna().unique()))

print("\nCantidad de documentos por año:")
print(df['Year'].value_counts().sort_index())

print(f"-> Total de documentos antes del filtro: {len(df)}")
df = df[
    (df['Document Type'].astype(str).str.strip() == 'Article') &
    (df['Language of Original Document'].astype(str).str.strip() == 'English')
]
print(f"-> Total de documentos después del filtro: {len(df)}")


print("\nCargando archivo secundario de métricas...")
archivo_secundario = 'Publications_at_Universidad_Aut_noma_Metropolitana_2014_-_2025.csv'


df_secundario = pd.read_csv(archivo_secundario, skiprows=21)

llave = 'EID'

if llave in df.columns and llave in df_secundario.columns:
    columnas_comunes = set(df.columns).intersection(set(df_secundario.columns))
    columnas_comunes.remove(llave)

    df_secundario = df_secundario.drop(columns=list(columnas_comunes))

    df = pd.merge(df, df_secundario, on=llave, how='left')
    print(
        "-> ¡Cruce exitoso! Las bases de datos se han unificado "
        "sin columnas repetidas."
    )

    columnas_id = [col for col in df.columns if 'id' in col.lower()]
    for col in columnas_id:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
            .replace('nan', '')
        )
else:
    print(
        f"-> [!] ERROR: No se encontró la columna '{llave}' "
        "para hacer el cruce. Revisa los archivos."
    )


print("\nIniciando limpieza de palabras clave...")

df = df.dropna(subset=['Index Keywords', 'Year'])

keywords_limpias_por_fila = {}

for indice, keywords_str in df['Index Keywords'].items():
    palabras_crudas = str(keywords_str).split(';')

    palabras = []

    for p in palabras_crudas:
        p_limpia = limpiar_palabra(p)

        if p_limpia != "":
            palabras.append(p_limpia)

    palabras_unicas = list(set(palabras))
    keywords_limpias_por_fila[indice] = palabras_unicas

'''
print("\nCalculando frecuencias por año...")

for anio, datos_del_anio in df.groupby('Year'):

    todas_las_palabras_anio = []

    for indice in datos_del_anio.index:
        todas_las_palabras_anio.extend(
            keywords_limpias_por_fila[indice]
        )

    contador_anio = Counter(todas_las_palabras_anio)

    df_frecuencias_anio = pd.DataFrame(
        contador_anio.items(),
        columns=['words', 'count']
    ).sort_values(by='count', ascending=False)

    # Guardar frecuencias del año
    nombre_csv = f"frecuencias_{int(anio)}.csv"

    df_frecuencias_anio.to_csv(
        nombre_csv,
        index=False
    )

    # ------------------------------------------
    # Distribución de frecuencias
    # X = número de apariciones
    # Y = número de palabras con esa frecuencia
    # ------------------------------------------

    distribucion = (
        df_frecuencias_anio['count']
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        distribucion.index,
        distribucion.values
    )

    plt.xlabel("Número de apariciones (count)")
    plt.ylabel("Número de palabras")
    plt.title(
        f"Distribución de frecuencias de palabras - {int(anio)}"
    )

    plt.tight_layout()

    nombre_histograma = f"histograma_{int(anio)}.png"

    plt.savefig(
        nombre_histograma,
        dpi=300
    )

    plt.close()

    print(
        f"-> {int(anio)}: "
        f"{len(contador_anio)} keywords únicas | "
        f"máxima frecuencia = "
        f"{max(contador_anio.values())}"
    )
'''

print("\nCreando redes por año sin outliers...")

FRECUENCIA_MINIMA = 3

print("\nDocumentos disponibles por año:")
print(df['Year'].value_counts().sort_index())

grupos_por_anio = df.groupby('Year')

for anio, datos_del_anio in grupos_por_anio:

    anio = int(anio)

    print(f"\nProcesando red del año: {anio}...")

    # -------------------------------------------------
    # Calcular frecuencias SOLO para este año
    # -------------------------------------------------

    palabras_del_anio = []

    for indice in datos_del_anio.index:
        palabras_del_anio.extend(
            keywords_limpias_por_fila[indice]
        )

    contador_anio = Counter(palabras_del_anio)

    palabras_frecuentes_anio = {
        palabra
        for palabra, frecuencia in contador_anio.items()
        if frecuencia >= FRECUENCIA_MINIMA
    }

    print(
        f" -> Keywords antes del filtro: "
        f"{len(contador_anio)}"
    )

    print(
        f" -> Keywords con frecuencia >= {FRECUENCIA_MINIMA}: "
        f"{len(palabras_frecuentes_anio)}"
    )

    print(
        f" -> Keywords eliminadas como outliers: "
        f"{len(contador_anio) - len(palabras_frecuentes_anio)}"
    )

    # -------------------------------------------------
    # Crear red del año
    # -------------------------------------------------

    G_anio = nx.Graph()

    for indice in datos_del_anio.index:

        palabras = [
            palabra
            for palabra in keywords_limpias_por_fila[indice]
            if palabra in palabras_frecuentes_anio
        ]

        G_anio.add_nodes_from(palabras)

        pares = combinations(palabras, 2)

        for palabra1, palabra2 in pares:

            if G_anio.has_edge(palabra1, palabra2):

                G_anio[palabra1][palabra2]['weight'] += 1

            else:

                G_anio.add_edge(
                    palabra1,
                    palabra2,
                    weight=1
                )

    nombre_archivo = (
        f"../Redes/red_coocurrencia_{anio}.gexf"
    )

    nx.write_gexf(
        G_anio,
        nombre_archivo
    )

    print(
        f" -> Nodos: {G_anio.number_of_nodes()} | "
        f"Enlaces: {G_anio.number_of_edges()}"
    )   

df.to_csv(
    '../base_datos_final_unificada.csv',
    index=False
)


print("\n¡Proceso completado exitosamente!")
