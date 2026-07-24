import pandas as pd
import networkx as nx
from itertools import combinations
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re 

# Descargar diccionarios de NLTK 
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()


df_diccionario = pd.read_csv('Coocurrencias/words.csv') 
df_excluidas = df_diccionario[df_diccionario['action'].astype(str).str.lower().str.strip() == 'x']
palabras_prohibidas = set(df_excluidas['words'].astype(str).str.strip().str.lower())

def limpiar_palabra(texto):
    texto = str(texto).strip().lower()
    
    if texto in palabras_prohibidas:
        return ""
        
    if re.search(r'\d', texto):
        return "" # si tiene un numero lo quita
        
    texto = texto.replace("&", "and").replace("<", "").replace(">", "").replace('"', '').replace("'", "")
    
    # en singular y N-gramas
    partes = texto.split(" ")
    partes_singulares = [lemmatizer.lemmatize(p) for p in partes]
    texto_final = "_".join(partes_singulares)
    
    # verificacion final contra la lista negra tras la limpieza
    if texto_final in palabras_prohibidas:
        return ""
        
    return texto_final


#carga de archivos y filtrado de datos
print("Cargando el archivo principal de Scopus...")
df = pd.read_csv('Coocurrencias/publicaciones_uam.csv') 

print(f"-> Total de documentos antes del filtro: {len(df)}")
df = df[
    (df['Document Type'].astype(str).str.strip() == 'Article') & 
    (df['Language of Original Document'].astype(str).str.strip() == 'English')
]
print(f"-> Total de documentos después del filtro: {len(df)}")


#Juntar con segundo archivo
print("\nCargando archivo secundario de métricas...")
archivo_secundario = 'Coocurrencias/Publications_at_Universidad_Aut_noma_Metropolitana_2014_-_2025.csv'

df_secundario = pd.read_csv(archivo_secundario, skiprows=21) 

llave = 'EID'

if llave in df.columns and llave in df_secundario.columns:
    columnas_comunes = set(df.columns).intersection(set(df_secundario.columns))
    columnas_comunes.remove(llave)
    
    df_secundario = df_secundario.drop(columns=list(columnas_comunes))
    
    df = pd.merge(df, df_secundario, on=llave, how='left')
    print("Las bases de datos se juntaron sin repetidos.")
else:
    print(f"ERROR: No se encontro la columna '{llave}' para el cruce.")


#Limpieza
print("\nIniciando procesamiento de palabras clave y redes...")

df = df.dropna(subset=['Index Keywords', 'Year'])
grupos_por_anio = df.groupby('Year')

todas_las_palabras = []

#creacion de redes por año
for anio, datos_del_anio in grupos_por_anio:
    print(f"Procesando red del año: {int(anio)}...")
    G_anio = nx.Graph()
    
    for keywords_str in datos_del_anio['Index Keywords']:
        palabras_crudas = keywords_str.split(';')
        
        palabras = []
        for p in palabras_crudas:
            p_limpia = limpiar_palabra(p)
            if p_limpia != "": 
                palabras.append(p_limpia)
                todas_las_palabras.append(p_limpia) 
                
        palabras_unicas = list(set(palabras))
        pares = list(combinations(palabras_unicas, 2))
        
        for palabra1, palabra2 in pares:
            if G_anio.has_edge(palabra1, palabra2):
                G_anio[palabra1][palabra2]['weight'] += 1
            else:
                G_anio.add_edge(palabra1, palabra2, weight=1)
    
    nombre_archivo = f"red_coocurrencia_{int(anio)}.gexf"
    nx.write_gexf(G_anio, nombre_archivo)
    
    print(f" -> Nodos: {G_anio.number_of_nodes()} | Enlaces: {G_anio.number_of_edges()}")

print("\nGenerando CSV de frecuencias general...")
contador = Counter(todas_las_palabras)
df_frecuencias = pd.DataFrame(contador.items(), columns=['words', 'count'])
df_frecuencias = df_frecuencias.sort_values(by='count', ascending=False)

df_frecuencias.to_csv('frecuencias_palabras_limpio.csv', index=False)

print("\n¡Proceso completado exitosamente!")