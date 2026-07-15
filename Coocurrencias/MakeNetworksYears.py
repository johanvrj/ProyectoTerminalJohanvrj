import pandas as pd
import networkx as nx
from itertools import combinations
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re #Librería para regex

# Descargar diccionarios de NLTK 
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

#Lista negra
df_diccionario = pd.read_csv('words.csv') 
df_excluidas = df_diccionario[df_diccionario['action'].astype(str).str.lower().str.strip() == 'x']
palabras_prohibidas = set(df_excluidas['words'].astype(str).str.strip().str.lower())

#Diccionario de Mapeo Semántico
try:
    df_mapeo = pd.read_csv('diccionario_mapeo_keywords.csv')
    diccionario_mapeo = dict(zip(df_mapeo['Original'].str.lower().str.strip(), df_mapeo['Concepto'].str.lower().str.strip()))
except FileNotFoundError:
    diccionario_mapeo = {}
    print("Aviso: No se encontró 'diccionario_mapeo_keywords.csv'. Se omitirá la homologación.")


def limpiar_palabra(texto):
    texto = str(texto).strip().lower()
    
    if texto in palabras_prohibidas:
        return ""
        
    if re.search(r'\d', texto):
        return "" #si tiene un número lo quita
        
    texto = texto.replace("&", "and").replace("<", "").replace(">", "").replace('"', '').replace("'", "")
    
    # Singular y N-gramas
    partes = texto.split(" ")
    partes_singulares = [lemmatizer.lemmatize(p) for p in partes]
    texto_final = "_".join(partes_singulares)
    
    #campo semantico
    if texto_final in diccionario_mapeo:
        texto_final = diccionario_mapeo[texto_final]
    
    if texto_final in palabras_prohibidas:
        return ""
        
    return texto_final

print("Cargando el archivo de Scopus...")
df = pd.read_csv('publicaciones_uam.csv') 

# Tirar filas vacías
df = df.dropna(subset=['Index Keywords', 'Year'])
grupos_por_anio = df.groupby('Year')

todas_las_palabras = []

for anio, datos_del_anio in grupos_por_anio:
    print(f"\nRed del año: {int(anio)}...")
    G_anio = nx.Graph()
    
    for keywords_str in datos_del_anio['Index Keywords']:
        palabras_crudas = keywords_str.split(';')
        
        palabras = []
        for p in palabras_crudas:
            p_limpia = limpiar_palabra(p)
            if p_limpia != "": 
                palabras.append(p_limpia)
                todas_las_palabras.append(p_limpia) 
                
        #quitar duplicados
        palabras_unicas = list(set(palabras))
        pares = list(combinations(palabras_unicas, 2))
        
        for palabra1, palabra2 in pares:
            if G_anio.has_edge(palabra1, palabra2):
                G_anio[palabra1][palabra2]['weight'] += 1
            else:
                G_anio.add_edge(palabra1, palabra2, weight=1)
    
    nombre_archivo = f"red_coocurrencia_{int(anio)}.gexf"
    nx.write_gexf(G_anio, nombre_archivo)
    
    print(f" Nodos: {G_anio.number_of_nodes()} | Enlaces: {G_anio.number_of_edges()}")


print("\nGenerando CSV de frecuencias general...")
contador = Counter(todas_las_palabras)
df_frecuencias = pd.DataFrame(contador.items(), columns=['words', 'count'])
df_frecuencias = df_frecuencias.sort_values(by='count', ascending=False)

df_frecuencias.to_csv('frecuencias_palabras_limpio.csv', index=False)

print("Proceso completado")