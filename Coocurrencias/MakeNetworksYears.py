import pandas as pd
import networkx as nx
from itertools import combinations

def limpiar_palabra(texto):
    texto = str(texto).strip()
    texto = texto.replace("&", "and").replace("<", "").replace(">", "").replace('"', '').replace("'", "")
    return texto

print("Cargando el archivo...")
df = pd.read_csv('publicaciones_uam.csv') 


#tirar filas vacías
df = df.dropna(subset=['Index Keywords', 'Year'])
grupos_por_anio = df.groupby('Year')

for anio, datos_del_anio in grupos_por_anio:
    print(f"\nRed del año: {int(anio)}...")
    G_anio = nx.Graph()
    
    for keywords_str in datos_del_anio['Index Keywords']:
        palabras_crudas = keywords_str.split(';')
        
        palabras = []
        for p in palabras_crudas:
            p_limpia = limpiar_palabra(p)
            if p_limpia != "": #solo agregar si la palabra no es null
                palabras.append(p_limpia)
                
        #quitar duplicados
        palabras_unicas = list(set(palabras))
        pares = list(combinations(palabras_unicas, 2))
        
        #agregar enlaces a la red
        for palabra1, palabra2 in pares:
            if G_anio.has_edge(palabra1, palabra2):
                G_anio[palabra1][palabra2]['weight'] += 1
            else:
                G_anio.add_edge(palabra1, palabra2, weight=1)
    
    
    nombre_archivo = f"red_coocurrencia_{int(anio)}.gexf"
    nx.write_gexf(G_anio, nombre_archivo)
    
    print(f" Nodos: {G_anio.number_of_nodes()} | Enlaces: {G_anio.number_of_edges()}")

print("\nProceso completado")