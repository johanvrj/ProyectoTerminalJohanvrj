import pandas as pd
import networkx as nx
from itertools import combinations
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re 

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

df_diccionario = pd.read_csv('Coocurrencias/words.csv') 

#palabras prohibidas
df_excluidas = df_diccionario[df_diccionario['action'].astype(str).str.lower().str.strip() == 'x']
palabras_prohibidas = set(df_excluidas['words'].astype(str).str.strip().str.lower())

#sustitución
df_sustitucion = df_diccionario[
    (df_diccionario['action'].astype(str).str.lower().str.strip() != 'x') & 
    (df_diccionario['action'].notna()) &
    (df_diccionario['action'].astype(str).str.strip() != '') &
    (df_diccionario['action'].astype(str).str.lower().str.strip() != 'nan')
]
#palabra_vieja': 'palabra_nueva'
diccionario_sustitucion = dict(zip(
    df_sustitucion['words'].astype(str).str.strip().str.lower(), 
    df_sustitucion['action'].astype(str).str.strip().str.lower()
))

def limpiar_palabra(texto):
    texto = str(texto).lower().strip()
    
    # NUEVO: Eliminar paréntesis y todo lo que esté adentro (ej. "hola (mundo)" -> "hola ")
    texto = re.sub(r'\(.*?\)', '', texto).strip()

    texto = texto.replace('(', '').replace(')', '').strip()
    
    # NUEVO: Sustituir guion medio por guion bajo
    texto = texto.replace('-', '_')
    
    if texto in palabras_prohibidas:
        return ""
        
    # NUEVO: Sustituir la palabra si está en nuestro diccionario de acción
    if texto in diccionario_sustitucion:
        texto = diccionario_sustitucion[texto]
        
    if re.search(r'\d', texto):
        return "" # si tiene un número lo quita
        
    texto = texto.replace("&", "and").replace("<", "").replace(">", "").replace('"', '').replace("'", "")
    
    # Singular y N-gramas
    partes = texto.split(" ")
    partes = [p for p in partes if p.strip() != ""] # Filtramos espacios en blanco extra que dejaron los paréntesis
    partes_singulares = [lemmatizer.lemmatize(p) for p in partes]
    texto_final = "_".join(partes_singulares)
    
    # Verificación final contra la lista negra tras la limpieza
    if texto_final in palabras_prohibidas:
        return ""
        
    return texto_final


# ==========================================
# FASE 1: CARGA Y FILTRADO INICIAL
# ==========================================
print("Cargando el archivo principal de Scopus...")
df = pd.read_csv('Coocurrencias/publicaciones_uam.csv') 

print(f"-> Total de documentos antes del filtro: {len(df)}")
df = df[
    (df['Document Type'].astype(str).str.strip() == 'Article') & 
    (df['Language of Original Document'].astype(str).str.strip() == 'English')
]
print(f"-> Total de documentos después del filtro: {len(df)}")


# ==========================================
# FASE 2: EMPATE CON LA SEGUNDA BASE DE DATOS
# ==========================================
print("\nCargando archivo secundario de métricas...")
archivo_secundario = 'Coocurrencias/Publications_at_Universidad_Aut_noma_Metropolitana_2014_-_2025.csv'

# NOTA: Ajusta el 'skiprows' si es necesario
df_secundario = pd.read_csv(archivo_secundario, skiprows=21) 

llave = 'EID'

if llave in df.columns and llave in df_secundario.columns:
    columnas_comunes = set(df.columns).intersection(set(df_secundario.columns))
    columnas_comunes.remove(llave)
    
    df_secundario = df_secundario.drop(columns=list(columnas_comunes))
    
    df = pd.merge(df, df_secundario, on=llave, how='left')
    print("-> ¡Cruce exitoso! Las bases de datos se han unificado sin columnas repetidas.")
    
    # NUEVO: Reparación de los IDs con decimales (.0)
    columnas_id = [col for col in df.columns if 'id' in col.lower()]
    for col in columnas_id:
        # Lo pasamos a texto, quitamos el '.0' del final usando RegEx, y limpiamos los 'nan'
        df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', '')
        
else:
    print(f"-> [!] ERROR: No se encontró la columna '{llave}' para hacer el cruce. Revisa los archivos.")


# ==========================================
# FASE 3: LIMPIEZA DEL CORPUS Y CREACIÓN DE REDES
# ==========================================
print("\nIniciando procesamiento de palabras clave y redes...")

df = df.dropna(subset=['Index Keywords', 'Year'])
grupos_por_anio = df.groupby('Year')

todas_las_palabras = []

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

df_frecuencias.to_csv('Coocurrencias/frecuencias_palabras_limpio.csv', index=False)

# Exportamos el CSV con las métricas unidas y los IDs corregidos para que lo revises
df.to_csv('Coocurrencias/base_datos_final_unificada.csv', index=False)

print("\n¡Proceso completado exitosamente!")