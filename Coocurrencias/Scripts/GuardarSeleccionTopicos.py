import pandas as pd
from pathlib import Path


# ============================================================
# Selección final de número de tópicos por año
# ============================================================

seleccion = [
    {
        "year": 2015,
        "best_k_silhouette": 16,
        "selected_k": 14,
        "justification": "Silhouette similar al máximo, con menor fragmentación y mejor coherencia temática."
    },
    {
        "year": 2016,
        "best_k_silhouette": 19,
        "selected_k": 18,
        "justification": "Mejor equilibrio entre separación de tópicos y coherencia temática."
    },
    {
        "year": 2017,
        "best_k_silhouette": 20,
        "selected_k": 15,
        "justification": "k=20 fragmenta varios tópicos; k=15 produce grupos más compactos e interpretables."
    },
    {
        "year": 2018,
        "best_k_silhouette": 18,
        "selected_k": 18,
        "justification": "Coinciden el máximo Silhouette y una estructura temática coherente."
    },
    {
        "year": 2019,
        "best_k_silhouette": 2,
        "selected_k": 18,
        "justification": "k=2 representa macroáreas; k=18 permite identificar tópicos científicos específicos."
    },
    {
        "year": 2020,
        "best_k_silhouette": 2,
        "selected_k": 20,
        "justification": "k=2 y k=3 representan macroáreas; k=20 proporciona mayor resolución temática."
    },
    {
        "year": 2021,
        "best_k_silhouette": 2,
        "selected_k": 19,
        "justification": "k=19 mejora la separación de áreas temáticas y reduce mezclas observadas con k=18."
    },
    {
        "year": 2022,
        "best_k_silhouette": 2,
        "selected_k": 20,
        "justification": "k=20 presenta grupos más especializados y reduce mezclas temáticas respecto a k=19."
    },
    {
        "year": 2023,
        "best_k_silhouette": 19,
        "selected_k": 19,
        "justification": "Coinciden el máximo Silhouette y una estructura temática coherente."
    },
    {
        "year": 2024,
        "best_k_silhouette": 20,
        "selected_k": 19,
        "justification": "k=19 ofrece mejor equilibrio temático; k=20 introduce fragmentación adicional."
    },
]


# ============================================================
# Recuperar los valores de Silhouette calculados por el pipeline
# ============================================================

filas = []

for item in seleccion:
    year = item["year"]

    ruta_evaluacion = Path(
        f"SkipGram/evaluacion_clusters_{year}.csv"
    )

    evaluacion = pd.read_csv(ruta_evaluacion)

    best_k = item["best_k_silhouette"]
    selected_k = item["selected_k"]

    best_score = evaluacion.loc[
        evaluacion["k"] == best_k, "silhouette_score"
    ].iloc[0]

    selected_score = evaluacion.loc[
        evaluacion["k"] == selected_k, "silhouette_score"
    ].iloc[0]

    filas.append({
        "year": year,
        "best_k_silhouette": best_k,
        "best_silhouette": best_score,
        "selected_k": selected_k,
        "selected_silhouette": selected_score,
        "justification": item["justification"]
    })


# ============================================================
# Guardar resultado
# ============================================================

resultado = pd.DataFrame(filas)

salida = Path("SkipGram/seleccion_topicos_2015_2024.csv")
resultado.to_csv(salida, index=False, encoding="utf-8-sig")

print("\nSelección final de tópicos:")
print(resultado.to_string(index=False))

print(f"\nArchivo guardado en: {salida}")