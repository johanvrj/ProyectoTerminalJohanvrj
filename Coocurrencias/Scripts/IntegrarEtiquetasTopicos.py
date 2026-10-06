import pandas as pd
from pathlib import Path


CARPETA = Path("SkipGram")

ARCHIVO_TOPICOS = CARPETA / "topicos_finales_2015_2024.csv"
ARCHIVO_ETIQUETAS = CARPETA / "etiquetas_topicos.csv"

ARCHIVO_SALIDA = (
    CARPETA / "topicos_finales_etiquetados_2015_2024.csv"
)


topicos = pd.read_csv(ARCHIVO_TOPICOS)
etiquetas = pd.read_csv(ARCHIVO_ETIQUETAS)


# ============================================================
# Integrar etiquetas usando año + cluster
# ============================================================

resultado = topicos.merge(
    etiquetas,
    on=["year", "cluster"],
    how="left",
    validate="one_to_one"
)

columnas_principales = [
    "topic_id",
    "year",
    "cluster",
    "selected_k",
    "tamano_cluster",
    "topic_label",
    "topic_description",
    "interpretation_confidence",
    "keywords_representativas",
    "similitudes_centro"
]

resultado = resultado[columnas_principales]

resultado.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8-sig"
)

etiquetados = resultado["topic_label"].notna().sum()
sin_etiquetar = resultado["topic_label"].isna().sum()

print("\n========================================")
print("INTEGRACIÓN DE ETIQUETAS")
print("========================================")

print(f"Total de tópicos: {len(resultado)}")
print(f"Tópicos etiquetados: {etiquetados}")
print(f"Tópicos pendientes: {sin_etiquetar}")

print("\nTópicos etiquetados por año:")

print(
    resultado[
        resultado["topic_label"].notna()
    ]
    .groupby("year")
    .size()
    .to_string()
)

print("\nEtiquetas de 2015:")

print(
    resultado.loc[
        resultado["year"] == 2015,
        [
            "topic_id",
            "cluster",
            "topic_label",
            "interpretation_confidence"
        ]
    ].to_string(index=False)
)

print(f"\nArchivo guardado en:\n{ARCHIVO_SALIDA}")