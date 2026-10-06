import pandas as pd
from pathlib import Path

CARPETA_SKIPGRAM = Path("SkipGram")

ARCHIVO_SELECCION = (
    CARPETA_SKIPGRAM / "seleccion_topicos_2015_2024.csv"
)

ARCHIVO_SALIDA = (
    CARPETA_SKIPGRAM / "topicos_finales_2015_2024.csv"
)


seleccion = pd.read_csv(ARCHIVO_SELECCION)

print("\nSelección de tópicos:")
print(
    seleccion[
        ["year", "selected_k", "selected_silhouette"]
    ].to_string(index=False)
)


resultados = []

for _, fila in seleccion.iterrows():

    year = int(fila["year"])
    k = int(fila["selected_k"])

    archivo_topicos = (
        CARPETA_SKIPGRAM /
        f"top_keywords_{year}_k{k}.csv"
    )

    if not archivo_topicos.exists():
        raise FileNotFoundError(
            f"No se encontró: {archivo_topicos}"
        )

    df = pd.read_csv(archivo_topicos)


    for cluster, grupo in df.groupby("cluster"):

        grupo = grupo.sort_values(
            "similitud_centro",
            ascending=False
        )

        keywords = grupo["keyword"].astype(str).tolist()

        similitudes = (
            grupo["similitud_centro"]
            .astype(float)
            .tolist()
        )

        tamano_cluster = int(
            grupo["tamano_cluster"].iloc[0]
        )

        resultados.append({
            "year": year,
            "selected_k": k,
            "cluster": int(cluster),
            "tamano_cluster": tamano_cluster,
            "keywords_representativas": "; ".join(keywords),
            "similitudes_centro": "; ".join(
                f"{x:.6f}" for x in similitudes
            )
        })

    print(
        f"{year}: k={k} -> "
        f"{df['cluster'].nunique()} tópicos"
    )


topicos_finales = pd.DataFrame(resultados)

topicos_finales = topicos_finales.sort_values(
    ["year", "cluster"]
).reset_index(drop=True)


topicos_finales.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8-sig"
)


print("\n========================================")
print("RESUMEN")
print("========================================")

resumen = (
    topicos_finales
    .groupby("year")
    .size()
    .reset_index(name="numero_topicos")
)

print(resumen.to_string(index=False))

print(
    f"\nTotal de tópicos: "
    f"{len(topicos_finales)}"
)

print(
    f"\nArchivo guardado en:\n"
    f"{ARCHIVO_SALIDA}"
)