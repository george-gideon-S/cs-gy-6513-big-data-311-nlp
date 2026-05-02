"""
geographic + census joins.

the community districts file has 71 polygons. the smart move is to do the
spatial join on the driver in geopandas (cheap, deterministic, easy to
debug) rather than in spark - spark's spatial story requires sedona which
is overkill for this volume of polygons.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def fetch_districts_geojson(out_path: str | Path) -> Path:
    """download the community districts geojson once and cache it on disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    url = "https://data.cityofnewyork.us/resource/jp9i-3b7y.geojson"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def assign_districts(
    df: DataFrame, districts_path: str | Path
) -> DataFrame:
    """
    add a `district` string column to the spark df by spatial-joining
    lat/lng to the districts geojson. uses geopandas on the driver.

    this is okay because we only need to do it once per pipeline run - the
    output gets cached.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    # load polygons in driver memory (small)
    gdf = gpd.read_file(districts_path)

    # bring lat/lng to driver as pandas. for 2m rows this is a few hundred mb
    # which fits comfortably on a colab high-ram machine.
    pdf = df.select("unique_key", "latitude", "longitude").toPandas()
    pdf = pdf.dropna(subset=["latitude", "longitude"])
    pdf["geometry"] = [Point(xy) for xy in zip(pdf["longitude"], pdf["latitude"])]

    points = gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    joined = gpd.sjoin(points, gdf[["boro_cd", "geometry"]], how="left", predicate="within")
    joined = joined[["unique_key", "boro_cd"]].rename(columns={"boro_cd": "district"})

    # send the join result back to spark and merge
    spark = SparkSession.builder.getOrCreate()
    sdf_join = spark.createDataFrame(joined)
    return df.join(sdf_join, on="unique_key", how="left")


def district_fingerprints(df: DataFrame, top_terms: int = 10) -> pd.DataFrame:
    """
    compute the most distinctive terms per district by tf-idf-style score:
    term-frequency in district / term-frequency in corpus.

    returns a pandas df of (district, top_terms, term_scores) for use in
    the dashboard.
    """
    # explode tokens, count per (district, term)
    exploded = df.select("district", F.explode("tokens").alias("term"))
    by_district = exploded.groupBy("district", "term").count().withColumnRenamed("count", "tf_district")

    # corpus-wide token frequencies
    corpus = exploded.groupBy("term").count().withColumnRenamed("count", "tf_corpus")
    corpus_total = corpus.agg(F.sum("tf_corpus")).collect()[0][0]
    corpus = corpus.withColumn("tf_corpus_norm", F.col("tf_corpus") / F.lit(corpus_total))

    joined = by_district.join(corpus, on="term", how="left")
    # district totals for normalization
    district_totals = (
        by_district.groupBy("district").agg(F.sum("tf_district").alias("d_total"))
    )
    joined = joined.join(district_totals, on="district", how="left")
    joined = joined.withColumn(
        "tf_district_norm", F.col("tf_district") / F.col("d_total")
    )
    # the score: how much more common is term in district vs corpus
    joined = joined.withColumn(
        "lift", F.col("tf_district_norm") / F.col("tf_corpus_norm")
    )

    pdf = joined.toPandas()
    out = []
    for district, sub in pdf.groupby("district"):
        # require minimum tf to filter typos that lift very high on tiny counts
        sub = sub[sub["tf_district"] >= 50]
        top = sub.nlargest(top_terms, "lift")
        out.append({
            "district": district,
            "top_terms": list(top["term"]),
            "lifts": list(top["lift"].round(2)),
        })
    return pd.DataFrame(out)
