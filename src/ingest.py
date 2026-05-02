"""
soda api ingestion + schema normalization for nyc 311.

the 2020+ dataset (erm2-nwe9) renamed several columns from the historical
2010-2019 set (76ig-c548). this module pulls both, harmonizes column names,
and produces a single clean parquet we can train against.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict
import time
import io

import pandas as pd
import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import SODA_APP_TOKEN


# the column rename map. left = historical (2010-19) name, right = canonical
# name we use throughout the project. the 2020+ dataset already uses some
# canonical-ish names but a few still differ.
# socrata always returns lowercase_with_underscores so we work in that form.
COLUMN_MAP: Dict[str, str] = {
    # 2010-19 historical -> canonical
    "complaint_type": "problem",
    "descriptor": "problem_detail",
    # both datasets already use these but we list them so theyre explicit
    "unique_key": "unique_key",
    "created_date": "created_date",
    "closed_date": "closed_date",
    "agency": "agency",
    "agency_name": "agency_name",
    "incident_zip": "incident_zip",
    "borough": "borough",
    "latitude": "latitude",
    "longitude": "longitude",
    "status": "status",
    "resolution_description": "resolution_description",
    "location_type": "location_type",
}

# the columns we actually care about for modeling. dropping the rest at
# ingest cuts our parquet size by ~70%.
KEEP_COLS: List[str] = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "problem",
    "problem_detail",
    "location_type",
    "incident_zip",
    "borough",
    "latitude",
    "longitude",
    "status",
    "resolution_description",
]


def _soda_headers() -> Dict[str, str]:
    """returns the auth header dict if we have an app token, else empty."""
    if SODA_APP_TOKEN:
        return {"X-App-Token": SODA_APP_TOKEN}
    return {}


def fetch_soda_paginated(
    endpoint: str,
    target_rows: int,
    page_size: int = 50_000,
    where: Optional[str] = None,
    select: Optional[str] = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    pull rows from a socrata endpoint with offset-based pagination.

    soda caps single requests at 50k rows and starts throttling around
    1k requests/hour for anonymous calls. with an app token were good
    for higher volume. for 2m rows that means ~40 requests, ~30 minutes.

    Args:
        endpoint: full soda csv endpoint url, e.g. SODA_2020_PLUS.
        target_rows: how many rows we want total.
        page_size: rows per request. socrata caps at 50k.
        where: optional soql where clause, e.g. "created_date > '2023-01-01'".
        select: optional column selection.
        timeout: seconds before requests gives up.

    Returns:
        a pandas dataframe with target_rows (or fewer if dataset has less).
    """
    frames: List[pd.DataFrame] = []
    fetched = 0
    offset = 0
    headers = _soda_headers()

    while fetched < target_rows:
        # cap the last batch so we dont over-pull
        this_batch = min(page_size, target_rows - fetched)

        params = {
            "$limit": this_batch,
            "$offset": offset,
            "$order": "unique_key",  # stable ordering so pages dont overlap
        }
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select

        # one retry on transient failures - soda flakes occasionally
        for attempt in range(3):
            try:
                resp = requests.get(
                    endpoint, params=params, headers=headers, timeout=timeout
                )
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt  # 1s then 2s before giving up
                print(f"  retry {attempt+1} after {wait}s: {e}")
                time.sleep(wait)

        # parse csv from response
        chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        if len(chunk) == 0:
            print(f"  hit end of dataset at offset {offset}")
            break

        frames.append(chunk)
        fetched += len(chunk)
        offset += len(chunk)
        print(f"  fetched {fetched:,} / {target_rows:,}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_columns(df: DataFrame) -> DataFrame:
    """
    apply COLUMN_MAP renames and trim to KEEP_COLS. works on a spark df.
    silently skips columns that arent present in the input.
    """
    # build the rename projection - only rename cols that actually exist
    existing = set(df.columns)
    renamed = df
    for old, new in COLUMN_MAP.items():
        if old in existing and old != new:
            renamed = renamed.withColumnRenamed(old, new)

    # add nulls for any KEEP_COLS that are missing so the schema is uniform
    final_existing = set(renamed.columns)
    for col in KEEP_COLS:
        if col not in final_existing:
            renamed = renamed.withColumn(col, F.lit(None))

    # cast types to be consistent across the union
    return (
        renamed.select(*KEEP_COLS)
        .withColumn("created_date", F.to_timestamp("created_date"))
        .withColumn("closed_date", F.to_timestamp("closed_date"))
        .withColumn("latitude", F.col("latitude").cast("double"))
        .withColumn("longitude", F.col("longitude").cast("double"))
    )


def fetch_311_to_parquet(
    spark: SparkSession,
    endpoint: str,
    target_rows: int,
    out_path: str,
    where: Optional[str] = None,
) -> int:
    """
    convenience wrapper - pulls from soda, normalizes, writes parquet.

    Args:
        spark: the spark session.
        endpoint: SODA_2020_PLUS or SODA_2010_2019.
        target_rows: how many to pull.
        out_path: where to write the parquet (drive path or local).
        where: optional soql filter.

    Returns:
        the number of rows actually written.
    """
    print(f"pulling {target_rows:,} rows from {endpoint}")
    pdf = fetch_soda_paginated(endpoint, target_rows, where=where)
    print(f"got {len(pdf):,} rows. converting to spark...")

    # spark cant infer datetime cleanly from csv, so we let normalize_columns
    # do the to_timestamp cast
    sdf = spark.createDataFrame(pdf)
    sdf = normalize_columns(sdf)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sdf.write.mode("overwrite").parquet(out_path)
    n = sdf.count()
    print(f"wrote {n:,} rows to {out_path}")
    return n


def stratified_sample(df: DataFrame, target_size: int, label_col: str = "problem") -> DataFrame:
    """
    proportional stratified sample using sampleBy. preserves class distribution
    so when we split for training later, every class is represented.

    randomSplit doesnt stratify so we cant use it here.
    """
    # compute per-class fractions that hit target_size in expectation
    counts = df.groupBy(label_col).count().collect()
    total = sum(r["count"] for r in counts)
    fraction = target_size / total

    fractions = {r[label_col]: min(1.0, fraction) for r in counts if r[label_col] is not None}
    return df.sampleBy(label_col, fractions=fractions, seed=42)
