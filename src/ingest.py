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

        # parse csv from response. dtype=str keeps every column as string so
        # arrow doesnt try (and fail) to type-infer columns like community_board
        # which look numeric except for '08'-style codes with leading zeros.
        # spark normalize_columns casts datetime/double explicitly later.
        chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False, dtype=str)
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


def _row_count_parquet(spark: SparkSession, path: str) -> int:
    """count rows in a parquet path. returns 0 if path missing or empty."""
    try:
        return spark.read.parquet(path).count()
    except Exception:
        return 0


def fetch_311_to_parquet(
    spark: SparkSession,
    endpoint: str,
    target_rows: int,
    out_path: str,
    where: Optional[str] = None,
) -> int:
    """
    pull from soda, normalize, write parquet. crash-safe and idempotent.

    behavior:
      1. if `out_path` already has at least `target_rows` rows, skip everything.
         re-running the cell after a successful pull is a no-op.
      2. otherwise, look for a `.raw.parquet` checkpoint next to out_path. if
         it exists with enough rows, skip the SODA pull and jump straight to
         the spark normalize step. so a JVM crash during normalize doesnt
         force a re-pull.
      3. if no checkpoint, pull from SODA into pandas, write the raw pandas
         straight to parquet via pyarrow (fast, memory-safe, no JVM involved).
      4. spark reads that parquet (columnar, efficient), runs normalize, writes
         the final parquet at out_path.

    NOTE: previously this function called spark.createDataFrame(pdf) on the
    pandas dataframe directly. on 5m+ rows in arrow-fallback mode that ships
    rows one at a time through py4j and OOM-killed the 8 GB JVM. the new path
    avoids createDataFrame entirely.

    Args:
        spark: the spark session.
        endpoint: SODA_2020_PLUS or SODA_2010_2019.
        target_rows: how many to pull.
        out_path: where to write the final normalized parquet.
        where: optional soql filter.

    Returns:
        the number of rows actually written.
    """
    out = Path(out_path)
    raw = out.with_name(out.name.replace(".parquet", ".raw.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)

    # step 1: already done? skip.
    if out.exists():
        existing = _row_count_parquet(spark, out_path)
        if existing >= target_rows:
            print(f"already have {existing:,} rows at {out_path}, skipping pull")
            return existing
        print(f"out_path has only {existing:,} rows, less than target {target_rows:,}; redoing")

    # step 2: have raw checkpoint? skip the SODA pull.
    have_checkpoint = False
    if raw.exists():
        raw_count = _row_count_parquet(spark, str(raw))
        if raw_count >= target_rows:
            print(f"raw checkpoint has {raw_count:,} rows at {raw}; skipping SODA pull")
            have_checkpoint = True
        else:
            print(f"raw checkpoint exists but has only {raw_count:,} rows; will re-pull")

    # step 3: pull from SODA + write raw checkpoint via pyarrow (no JVM)
    if not have_checkpoint:
        print(f"pulling {target_rows:,} rows from {endpoint}")
        pdf = fetch_soda_paginated(endpoint, target_rows, where=where)
        print(f"got {len(pdf):,} rows. checkpointing to {raw} (no spark conversion)")
        # pandas to_parquet uses pyarrow under the hood. memory-stable and
        # crash-safe - if the next step blows up, this checkpoint persists.
        pdf.to_parquet(str(raw), index=False, engine="pyarrow")
        del pdf  # free pandas memory before spark loads the parquet
        print(f"raw checkpoint written")

    # step 4: spark reads the raw parquet (columnar, no py4j row-by-row),
    # normalizes columns, writes the final parquet
    print(f"loading raw parquet into spark for normalization...")
    sdf = spark.read.parquet(str(raw))
    sdf = normalize_columns(sdf)
    sdf.write.mode("overwrite").parquet(out_path)
    n = sdf.count()
    print(f"wrote {n:,} normalized rows to {out_path}")
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
