"""
spark session bootstrap. one place to configure things so notebooks dont
each have to remember the delta lake config string.

also handles the classic pyspark gotcha where the driver can `import src.*`
but worker subprocesses cant - we ship a zip of src/ to workers via addPyFile
so udfs that reference our modules dont blow up at unpickle time.
"""
from __future__ import annotations
from pathlib import Path
import os
import tempfile
import zipfile

from pyspark.sql import SparkSession


def _zip_src_for_workers(project_root: Path) -> Path | None:
    """
    walk project_root/src/ and bundle every .py into a zip whose internal
    layout looks like:
        src/__init__.py
        src/config.py
        src/preprocess.py
        ...
    so that workers can import src.foo after addPyFile.

    returns the zip path, or None if src/ doesnt exist.
    """
    src_dir = project_root / "src"
    if not src_dir.is_dir():
        return None

    zip_path = Path(tempfile.gettempdir()) / "src_for_spark.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for py in src_dir.rglob("*.py"):
            arcname = py.relative_to(project_root).as_posix()
            zf.write(py, arcname)
    return zip_path


def _detect_project_root() -> Path:
    """
    find the project root - the directory that contains src/.
    on colab this is /content/project. when running tests locally its
    usually the parent of this file.
    """
    # this file is at <root>/src/spark_setup.py
    here = Path(__file__).resolve()
    candidate = here.parent.parent
    if (candidate / "src").is_dir():
        return candidate
    # colab fallback
    colab_root = Path("/content/project")
    if (colab_root / "src").is_dir():
        return colab_root
    return candidate  # last resort


def _kill_stale_session() -> None:
    """
    if a previous spark session crashed (jvm died, py4j socket dropped),
    SparkSession.getOrCreate() will try to reuse the dead handle and fail
    with `ConnectionRefusedError: [Errno 111] Connection refused`.

    we defend against this by explicitly clearing the singleton handles
    before creating a new session. swallow exceptions because the dead
    session will throw on .stop() too - we just need the references gone.
    """
    try:
        existing = SparkSession._instantiatedSession  # type: ignore[attr-defined]
    except Exception:
        existing = None
    if existing is not None:
        try:
            existing.stop()
            print("stopped stale spark session before creating new one")
        except Exception:
            pass
        try:
            SparkSession._instantiatedSession = None  # type: ignore[attr-defined]
        except Exception:
            pass

    # also clear the SparkContext singleton if it lingers
    try:
        from pyspark import SparkContext
        if SparkContext._active_spark_context is not None:  # type: ignore[attr-defined]
            try:
                SparkContext._active_spark_context.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            SparkContext._active_spark_context = None  # type: ignore[attr-defined]
    except Exception:
        pass


def _verify_java(java_home: str) -> None:
    """
    sanity check: java binary must exist before we ask spark to launch jvm.
    raises a clear error instead of letting py4j throw a cryptic
    ConnectionRefusedError later when the jvm fails to start.
    """
    java_bin = Path(java_home) / "bin" / "java"
    if not java_bin.exists():
        raise RuntimeError(
            f"java not found at {java_bin}. "
            f"on colab, the bootstrap cell should run "
            f"`apt-get install -y openjdk-11-jre-headless` and set "
            f"JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64."
        )


def get_spark(app_name: str = "cs6513-311-nlp", local_cores: str = "*") -> SparkSession:
    """
    create or return the active spark session, configured for delta lake
    plus the addPyFile dance that makes udfs work.

    Args:
        app_name: shows up in the spark ui, useful when youve got multiple
                  notebooks running.
        local_cores: "*" uses all cores. set to "4" or "8" if you want to
                     leave headroom for other stuff.

    Returns:
        a configured SparkSession.
    """
    # belt-and-suspenders: kill any lingering spark state from a previous
    # crashed run before we build a fresh session
    _kill_stale_session()

    project_root = _detect_project_root()

    # verify java is actually present before py4j bothers trying to talk to it
    java_home = os.environ.get("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")
    _verify_java(java_home)

    # belt-and-suspenders: also set PYTHONPATH so any *new* worker process
    # picks up the src/ path even before addPyFile runs.
    existing_pp = os.environ.get("PYTHONPATH", "")
    if str(project_root) not in existing_pp:
        os.environ["PYTHONPATH"] = (
            f"{project_root}{os.pathsep}{existing_pp}" if existing_pp else str(project_root)
        )

    # delta lake plugin coords. these stay in step with delta-spark in
    # requirements.txt. if you bump delta-spark, bump this too.
    delta_pkg = "io.delta:delta-spark_2.12:3.0.0"

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{local_cores}]")
        # delta lake hookup
        .config("spark.jars.packages", delta_pkg)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # bumping driver memory because 2m rows of text descriptions blows
        # out the default 1g pretty fast during tf-idf
        .config("spark.driver.memory", "8g")
        # arrow makes pandas <-> spark conversions sane
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # quieter logs - the default INFO is just noise during demos
        .config("spark.ui.showConsoleProgress", "false")
        # make sure executors inherit our PYTHONPATH so they can find src/
        .config("spark.executorEnv.PYTHONPATH", str(project_root))
        # parquet timestamp rebasing for ancient dates. some 311 rows have
        # closed_date/created_date before 1900-01-01 (data quality artifacts).
        # spark 3.0+ refuses to write these to parquet INT96 without explicit
        # rebase config. CORRECTED keeps proleptic-gregorian semantics (the
        # right thing for modern readers); LEGACY would rebase to old hybrid
        # calendar for spark-2.x interop, which we dont need.
        .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # ship src/ to workers. without this, udfs that reference src.* fail
    # with `ModuleNotFoundError: No module named 'src'` when pyspark tries
    # to unpickle them on the worker side.
    zip_path = _zip_src_for_workers(project_root)
    if zip_path is not None:
        spark.sparkContext.addPyFile(str(zip_path))
        print(f"shipped {zip_path.name} to workers (so udfs can import src.*)")

    return spark
