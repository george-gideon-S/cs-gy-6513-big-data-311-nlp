"""
spark session bootstrap. one place to configure things so notebooks dont
each have to remember the delta lake config string.
"""
from pyspark.sql import SparkSession


def get_spark(app_name: str = "cs6513-311-nlp", local_cores: str = "*") -> SparkSession:
    """
    create or return the active spark session, configured for delta lake
    and reasonable defaults for 2m row workloads on a colab high-ram machine.

    Args:
        app_name: shows up in the spark ui, useful when youve got multiple
                  notebooks running.
        local_cores: "*" uses all cores. set to "4" or "8" if you want to
                     leave headroom for other stuff.

    Returns:
        a configured SparkSession.
    """
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
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
