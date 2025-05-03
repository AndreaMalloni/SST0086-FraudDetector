from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, countDistinct, isnan, when

from fraud_detector.core.logger import LoggingManager

__all__ = ["analyze"]

def analyze(df: DataFrame) -> None:
    logger = LoggingManager().get_logger("Analysis")

    logger.info("Starting full dataset analysis.")

    logger.info("Schema of the DataFrame:")
    logger.info(df._jdf.schema().treeString())  # `printSchema()` equivalent

    logger.info("Descriptive statistics:")
    df.describe().toPandas().apply(lambda row: logger.info(row.to_string()), axis=1)

    logger.info("Null or NaN values per column:")
    null_counts = df.select([
        count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns
    ])
    null_counts_pandas = null_counts.toPandas()
    logger.info(null_counts_pandas.to_string(index=False))

    logger.info("Cardinality (number of distinct values) per column:")
    cardinalities = df.agg(*[countDistinct(col(c)).alias(c) for c in df.columns])
    logger.info(cardinalities.toPandas().to_string(index=False))

    numeric_cols = [
        field.name for field in df.schema.fields
        if str(field.dataType) in ["IntegerType", "DoubleType", "LongType", "FloatType"]
    ]
    if numeric_cols:
        logger.info(f"Numeric columns detected: {numeric_cols}")
        try:
            assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
            df_vector = assembler.transform(df.select(numeric_cols)).select("features")
            corr_matrix = Correlation.corr(df_vector, "features", "pearson").head()[0]
            logger.info(f"Pearson correlation matrix:\n{corr_matrix}")
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")

    logger.info("Value distribution for categorical columns (<=20 unique values):")
    for col_name in df.columns:
        try:
            distinct_count = df.select(col_name).distinct().count()
            if distinct_count <= 20:
                logger.info(f"{col_name} ({distinct_count} unique values):")
                values = df.groupBy(col_name).count().orderBy("count", ascending=False)
                values.toPandas().apply(lambda row: logger.info(row.to_string()), axis=1)
        except Exception as e:
            logger.warning(f"Skipping column '{col_name}' due to error: {e}")

    logger.info("Dataset analysis completed.")
