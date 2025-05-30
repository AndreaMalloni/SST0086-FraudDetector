from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, countDistinct, isnan, when
import seaborn as sns

from fraud_detector.core.logger import LoggingManager

__all__ = ["analyze"]


def analyze(df: DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Analyze the dataset and return results.

    Args:
        df: Spark DataFrame containing the data
        config: Configuration dictionary

    Returns:
        Dictionary containing analysis results:
        - statistics: DataFrame with basic statistics
        - correlation_matrix: DataFrame with correlation matrix
        - plots: Dictionary of matplotlib figures
    """
    logger = LoggingManager().get_logger("Analysis")

    logger.info("Starting full dataset analysis.")

    logger.info("Schema of the DataFrame:")
    logger.info(df._jdf.schema().treeString())  # `printSchema()` equivalent

    logger.info("Descriptive statistics:")
    df.describe().toPandas().apply(lambda row: logger.info(row.to_string()), axis=1)

    logger.info("Null or NaN values per column:")
    null_counts = df.select(
        [count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in df.columns]
    )
    null_counts_pandas = null_counts.toPandas()
    logger.info(null_counts_pandas.to_string(index=False))

    logger.info("Cardinality (number of distinct values) per column:")
    cardinalities = df.agg(*[countDistinct(col(c)).alias(c) for c in df.columns])
    logger.info(cardinalities.toPandas().to_string(index=False))

    numeric_cols = [
        field.name
        for field in df.schema.fields
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

    # Convert to pandas for analysis
    pdf = df.toPandas()

    results = {"statistics": pd.DataFrame(), "plots": {}}

    # Basic statistics
    stats = pdf.describe()
    results["statistics"] = stats

    # Correlation analysis
    if "correlation" in config.get("analysis_params", {}).get("statistics", []):
        corr_matrix = pdf.corr()
        results["correlation_matrix"] = corr_matrix

        if "correlation" in config.get("analysis_params", {}).get("plot_types", []):
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
            plt.title("Correlation Matrix")
            results["plots"]["correlation_matrix"] = plt.gcf()

    # Distribution plots
    if "distribution" in config.get("analysis_params", {}).get("plot_types", []):
        for column_name in pdf.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=pdf, x=column_name, hue="Class", multiple="stack")
            plt.title(f"Distribution of {column_name}")
            results["plots"][f"distribution_{column_name}"] = plt.gcf()

    # Box plots
    if "box" in config.get("analysis_params", {}).get("plot_types", []):
        for column_name in pdf.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=pdf, x="Class", y=column_name)
            plt.title(f"Box Plot of {column_name} by Class")
            results["plots"][f"box_{column_name}"] = plt.gcf()

    # Scatter plots
    if "scatter" in config.get("analysis_params", {}).get("plot_types", []):
        numeric_cols = pdf.select_dtypes(include=[np.number]).columns
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=pdf, x=col1, y=col2, hue="Class")
                plt.title(f"Scatter Plot: {col1} vs {col2}")
                results["plots"][f"scatter_{col1}_{col2}"] = plt.gcf()

    return results
