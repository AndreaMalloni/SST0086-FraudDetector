from pyspark.sql.functions import col, countDistinct

from fraud_detector.core.logger import LoggingManager


def clean(df):
    logger = LoggingManager().get_logger("Processing")

    # 1. Rimuove righe con valori null o NaN
    df = df.na.drop()

    # 2. Rimuove duplicati
    initial_count = df.count()
    df = df.dropDuplicates()
    final_count = df.count()
    logger.info(f"Removed {initial_count - final_count} duplicate rows.")

    # 3. Rimuove colonne con un solo valore distinto (costanti)
    distinct_counts = df.agg(*[
        countDistinct(col(c)).alias(c) for c in df.columns
    ]).collect()[0].asDict()

    cols_to_drop = [col_name for col_name, distinct_count in distinct_counts.items() if distinct_count <= 1]
    if cols_to_drop:
        logger.info(f"Dropping constant columns: {cols_to_drop}")
        df = df.drop(*cols_to_drop)

    # 4. Converte in float colonne numeriche che non sono già in tipo numerico
    numeric_columns = [f.name for f in df.schema.fields if f.dataType.typeName() not in ["integer", "double", "float", "long"]]
    for col_name in numeric_columns:
        try:
            df = df.withColumn(col_name, col(col_name).cast("float"))
        except Exception as e:
            logger.warning(f"Could not cast column {col_name} to float: {e}")