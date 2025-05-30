from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from xgboost.spark import SparkXGBClassifier

from fraud_detector.core.config import ConfigManager
from fraud_detector.core.logger import LoggingManager


def xgboost_spark(df, output_name: str | None = None):
    logger = LoggingManager().get_logger("Training")
    config = ConfigManager().config

    target_col = "Class"
    features_col = [c for c in df.columns if c not in [target_col, "id"]]
    logger.info(f"Colonne utilizzate per l'addestramento: {features_col}")

    # Verifica distribuzione classi
    class_distribution = df.groupBy(target_col).count().toPandas()
    logger.info("Distribuzione Classi:")
    for _, row in class_distribution.iterrows():
        logger.info(f"Classe {int(row[target_col])}: {row['count']} istanze")

    assembler = VectorAssembler(inputCols=features_col, outputCol="features")

    xgb = SparkXGBClassifier(
        features_col="features",
        label_col=target_col,
        prediction_col="prediction",
        missing=0,
        num_workers=1,
    )

    pipeline = Pipeline(stages=[assembler, xgb])

    # Build parameter grid from config
    param_grid_builder = ParamGridBuilder()
    tuning_params = config.get("tuning", {})

    if tuning_params:
        for param_name, values in tuning_params.items():
            param_grid_builder.addGrid(xgb.getParam(param_name), values)
        param_grid = param_grid_builder.build()

        evaluator = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderPR")

        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=8,
            seed=42,
        )

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        logger.info("Inizio tuning del modello con Cross-Validation...")

        cv_model = crossval.fit(train_df)
        logger.info("Tuning completato.")

        best_model = cv_model.bestModel
    else:
        logger.info("Nessun parametro di tuning specificato, addestramento del modello base...")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        best_model = pipeline.fit(train_df)

    predictions = best_model.transform(test_df)

    # METRICHE
    evaluator_auc = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
    evaluator_pr = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderPR")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target_col, metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol=target_col, metricName="precisionByLabel"
    )
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol=target_col, metricName="recallByLabel"
    )

    auc_roc = evaluator_auc.evaluate(predictions)
    pr_auc = evaluator_pr.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)

    logger.info(f"AUC ROC: {auc_roc:.4f}")
    logger.info(f"PR AUC: {pr_auc:.4f}")
    logger.info(f"F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # Confusion Matrix
    rdd = predictions.select("prediction", target_col).rdd.map(
        lambda row: (float(row[0]), float(row[1]))
    )
    metrics = MulticlassMetrics(rdd)
    matrix = metrics.confusionMatrix().toArray()

    tn, fp, fn, tp = matrix.flatten()
    logger.info("Confusion Matrix:")
    logger.info(f"TP: {int(tp)} | FP: {int(fp)} | TN: {int(tn)} | FN: {int(fn)}")

    return best_model
