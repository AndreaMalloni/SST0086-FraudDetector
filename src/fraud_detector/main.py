import os
from pathlib import Path

from kagglehub import dataset_download
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from rich.console import Console
from typer import Argument, BadParameter, Option, Typer
from xgboost.spark import SparkXGBClassifier

from fraud_detector.core.config import ConfigManager
from fraud_detector.core.logger import LoggingManager
import fraud_detector.data.analysis as analysis
from fraud_detector.data.processing import clean

app = Typer()
console = Console()

def train_xgboost_spark(df, output_name: str | None = None):
    console.rule("[bold blue]Training con SparkXGBClassifier")
    
    # Separazione target e feature
    target_col = "Class"
    features_col = [c for c in df.columns if c != target_col]

    # Assembla le feature in un unico vettore
    assembler = VectorAssembler(inputCols=features_col, outputCol="features")

    # Inizializza il classificatore XGBoost
    xgb = SparkXGBClassifier(
        features_col="features",
        label_col=target_col,
        prediction_col="prediction",
        numRound=100,
        maxDepth=6,
        eta=0.1,
        num_workers=1,
        missing=0
    )

    # Costruisce il pipeline Spark
    pipeline = Pipeline(stages=[assembler, xgb])

    # Split train/test
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Fit del modello
    model = pipeline.fit(train_df)

    # Valutazione
    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol=target_col)
    auc = evaluator.evaluate(predictions)
    console.print(f"[green]AUC ROC sul test set: {auc:.4f}[/green]")

    # Salvataggio del modello
    model_name = output_name or "spark_xgb_model"
    model_path = f"models/{model_name}"
    
    model.write().overwrite().save(model_path)
    console.print(f"[green]Modello salvato in:[/green] {model_path}")

def initialize(config_path: Path) -> None:
    config = ConfigManager().load(config_path)
    for folder in ["models", "logs"]:
        path = Path(folder)
        if not path.exists():
            path.mkdir(parents=True)

    manager = LoggingManager()
    for name in manager.supported_loggers:
        manager.init_logger(
            name=name,
            enabled=config.get("logging", True),
            rotation="time",
            when="midnight",
            backup_count=7,
        )


def validate_config_name(value: str, command: str) -> str:
    available = ConfigManager.available_configs().get(command, [])
    if value not in available:
        raise BadParameter(
            f"Configurazione '{value}' non valida per il comando '{command}'.\n"
            f"Disponibili: {', '.join(sorted(available))}"
        )
    return value


@app.command()
def train(
    config_name: str = Argument(
        ...,
        callback=lambda val: validate_config_name(val, "train"),
        help="Name of the configuration to use for training",
    ),
    output_name: str = Option(None, "--output", "-o", help="Optional name for the model output"),
) -> None:
    initialize(Path(f"config/{config_name}.yaml"))

    logger = LoggingManager().get_logger("Training")
    console.rule("[bold green]Train Command")
    logger.info(f"Training started with configuration: {config_name}")

    spark = SparkSession.builder.appName("CreditCardFraudDetector") \
        .master("local[1]") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "4g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    download_path = dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
    csv_path = Path(download_path) / "creditcard_2023.csv"

    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)
    #clean(df)
    df = df.repartition(1)
    train_xgboost_spark(df, "test")

    # Add your training logic here
    logger.info("Training completed successfully.")


@app.command()
def run(
    model_name: str = Argument(..., help="Name of the model to run"),
    config_name: str = Argument(
        ...,
        callback=lambda val: validate_config_name(val, "run"),
        help="Name of the configuration to use",
    ),
) -> None:
    initialize(Path(f"config/{config_name}.yaml"))
    logger = LoggingManager().get_logger("Running")

    console.rule("[bold blue]Run Command")
    logger.info(f"Running model: {model_name} with config: {config_name}")

    # Add your running logic here
    logger.info("Run completed successfully.")


@app.command()
def analyze(
    config_name: str = Argument(
        ..., callback=lambda val: validate_config_name(val, "train"),
        help="Name of the configuration to use for analysis",
    )
) -> None:
    initialize(Path(f"config/{config_name}.yaml"))

    logger = LoggingManager().get_logger("Analysis")
    console.rule("[bold magenta]Analysis Command")
    logger.info(f"Running data analysis with config: {config_name}")

    spark = SparkSession.builder.appName("DataAnalysis") \
        .getOrCreate()

    download_path = dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
    csv_path = Path(download_path) / "creditcard_2023.csv"

    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)
    analysis.analyze(df)

    logger.info("Analysis completed successfully.")

if __name__ == "__main__":
    app()
