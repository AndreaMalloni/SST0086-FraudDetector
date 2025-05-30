from datetime import datetime
from pathlib import Path
from typing import Optional

from kagglehub import dataset_download
import matplotlib
from pyspark.sql import SparkSession
from rich.console import Console
from typer import BadParameter, Option, Typer

matplotlib.use("Agg")  # Set the backend to non-interactive

import pandas as pd

from fraud_detector.core.config import ConfigManager
from fraud_detector.core.logger import LoggingManager
import fraud_detector.data.analysis as analysis
from fraud_detector.data.processing import clean
from fraud_detector.explain_methods import get_explanation_method
from fraud_detector.training import xgboost_spark

app = Typer()
console = Console()


def initialize(config_path: Optional[Path] = None, command: Optional[str] = None) -> None:
    config = ConfigManager().load(config_path) if config_path else {"logging": True}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(f"logs/[{timestamp}]")

    # Create necessary directories
    for folder in ["models", log_dir]:
        if not Path(folder).exists():
            folder.mkdir(parents=True)

    # Create plots directory inside log directory
    plots_dir = log_dir / "plots"
    if command == "explain" and not plots_dir.exists():
        plots_dir.mkdir(parents=True)

    manager = LoggingManager()

    # Define which loggers to initialize based on the command
    loggers_to_init = {
        "train": ["Training", "Processing"],
        "explain": ["Explanation"],
        "analyze": ["Analysis"],
    }

    # If no command specified, initialize all loggers
    loggers = (
        loggers_to_init.get(command, manager.supported_loggers)
        if command
        else manager.supported_loggers
    )

    for name in loggers:
        manager.init_logger(
            name=name,
            log_dir=str(log_dir),
            enabled=config.get("logging", True),
            rotation="time",
            when="midnight",
            backup_count=7,
        )


def validate_config_name(value: str) -> str:
    available = ConfigManager.available_configs()
    if value not in available:
        raise BadParameter(
            f"Configurazione '{value}' non valida'.\nDisponibili: {', '.join(sorted(available))}"
        )
    return value


def create_spark_session(config: dict) -> SparkSession:
    """Create a Spark session with the given configuration."""
    spark_config = config.get("spark", {})
    builder = SparkSession.builder.appName(spark_config.get("app_name", "CreditCardFraudDetector"))

    if "master" in spark_config:
        builder = builder.master(spark_config["master"])
    if "executor_memory" in spark_config:
        builder = builder.config("spark.executor.memory", spark_config["executor_memory"])
    if "driver_memory" in spark_config:
        builder = builder.config("spark.driver.memory", spark_config["driver_memory"])
    if spark_config.get("offheap_enabled", False):
        builder = builder.config("spark.memory.offHeap.enabled", "true")
        if "offheap_size" in spark_config:
            builder = builder.config("spark.memory.offHeap.size", spark_config["offheap_size"])
    if "shuffle_partitions" in spark_config:
        builder = builder.config(
            "spark.sql.shuffle.partitions", str(spark_config["shuffle_partitions"])
        )
    if "auto_broadcast_threshold" in spark_config:
        builder = builder.config(
            "spark.sql.autoBroadcastJoinThreshold", spark_config["auto_broadcast_threshold"]
        )

    return builder.getOrCreate()


@app.command()
def train(
    config_name: Optional[str] = Option(
        None, "--config", "-c", help="Name of the configuration file to use"
    ),
    model_name: Optional[str] = Option(None, "--model-name", "-m", help="Name of the model"),
    shuffle: Optional[bool] = Option(None, "--shuffle", help="Whether to shuffle the data"),
    validation_split: Optional[float] = Option(
        None, "--validation-split", "-v", help="Validation split ratio"
    ),
) -> None:
    # Initialize with config file if provided
    config_path = Path(f"config/{config_name}.yaml") if config_name else None
    initialize(config_path, "train")

    # Get base configuration
    config = ConfigManager().get_command_config("train") if config_name else {}

    # Override with command line arguments
    if model_name:
        config["model_name"] = model_name
    if shuffle is not None:
        config["shuffle"] = shuffle
    if validation_split:
        config["validation_split"] = validation_split

    logger = LoggingManager().get_logger("Training")
    console.rule("[bold green]Train Command")
    logger.info(
        f"Training started with config: {config_name if config_name else 'inline parameters'}"
    )

    # Create Spark session
    spark = create_spark_session(ConfigManager().config)
    spark.sparkContext.setLogLevel("ERROR")

    # Load dataset
    download_path = dataset_download(ConfigManager().config["kagglehub_dataset"])
    csv_path = Path(download_path) / ConfigManager().config["dataset_filename"]

    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)

    # Apply data processing if configured
    if config.get("data_processing", {}).get("clean_data", False):
        clean(df)
    if "repartition" in config.get("data_processing", {}):
        df = df.repartition(config["data_processing"]["repartition"])
    if config.get("data_processing", {}).get("cache_data", False):
        df.cache()

    console.rule("[bold blue]Training con SparkXGBClassifier")
    model = xgboost_spark(df, config["model_name"])

    # Save model in both formats
    spark_model_path = Path("models") / config["model_name"]
    model.write().overwrite().save(str(spark_model_path))
    logger.info(f"Modello salvato in formato Spark in: {spark_model_path}")

    # Save in XGBoost format for SHAP
    xgb_model = model.stages[-1].get_booster()
    xgb_model_path = spark_model_path / f"{config['model_name']}.json"
    xgb_model.save_model(str(xgb_model_path))
    logger.info(f"Modello salvato in formato XGBoost per SHAP in: {xgb_model_path}")

    logger.info("Training completed successfully.")


@app.command()
def explain(
    config_name: Optional[str] = Option(
        None, "--config", "-c", help="Name of the configuration file to use"
    ),
    model_name: Optional[str] = Option(None, "--model", "-m", help="Name of the model to explain"),
    method: Optional[str] = Option(
        None, "--method", help="Explanation method to use (shap or lime)"
    ),
    input_data: Optional[str] = Option(None, "--input", "-i", help="Path to input data"),
    threshold: Optional[float] = Option(None, "--threshold", "-t", help="Prediction threshold"),
    num_samples: Optional[int] = Option(
        None, "--samples", "-n", help="Number of samples to explain"
    ),
) -> None:
    # Initialize with config file if provided
    config_path = Path(f"config/{config_name}.yaml") if config_name else None
    initialize(config_path, "explain")

    # Get base configuration
    config = ConfigManager().get_command_config("explain") if config_name else {}

    # Override with command line arguments
    if model_name:
        config["model_name"] = model_name
    if method:
        config["method"] = method
    if input_data:
        config["input_data"] = input_data
    if threshold:
        config["threshold"] = threshold
    if num_samples:
        config["num_samples"] = num_samples

    # Validate required parameters
    if not config.get("method"):
        raise ValueError(
            "Explanation method must be specified in config file \
                         or via --method parameter"
        )
    if not config.get("model_name"):
        raise ValueError("Model name must be specified in config file or via --model parameter")

    logger = LoggingManager().get_logger("Explanation")
    console.rule("[bold blue]Explain Command")
    logger.info(
        f"Explaining model: {config['model_name']} using method: {config['method']} \
        with config: {config_name if config_name else 'inline parameters'}"
    )

    # Construct model path using the model name
    model_path = Path(f"models/{config['model_name']}_xgb.json")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load the dataset
    download_path = dataset_download(ConfigManager().config["kagglehub_dataset"])
    csv_path = Path(download_path) / ConfigManager().config["dataset_filename"]
    df = pd.read_csv(csv_path)

    try:
        # Get the appropriate explanation method and generate explanations
        explanation_method = get_explanation_method(config["method"])
        results = explanation_method.explain(model_path, df)

        # Log the explanation results
        logger.info("Explanation results:")
        if isinstance(results, dict):
            for key, value in results.items():
                logger.info(f"{key}: {value}")
        else:
            logger.info(str(results))

        logger.info(f"Explanation completed successfully using {config['method']} method")
    except Exception as e:
        logger.error(f"Error during explanation: {e!s}")
        raise


@app.command()
def analyze(
    config_name: Optional[str] = Option(
        None, "--config", "-c", help="Name of the configuration file to use"
    ),
    dataset_path: Optional[str] = Option(None, "--dataset", "-d", help="Path to the dataset"),
    output_dir: Optional[str] = Option(
        None, "--output-dir", "-o", help="Directory to save analysis results"
    ),
    generate_plots: Optional[bool] = Option(
        None, "--plots", "-p", help="Whether to generate plots"
    ),
    save_statistics: Optional[bool] = Option(
        None, "--stats", "-s", help="Whether to save statistics"
    ),
) -> None:
    # Initialize with config file if provided
    config_path = Path(f"config/{config_name}.yaml") if config_name else None
    initialize(config_path, "analyze")

    # Get base configuration
    config = ConfigManager().get_command_config("analyze") if config_name else {}

    # Override with command line arguments
    if dataset_path:
        config["dataset_path"] = dataset_path
    if generate_plots is not None:
        config["generate_plots"] = generate_plots
    if save_statistics is not None:
        config["save_statistics"] = save_statistics

    logger = LoggingManager().get_logger("Analysis")
    console.rule("[bold magenta]Analysis Command")
    logger.info(
        f"Running data analysis with config: {config_name if config_name else 'inline parameters'}"
    )

    # Create Spark session
    spark = create_spark_session(ConfigManager().config)

    # Load dataset
    download_path = dataset_download(ConfigManager().config["kagglehub_dataset"])
    csv_path = Path(download_path) / ConfigManager().config["dataset_filename"]

    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)
    analysis.analyze(df, config)

    logger.info("Analysis completed successfully.")


if __name__ == "__main__":
    app()
