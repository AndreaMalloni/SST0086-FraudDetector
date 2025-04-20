from pathlib import Path

from kagglehub import dataset_download
from pyspark.sql import SparkSession
from rich.console import Console
from typer import Argument, BadParameter, Option, Typer

from fraud_detector.core.config import ConfigManager
from fraud_detector.core.logger import LoggingManager

app = Typer()
console = Console()


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

    spark = SparkSession.builder.appName("CreditCardFraudDetector").getOrCreate()

    download_path = dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
    csv_path = Path(download_path) / "creditcard_2023.csv"

    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)
    df.show(5)

    if output_name:
        logger.info(f"Output model name: {output_name}")
    else:
        console.print("[italic]No output name provided[/italic]")

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


if __name__ == "__main__":
    app()
