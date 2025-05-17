"""
main.py

Entry point for the GLIMMER unsupervised multi-document summarization system.
This script loads configuration settings from config.yaml, initializes logging,
loads the dataset via DatasetLoader, instantiates the GLIMMER model and Trainer to run
the summarization pipeline, and evaluates the generated summary (if reference summaries
are provided).

Usage:
    python main.py

Dependencies:
    - PyYAML for configuration loading.
    - Standard modules: os, sys, json, logging.
    - Modules from the project: dataset_loader, model, trainer, evaluation.
    
All configuration values are set in config.yaml. Default values are provided
if configuration entries are missing.
"""

import os
import sys
import json
import logging
import yaml

from dataset_loader import DatasetLoader
from model import GLIMMER
from trainer import Trainer
from evaluation import Evaluation

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load the configuration settings from a YAML file.

    Args:
        config_path: Path to the configuration file (default "config.yaml").

    Returns:
        A dictionary with configuration settings.
    """
    if not os.path.exists(config_path):
        print(f"Configuration file '{config_path}' not found. Exiting.")
        sys.exit(1)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Configuration file is not valid YAML mapping.")
            return config
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        sys.exit(1)

def setup_logging(config: dict) -> None:
    """
    Set up the logging configuration based on the config dictionary.

    Args:
        config: Configuration dictionary containing "logging" settings.
    """
    log_level_str = config.get("logging", {}).get("level", "INFO")
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Logging is set to level: %s", log_level_str)

def load_reference_summaries(config: dict) -> dict:
    """
    Optionally load reference summaries for evaluation if a reference file is specified
    in the configuration under dataset.reference_path. The reference file is expected to be
    either JSON or YAML containing a mapping from document IDs or a key ("combined") to a summary.

    Args:
        config: Configuration dictionary.

    Returns:
        A dictionary mapping keys to reference summary strings, or an empty dictionary if not provided.
    """
    dataset_config = config.get("dataset", {})
    reference_path = dataset_config.get("reference_path", "")
    logger = logging.getLogger(__name__)
    if reference_path and os.path.exists(reference_path):
        logger.info("Loading reference summaries from '%s'.", reference_path)
        try:
            with open(reference_path, "r", encoding="utf-8") as rf:
                if reference_path.lower().endswith(('.yaml', '.yml')):
                    references = yaml.safe_load(rf)
                else:
                    references = json.load(rf)
                if not isinstance(references, dict):
                    logger.warning("Reference summaries file does not contain a mapping. Skipping evaluation.")
                    return {}
                logger.info("Loaded %d reference summaries.", len(references))
                return references
        except Exception as e:
            logger.error("Error loading reference summaries: %s", str(e))
            return {}
    else:
        logger.info("No reference summaries provided (or file not found). Skipping evaluation.")
        return {}

def main() -> None:
    """
    Main function orchestrating the GLIMMER summarization pipeline.
    """
    # Load configuration from config.yaml
    config = load_config("config.yaml")
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded successfully.")

    # Instantiate DatasetLoader and load data
    dataset_loader = DatasetLoader(config)
    data = dataset_loader.load_data()
    if not data:
        logger.error("No data loaded from the dataset. Exiting.")
        sys.exit(1)
    logger.info("Loaded %d document sets from the dataset.", len(data))

    # Instantiate the GLIMMER model with configuration parameters
    glimmer_model = GLIMMER(config)
    logger.info("GLIMMER model initialized.")

    # Instantiate Trainer with the GLIMMER model, data, and configuration; then run pipeline
    trainer = Trainer(glimmer_model, data, config)
    logger.info("Starting the summarization pipeline via Trainer.")
    generated_summary: str = trainer.run()
    if not generated_summary:
        logger.error("No summary was generated. Exiting.")
        sys.exit(1)
    logger.info("Generated summary (truncated to desired output length):\n%s", generated_summary)
    print("Generated Summary:\n", generated_summary)

    # Optionally, load reference summaries and evaluate if available.
    reference_summaries = load_reference_summaries(config)
    if reference_summaries:
        # For evaluation, wrap summaries in dictionaries.
        # Here, we assume a single combined summary under the key "combined".
        generated_dict = {"combined": generated_summary}
        # Reference summary should have matching key; default to empty string if not found.
        reference_dict = {"combined": reference_summaries.get("combined", "")}
        evaluator = Evaluation(glimmer_model, reference_dict, generated_dict, config)
        eval_metrics = evaluator.evaluate()
        logger.info("Evaluation Metrics: %s", eval_metrics)
        print("Evaluation Metrics:", eval_metrics)
    else:
        logger.info("Reference summaries not provided. Skipping evaluation step.")

if __name__ == "__main__":
    main()
