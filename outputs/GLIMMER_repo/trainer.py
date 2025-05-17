"""
trainer.py

This module defines the Trainer class which orchestrates the execution of the GLIMMER
unsupervised multi-document summarization pipeline. It takes a pre-initialized GLIMMER
model instance and a preprocessed dataset (loaded via DatasetLoader), then runs the pipeline,
logs intermediate steps (such as the number of sentences processed and runtime), and returns
the final generated summary. All configuration values (e.g., fixed cluster number, similarity thresholds)
are obtained from the configuration dictionary (e.g., parsed from config.yaml).
"""

import time
import logging
from typing import Any, List, Dict

# Import the GLIMMER model from model.py.
from model import GLIMMER


class Trainer:
    """
    Trainer class that orchestrates the full GLIMMER pipeline.

    Attributes:
        model (GLIMMER): An instance of the GLIMMER summarization model.
        data (List[Dict[str, Any]]): The preprocessed dataset loaded by DatasetLoader.
        config (Dict[str, Any]): Configuration settings parsed from config.yaml.
        logger (logging.Logger): Logger for recording key information.
    """

    def __init__(self, model: GLIMMER, data: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """
        Initialize the Trainer.

        Args:
            model: An instance of the GLIMMER model.
            data: Preprocessed dataset (a list of document sets, each a dict with keys "doc_id", "sentences", and "raw_text").
            config: Configuration dictionary (e.g., parsed from config.yaml) containing keys for logging, glimmer, clustering, dataset, etc.
        """
        self.model: GLIMMER = model
        self.data: List[Dict[str, Any]] = data
        self.config: Dict[str, Any] = config

        # Set up logging based on configuration.
        log_level_str: str = config.get("logging", {}).get("level", "INFO")
        numeric_level: int = getattr(logging, log_level_str.upper(), logging.INFO)
        logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")
        self.logger = logging.getLogger(__name__)
        self.logger.info("Trainer initialized with configuration.")

    def run(self) -> str:
        """
        Execute the GLIMMER summarization pipeline.

        Steps:
            1. Logs the start of the summarization process and records the start time.
            2. Counts the total number of sentences across the preprocessed dataset.
            3. Calls the GLIMMER model's run_pipeline() method with the dataset.
            4. Logs the time taken and final summary output.
            5. Returns the final summary string.

        Returns:
            A string containing the final multi-document summary.
        """
        try:
            self.logger.info("Starting the summarization pipeline.")
            start_time: float = time.time()

            # Count total sentences in the dataset for logging purposes.
            total_sentences: int = 0
            for document in self.data:
                sentences: List[str] = document.get("sentences", [])
                total_sentences += len(sentences)
            self.logger.info("Total sentences in dataset: %d", total_sentences)

            # Run the GLIMMER pipeline; this will build the sentence graph, perform clustering,
            # generate summaries for each cluster, and combine them into a final output.
            summary: str = self.model.run_pipeline(self.data)

            elapsed_time: float = time.time() - start_time
            self.logger.info("Summarization pipeline completed in %.2f seconds.", elapsed_time)
            self.logger.info("Generated summary: %s", summary)

            return summary
        except Exception as error:
            self.logger.error("Error during the summarization pipeline: %s", str(error), exc_info=True)
            return ""


if __name__ == "__main__":
    # For demonstration purposes only.
    # In practice, the dataset should be loaded via DatasetLoader and configuration through config.yaml.

    # Sample configuration dictionary (default values taken from config.yaml).
    sample_config: Dict[str, Any] = {
        "logging": {"level": "INFO"},
        "glimmer": {
            "similar_word_threshold": 0.65,
            "sentence_similarity_threshold": 0.98,
            "sigma": 0.05,
            "beta": 4,
            "min_summary_length": 6,
            "output_length": {"multi_news": 256, "multi_xscience": 128, "duc_2004": 128},
            "truncation": {"multi_news": 500, "duc_2004": 500},
        },
        "clustering": {
            "fixed_cluster_number": {"multi_news": 9, "multi_xscience": 7, "duc_2004": 5}
        },
        "dataset": {"name": "Multi-News"}
    }

    # Dummy dataset for testing (normally provided by DatasetLoader).
    dummy_data: List[Dict[str, Any]] = [{
        "doc_id": "dummy_doc",
        "sentences": [
            "This is the first sentence of the dummy document.",
            "The second sentence contains interesting information.",
            "Finally, the third sentence wraps up the summary."
        ],
        "raw_text": (
            "This is the first sentence of the dummy document. "
            "The second sentence contains interesting information. "
            "Finally, the third sentence wraps up the summary."
        )
    }]

    # Initialize the GLIMMER model with the sample configuration.
    glimmer_model: GLIMMER = GLIMMER(sample_config)
    # Initialize the Trainer with the GLIMMER model, dummy dataset, and configuration.
    trainer: Trainer = Trainer(glimmer_model, dummy_data, sample_config)
    # Run the pipeline and print the final summary.
    final_summary: str = trainer.run()
    print("Final Summary:", final_summary)
