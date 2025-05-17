"""
dataset_loader.py

This module defines the DatasetLoader class which is responsible for loading and preprocessing
multi-document datasets for the GLIMMER summarization system. It reads raw input files (assumed
to be stored in a dataset-specific directory), performs basic text cleaning, applies sentence
segmentation using spaCy, and (if applicable) truncates documents according to configuration.
The output is a standardized list of document sets, where each document set is represented
as a dictionary with keys "doc_id", "sentences", and "raw_text".
"""

import os
import json
import math
import logging
from typing import List, Dict, Any

import spacy
import nltk

# Download NLTK punkt tokenizer if not already available.
nltk.download('punkt', quiet=True)


class DatasetLoader:
    """Class for loading and preprocessing multi-document datasets."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DatasetLoader with configuration.

        Args:
            config: A dictionary containing configuration parameters.
                    Expected keys include:
                      - "dataset": {"name": <dataset_name>, ...}
                      - "glimmer": {"truncation": {"multi_news": int, "duc_2004": int}, ...}
                      - "logging": {"level": <logging_level>}
        """
        self.config = config

        # Set dataset name, defaulting to "Multi-News" if not provided.
        self.dataset_name: str = config.get("dataset", {}).get("name", "Multi-News")
        self.dataset_name_lower: str = self.dataset_name.lower()

        # Determine truncation value based on dataset.
        # For Multi-News and DUC-2004, apply truncation; for Multi-XScience, no truncation is needed.
        glimmer_config: Dict[str, Any] = config.get("glimmer", {})
        truncation_config: Dict[str, Any] = glimmer_config.get("truncation", {})
        if self.dataset_name_lower == "multi-news":
            self.truncation_value: int = int(truncation_config.get("multi_news", 500))
        elif self.dataset_name_lower == "duc-2004":
            self.truncation_value: int = int(truncation_config.get("duc_2004", 500))
        else:
            self.truncation_value = None  # No truncation applied for Multi-XScience and others

        # Set the data path.
        # If a specific path is provided in config["dataset"]["path"], use it; otherwise, use a default folder.
        self.data_path: str = config.get("dataset", {}).get(
            "path", os.path.join("data", self.dataset_name.replace(" ", ""))
        )

        # Initialize spaCy language model for sentence segmentation, tokenization, POS, and NER.
        # Using the default small English model.
        self.nlp = spacy.load("en_core_web_sm")

        # Set up logging based on configuration.
        log_level_str: str = config.get("logging", {}).get("level", "INFO")
        numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
        logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatasetLoader initialized with dataset '%s' and data path '%s'.",
                         self.dataset_name, self.data_path)

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load and preprocess the dataset.

        Returns:
            A list of document set dictionaries. Each dictionary contains:
                - "doc_id": A unique identifier for the document set.
                - "sentences": A list of tokenized sentences (strings).
                - "raw_text": The original raw text (or combined text) of the document set.
        """
        dataset_list: List[Dict[str, Any]] = []

        if not os.path.exists(self.data_path):
            self.logger.error("Data path '%s' does not exist.", self.data_path)
            return dataset_list

        file_list: List[str] = [file for file in os.listdir(self.data_path)
                                if os.path.isfile(os.path.join(self.data_path, file))
                                and file.lower().endswith(('.json', '.txt'))]

        if not file_list:
            self.logger.warning("No valid files found in data path '%s'.", self.data_path)
            return dataset_list

        for file_name in file_list:
            file_path: str = os.path.join(self.data_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as fp:
                    # Attempt to load file as JSON; if fails, treat as plain text.
                    if file_name.lower().endswith('.json'):
                        data = json.load(fp)
                        # Expecting data to be a dict with a "documents" key containing a list.
                        if isinstance(data, dict) and "documents" in data and isinstance(data["documents"], list):
                            source_documents = data["documents"]
                        else:
                            # If JSON structure is not as expected, treat entire content as one document.
                            source_documents = [json.dumps(data)]
                    else:
                        # For plain text files, read the entire content.
                        file_content: str = fp.read()
                        source_documents = [file_content]

                # Apply truncation if needed (for Multi-News and DUC-2004).
                if self.truncation_value is not None:
                    source_documents = self._apply_truncation(source_documents, self.truncation_value)
                # For Multi-XScience or other datasets, do not apply truncation.

                # Combine source documents into a single raw text (separated by newlines).
                combined_text: str = "\n".join(doc.strip() for doc in source_documents if doc.strip())
                combined_text = combined_text.strip()

                # Use spaCy to segment the combined text into sentences.
                doc = self.nlp(combined_text)
                sentences: List[str] = [sent.text.strip() for sent in doc.sents if sent.text.strip() != ""]

                # Use the file name (without extension) as the document set ID.
                doc_id: str = os.path.splitext(file_name)[0]

                document_set: Dict[str, Any] = {
                    "doc_id": doc_id,
                    "sentences": sentences,
                    "raw_text": combined_text
                }
                dataset_list.append(document_set)
                self.logger.info("Loaded document set '%s' with %d sentences.", doc_id, len(sentences))
            except Exception as e:
                self.logger.error("Error loading file '%s'. Error: %s", file_path, str(e))

        self.logger.info("Total document sets loaded: %d", len(dataset_list))
        return dataset_list

    def _apply_truncation(self, documents: List[str], total_truncation: int) -> List[str]:
        """
        Apply truncation logic to a list of source documents.

        For datasets like Multi-News and DUC-2004, the truncation rule is to extract the first N/S tokens
        from each source document, where N is the total truncation length and S is the number of source documents.

        Args:
            documents: List of source document strings.
            total_truncation: Total number of tokens to retain per document set.

        Returns:
            A list of truncated document strings.
        """
        truncated_documents: List[str] = []
        num_docs: int = len(documents)
        if num_docs == 0:
            return truncated_documents

        # Calculate number of tokens per document.
        tokens_per_doc: int = math.floor(total_truncation / num_docs)
        self.logger.debug("Applying truncation: total_truncation=%d, num_docs=%d, tokens_per_doc=%d",
                          total_truncation, num_docs, tokens_per_doc)

        for doc_text in documents:
            # Tokenize the document using NLTK's word_tokenize.
            tokens = nltk.word_tokenize(doc_text)
            truncated_tokens = tokens[:tokens_per_doc]
            truncated_text = " ".join(truncated_tokens)
            truncated_documents.append(truncated_text)
        return truncated_documents


# If this module is executed directly, demonstrate a simple test of the DatasetLoader.
if __name__ == "__main__":
    # Example configuration dictionary based on config.yaml.
    example_config = {
        "dataset": {
            "name": "Multi-News",
            # Optional: "path": "data/MultiNews"  (will default if not provided)
        },
        "glimmer": {
            "truncation": {
                "multi_news": 500,
                "duc_2004": 500
            }
        },
        "logging": {
            "level": "INFO"
        }
    }
    loader = DatasetLoader(example_config)
    data = loader.load_data()
    print(f"Loaded {len(data)} document sets.")
