"""
utils.py

This module provides utility functions for the GLIMMER unsupervised multi-document 
summarization system. It includes functions for:
  • Computing type-token ratios (TTR) and estimating the D parameter for TTR calibration.
  • Estimating sentence-level TTR values for clustering.
  • Calculating n-gram probabilities for fluency re-ranking using a Laplace-smoothed
    language model trained on the Brown corpus.
  • Computing cosine similarity between vectors and determining word similarity based on
    a provided embedding model.
  • Extracting word context and computing context coincidence (using Jaccard similarity) 
    to help measure “context coincidence” when mapping words in the word graph.
  • Converting token lists back into text and loading stopwords.

All thresholds and hyperparameters (e.g., similar word threshold, sigma value) are to be 
obtained from the shared configuration (as provided in config.yaml). Default values are set 
here when not explicitly provided via configuration.
"""

import math
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import nltk
from nltk.util import ngrams
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.corpus import brown, stopwords

# Ensure required NLTK resources are downloaded.
nltk.download('punkt', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('stopwords', quiet=True)

# Global cache for n-gram language models to avoid retraining.
NGRAM_MODELS: Dict[int, Laplace] = {}

def compute_ttr(token_list: List[str]) -> float:
    """
    Compute the Type-Token Ratio (TTR) of a list of tokens.
    
    Args:
        token_list: List of token strings.
    
    Returns:
        TTR as a float: (number of unique tokens) / (total number of tokens).
        Returns 0.0 for an empty token list.
    """
    if not token_list:
        return 0.0
    unique_tokens = set(token_list)
    return len(unique_tokens) / float(len(token_list))

def estimate_d_parameter(text: str) -> float:
    """
    Estimate the D parameter used for calibrating the estimated TTR per sentence.
    
    The function samples segments of the input text for lengths ranging from 36 to 50 tokens,
    computes the TTR for each sample (using a few sliding windows for longer texts), and returns
    the average TTR over these samples as the D parameter.
    
    Args:
        text: The raw input text as a string.
    
    Returns:
        Estimated D value as a float.
    """
    tokens = nltk.word_tokenize(text)
    if not tokens:
        return 0.0

    sample_ttrs: List[float] = []
    # Range from 36 to 50 tokens
    for L in range(36, 51):
        if len(tokens) < L:
            # If text is shorter than L, use the whole text.
            sample_ttrs.append(compute_ttr(tokens))
        else:
            # For longer texts, sample multiple windows to get a stable estimate.
            window_ttrs: List[float] = []
            num_windows = 3
            max_start = len(tokens) - L
            step = max(1, max_start // num_windows)
            for start in range(0, max_start + 1, step):
                window = tokens[start:start + L]
                window_ttrs.append(compute_ttr(window))
                if len(window_ttrs) >= num_windows:
                    break
            if window_ttrs:
                avg_window_ttr = sum(window_ttrs) / len(window_ttrs)
                sample_ttrs.append(avg_window_ttr)
    D = sum(sample_ttrs) / len(sample_ttrs) if sample_ttrs else 0.0
    return D

def estimate_sentence_ttr(sentence_tokens: List[str], D: float) -> Dict[str, float]:
    """
    Estimate the true and calibrated TTR for a given sentence.
    
    Args:
        sentence_tokens: List of tokens (strings) for the sentence.
        D: The D parameter estimated from the entire input text.
        
    Returns:
        A dictionary with keys:
          - "true_ttr": The actual TTR computed for the sentence.
          - "estimated_ttr": The calibrated (estimated) TTR value. 
            (Currently, this is set to D as a simple heuristic.)
    """
    true_ttr = compute_ttr(sentence_tokens)
    # In this basic implementation, we use the estimated D value directly.
    estimated_ttr = D
    return {"true_ttr": true_ttr, "estimated_ttr": estimated_ttr}

def get_ngram_model(n: int) -> Laplace:
    """
    Retrieve or train a Laplace-smoothed n-gram language model using the Brown corpus.
    
    Args:
        n: The order of the n-gram model (e.g., 3 for trigram).
        
    Returns:
        A trained Laplace language model.
    """
    global NGRAM_MODELS
    if n in NGRAM_MODELS:
        return NGRAM_MODELS[n]
    
    # Train the model on the Brown corpus.
    brown_sents = list(brown.sents())
    train_data, padded_sents = padded_everygram_pipeline(n, brown_sents)
    model = Laplace(n)
    model.fit(train_data, padded_sents)
    NGRAM_MODELS[n] = model
    return model

def calculate_ngram_probability(token_sequence: List[str], n: int) -> float:
    """
    Calculate the sum of n-gram probabilities for a given token sequence using a pre-trained 
    n-gram language model.
    
    This function uses a Laplace-smoothed language model (trained on the Brown corpus) to 
    compute the probabilities for each n-gram (with padding) and returns the sum of these probabilities.
    
    Args:
        token_sequence: List of token strings representing the candidate sequence.
        n: Order of the n-gram model (e.g., 3 for trigram).
    
    Returns:
        A float representing the aggregated n-gram probability sum for the sequence.
    """
    model = get_ngram_model(n)
    padded_sequence = list(pad_both_ends(token_sequence, n))
    ngram_list = list(ngrams(padded_sequence, n))
    total_probability = 0.0
    for ngram in ngram_list:
        context = ngram[:-1]
        word = ngram[-1]
        prob = model.score(word, context)
        total_probability += prob
    return total_probability

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    
    Args:
        vec1: A numpy array representing the first vector.
        vec2: A numpy array representing the second vector.
    
    Returns:
        Cosine similarity as a float in [0, 1]. Returns 0.0 if either vector is zero.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def is_similar_word(word1: str, word2: str, embedding_model: Any, threshold: float = 0.65) -> bool:
    """
    Determine whether two words are similar based on cosine similarity of their embeddings.
    
    Args:
        word1: First word as a string.
        word2: Second word as a string.
        embedding_model: An embedding model that provides an 'encode' method for words.
        threshold: Similarity threshold; defaults to 0.65 (should be obtained from config["glimmer"]["similar_word_threshold"]).
    
    Returns:
        True if cosine similarity of the word embeddings is greater than or equal to the threshold, False otherwise.
    """
    vec1 = embedding_model.encode(word1)
    vec2 = embedding_model.encode(word2)
    similarity = compute_cosine_similarity(np.array(vec1), np.array(vec2))
    return similarity >= threshold

def get_word_context(word_index: int, sentence_tokens: List[str], window: int = 2) -> List[str]:
    """
    Extract the context for a word in a sentence.
    
    The context is defined as the list of tokens within a specified window before and after
    the target word.
    
    Args:
        word_index: The index of the target word in the sentence.
        sentence_tokens: List of tokens (strings) representing the sentence.
        window: Number of tokens to include on each side of the target word (default is 2).
    
    Returns:
        A list of tokens representing the context.
    """
    left_context = sentence_tokens[max(0, word_index - window):word_index]
    right_context = sentence_tokens[word_index + 1: word_index + 1 + window]
    return left_context + right_context

def compute_context_coincidence(context1: List[str], context2: List[str]) -> float:
    """
    Compute the context coincidence between two contexts using Jaccard similarity.
    
    Args:
        context1: List of tokens representing the first context.
        context2: List of tokens representing the second context.
    
    Returns:
        A float representing the Jaccard similarity between the two contexts.
    """
    set1 = set(context1)
    set2 = set(context2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def tokens_to_text(token_list: List[str]) -> str:
    """
    Convert a list of tokens into a well-formed text string.
    
    This function simply joins tokens with a space. Further refinement (e.g., handling punctuation)
    can be implemented if needed.
    
    Args:
        token_list: A list of tokens (strings).
    
    Returns:
        A string representing the combined text.
    """
    return " ".join(token_list)

def load_stopwords() -> set:
    """
    Load a set of English stopwords using NLTK.
    
    Returns:
        A set of stopword strings.
    """
    return set(stopwords.words('english'))
