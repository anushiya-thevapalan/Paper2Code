"""
model.py

This module defines the GLIMMER class which implements the core unsupervised 
multi-document summarization algorithm described in the GLIMMER paper. The class
includes methods for constructing a sentence graph via multiple lexical and semantic 
indicators, identifying semantic clusters using spectral clustering (with a TTR‐based 
method), and summarizing each cluster through a directed word graph with fluency re‐ranking.
All configuration values are obtained from a shared configuration dictionary passed at initialization.
The implementation uses spaCy, NLTK, networkx, NumPy, scikit-learn, and SentenceTransformer.
"""

import math
import logging
import networkx as nx
import numpy as np
from numpy.linalg import eigh
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Tuple, Set

import spacy
import nltk
from nltk.corpus import wordnet

# Import utility functions from utils.py
from utils import (
    compute_ttr,
    estimate_d_parameter,
    estimate_sentence_ttr,
    calculate_ngram_probability,
    compute_cosine_similarity,
    is_similar_word,
    get_word_context,
    compute_context_coincidence,
    tokens_to_text,
    load_stopwords
)

from sentence_transformers import SentenceTransformer

# Ensure required NLTK resources have been downloaded
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Predefined list of conjunctions (using a sample set; in practice, list should include 39 conjunctions)
CONJUNCTIONS: List[str] = [
    "and", "but", "or", "nor", "for", "yet", "so", "although", "because", "since",
    "unless", "while", "whereas", "though", "if", "after", "before", "once", "even",
    "when", "whenever", "where", "wherever", "until", "as", "as if", "as though"
]

# List of non-notional verbs (simple common auxiliaries and linking verbs)
NON_NOTIONAL_VERBS: List[str] = [
    "be", "have", "do", "is", "are", "was", "were", "been", "being", "had", "did"
]

def get_nominalizations(token: spacy.tokens.Token) -> List[str]:
    """
    Given a spaCy token assumed to be a verb, use NLTK's WordNet to extract possible nominalizations.
    Only include candidate lemmas that have at least one noun synset.
    
    Args:
        token: A spaCy token representing a verb.
        
    Returns:
        A list of nominalization strings.
    """
    nominalizations: Set[str] = set()
    for syn in wordnet.synsets(token.lemma_, pos=wordnet.VERB):
        for lemma_name in syn.lemma_names():
            # Check if this lemma can be found as a noun in WordNet
            noun_synsets = wordnet.synsets(lemma_name, pos=wordnet.NOUN)
            if noun_synsets:
                nominalizations.add(lemma_name.lower())
    return list(nominalizations)


class GLIMMER:
    """
    GLIMMER: Graph and LexIcal features based unsupervised Multi-docuMEnt Summarization.

    Public methods:
        __init__(params: dict)
        construct_sentence_graph(sentences: List[str]) -> nx.Graph
        semantic_cluster_identification(graph: nx.Graph, sentences: List[str], raw_text: str) -> List[List[int]]
        cluster_summarization(clusters: List[List[int]], sentences: List[str]) -> List[str]
        run_pipeline(data: List[Dict[str, Any]]) -> str
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the GLIMMER model with configuration parameters.
        
        Args:
            params: Configuration dictionary containing required keys:
                - "glimmer": thresholds and output_length, truncation, etc.
                - "clustering": fixed_cluster_number settings.
                - "dataset": name of dataset (e.g., "Multi-News", "Multi-XScience", "DUC-2004").
        """
        self.params: Dict[str, Any] = params

        glimmer_config: Dict[str, Any] = self.params.get("glimmer", {})
        clustering_config: Dict[str, Any] = self.params.get("clustering", {})
        dataset_config: Dict[str, Any] = self.params.get("dataset", {})

        self.similar_word_threshold: float = float(glimmer_config.get("similar_word_threshold", 0.65))
        self.sentence_similarity_threshold: float = float(glimmer_config.get("sentence_similarity_threshold", 0.98))
        self.sigma: float = float(glimmer_config.get("sigma", 0.05))
        self.beta: float = float(glimmer_config.get("beta", 4))
        self.min_summary_length: int = int(glimmer_config.get("min_summary_length", 6))

        # Determine output length based on dataset name
        output_length_config: Dict[str, Any] = glimmer_config.get("output_length", {})
        dataset_name: str = dataset_config.get("name", "Multi-News").lower()
        if dataset_name == "multi-news":
            self.output_length: int = int(output_length_config.get("multi_news", 256))
        elif dataset_name == "multi-xscience":
            self.output_length = int(output_length_config.get("multi_xscience", 128))
        elif dataset_name == "duc-2004":
            self.output_length = int(output_length_config.get("duc_2004", 128))
        else:
            self.output_length = 256  # default

        # Fixed cluster number fallback based on dataset
        fixed_cluster_numbers: Dict[str, Any] = clustering_config.get("fixed_cluster_number", {})
        if dataset_name == "multi-news":
            self.fixed_cluster_number: int = int(fixed_cluster_numbers.get("multi_news", 9))
        elif dataset_name == "multi-xscience":
            self.fixed_cluster_number = int(fixed_cluster_numbers.get("multi_xscience", 7))
        elif dataset_name == "duc-2004":
            self.fixed_cluster_number = int(fixed_cluster_numbers.get("duc_2004", 5))
        else:
            self.fixed_cluster_number = 9

        # Initialize spaCy model (for tokenization, POS, and NER)
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize sentence embedding model (using 'deberta-xlarge-mnli' as specified in the paper)
        try:
            self.embed_model = SentenceTransformer("deberta-xlarge-mnli")
        except Exception:
            # Fallback to a smaller model if the specified model is not available.
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("GLIMMER initialized with dataset '%s'", dataset_config.get("name", "Multi-News"))

    def construct_sentence_graph(self, sentences: List[str]) -> nx.Graph:
        """
        Construct a binary undirected sentence graph where each node represents a sentence.
        Edges are added if at least one indicator is triggered:
              1. Deverbal Noun Reference (adjacent sentences)
              2. Conjunction indicator (adjacent sentences)
              3. Entity Consistency (all pairs)
              4. Semantic Similarity (all pairs)
        
        Args:
            sentences: List of sentence strings.
        
        Returns:
            A networkx Graph object with nodes as sentence indices.
        """
        graph = nx.Graph()
        num_sentences: int = len(sentences)
        for idx in range(num_sentences):
            graph.add_node(idx)

        # Precompute spaCy docs for each sentence and sentence embeddings
        spacy_docs: List[spacy.tokens.Doc] = [self.nlp(sent) for sent in sentences]
        embeddings: np.ndarray = self.embed_model.encode(sentences, convert_to_numpy=True)

        # Iterate over all sentence pairs (i, j) with i < j
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                indicator_triggered: bool = False

                # For adjacent sentences, check deverbal noun reference and conjunction indicator
                if j == i + 1:
                    # Indicator 1: Deverbal Noun Reference
                    nominalizations: List[str] = []
                    # Extract verbs from sentence i
                    for token in spacy_docs[i]:
                        if token.pos_ == "VERB" and token.lemma_.lower() not in NON_NOTIONAL_VERBS:
                            nominals = get_nominalizations(token)
                            nominalizations.extend(nominals)
                    # If any noun in sentence j is similar to one of the nominalizations, trigger indicator.
                    if nominalizations:
                        for token in spacy_docs[j]:
                            if token.pos_ in ("NOUN", "PROPN"):
                                for nominal in nominalizations:
                                    if is_similar_word(token.text, nominal, self.embed_model, threshold=self.similar_word_threshold):
                                        indicator_triggered = True
                                        break
                                if indicator_triggered:
                                    break

                    # Indicator 2: Conjunction indicator - check if sentence j begins with a known conjunction
                    first_token = spacy_docs[j][0].text.lower() if len(spacy_docs[j]) > 0 else ""
                    if first_token in CONJUNCTIONS:
                        indicator_triggered = True

                # Indicator 3: Entity Consistency - check if sentences share at least one named entity with same label
                ents_i = {(ent.text.lower(), ent.label_) for ent in spacy_docs[i].ents}
                ents_j = {(ent.text.lower(), ent.label_) for ent in spacy_docs[j].ents}
                if ents_i.intersection(ents_j):
                    indicator_triggered = True

                # Indicator 4: Semantic Similarity - use cosine similarity of sentence embeddings
                cosine_sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                if cosine_sim >= self.sentence_similarity_threshold:
                    indicator_triggered = True

                if indicator_triggered:
                    graph.add_edge(i, j)
                    self.logger.debug("Edge added between sentence %d and %d", i, j)

        self.logger.info("Constructed sentence graph with %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def semantic_cluster_identification(self, graph: nx.Graph, sentences: List[str], raw_text: str) -> List[List[int]]:
        """
        Identify semantic clusters by applying spectral clustering on the sentence graph.
        Uses a TTR-based method to determine the optimal number of clusters.
        
        Args:
            graph: The networkx Graph from construct_sentence_graph.
            sentences: List of sentence texts.
            raw_text: Combined raw text for computing global TTR and D parameter.
        
        Returns:
            A list of clusters, where each cluster is a list of sentence indices.
        """
        # Convert graph to n x n binary adjacency matrix W
        nodes_sorted = sorted(graph.nodes())
        W: np.ndarray = nx.to_numpy_array(graph, nodelist=nodes_sorted)
        n_sent: int = len(sentences)

        # Compute global TTR using the combined raw text
        global_tokens = nltk.word_tokenize(raw_text)
        global_TTR: float = compute_ttr(global_tokens)

        # Estimate D parameter from raw text
        D_value: float = estimate_d_parameter(raw_text)

        # Count high-TTR and low-TTR sentences
        n_low: int = 0
        n_high: int = 0
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            ttr_result = estimate_sentence_ttr(tokens, D_value)
            true_ttr = ttr_result.get("true_ttr", 0.0)
            estimated_ttr = ttr_result.get("estimated_ttr", D_value)
            if true_ttr < estimated_ttr * (1 - self.sigma):
                n_low += 1
            elif true_ttr >= estimated_ttr * (1 + self.sigma):
                n_high += 1

        k_candidate: int = math.floor((n_sent - self.beta * (n_low - n_high)) * global_TTR)
        if k_candidate <= 0 or k_candidate > n_sent:
            self.logger.warning("Calculated cluster number %d is invalid. Falling back to fixed cluster number %d.", k_candidate, self.fixed_cluster_number)
            k_candidate = self.fixed_cluster_number

        self.logger.info("Determined number of clusters (k): %d", k_candidate)

        # Compute Laplacian: L = D - W
        degree_matrix = np.diag(W.sum(axis=1))
        L = degree_matrix - W

        # Compute eigenvectors of L
        eigenvalues, eigenvectors = eigh(L)
        # Select first k_candidate eigenvectors (columns)
        U = eigenvectors[:, :k_candidate]

        # Apply k-means clustering on rows of U
        kmeans = KMeans(n_clusters=k_candidate, random_state=42)
        labels = kmeans.fit_predict(U)
        
        # Group sentence indices by cluster label
        clusters: List[List[int]] = [[] for _ in range(k_candidate)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        self.logger.info("Semantic clustering produced %d clusters.", len(clusters))
        return clusters

    def cluster_summarization(self, clusters: List[List[int]], sentences: List[str]) -> List[str]:
        """
        For each semantic cluster, construct a directed word graph and extract a summary 
        sentence through shortest path selection with fluency adjustment.

        Args:
            clusters: List of clusters (each cluster is a list of sentence indices).
            sentences: Full list of sentence texts.
        
        Returns:
            A list of summary sentences, one per cluster.
        """
        cluster_summaries: List[str] = []
        epsilon: float = 1e-6  # small constant to avoid division by zero

        # Process clusters in natural order based on the minimum sentence index in each cluster
        sorted_clusters = sorted(clusters, key=lambda cl: min(cl) if cl else float('inf'))

        for cluster in sorted_clusters:
            if not cluster:
                continue

            # Build directed word graph for the cluster
            word_graph = nx.DiGraph()
            word_graph.add_node("START", word="START")
            word_graph.add_node("END", word="END")

            # Global mapping for word nodes across sentences in this cluster
            global_node_map: Dict[Tuple[str, str], str] = {}  # key: (lowercase, pos); value: node id
            node_frequency: Dict[str, int] = {}
            node_counter: int = 0

            # For storing the word path for each sentence (list of node ids)
            sentence_node_paths: List[List[str]] = []

            for sent_idx in sorted(cluster):
                sent = sentences[sent_idx]
                doc = self.nlp(sent)
                local_used: Set[Tuple[str, str]] = set()
                node_path: List[str] = []
                for token in doc:
                    # Skip if token is punctuation
                    if token.is_punct:
                        continue
                    key = (token.text.lower(), token.pos_)
                    # Ensure within the same sentence, duplicate mapping is avoided:
                    if key in local_used:
                        # Create a unique key for this occurrence
                        unique_key = (token.text.lower() + f"_{sent_idx}_{len(node_path)}", token.pos_)
                        node_id = f"node_{node_counter}"
                        node_counter += 1
                    else:
                        if key in global_node_map:
                            node_id = global_node_map[key]
                        else:
                            node_id = f"node_{node_counter}"
                            global_node_map[key] = node_id
                            node_counter += 1
                        local_used.add(key)
                    node_path.append(node_id)
                    # Update node frequency count
                    node_frequency[node_id] = node_frequency.get(node_id, 0) + 1
                    # Add node to the graph if not already present with attribute 'word'
                    if not word_graph.has_node(node_id):
                        word_graph.add_node(node_id, word=token.text.lower())
                sentence_node_paths.append(node_path)

                # For each sentence, add directed edges: START -> first node, consecutive nodes, last node -> END
                if node_path:
                    # Edge from START to first node
                    if word_graph.has_edge("START", node_path[0]):
                        word_graph["START"][node_path[0]]["weight"] += 1.0
                    else:
                        word_graph.add_edge("START", node_path[0], weight=1.0)
                    # Consecutive edges within the sentence
                    for i in range(len(node_path) - 1):
                        u = node_path[i]
                        v = node_path[i + 1]
                        # Positional difference is 1 for consecutive words
                        edge_weight = (1.0 / node_frequency.get(u, 1)) * (1.0 / 1)
                        if word_graph.has_edge(u, v):
                            word_graph[u][v]["weight"] += edge_weight
                        else:
                            word_graph.add_edge(u, v, weight=edge_weight)
                    # Edge from last node to END
                    if word_graph.has_edge(node_path[-1], "END"):
                        word_graph[node_path[-1]]["END"]["weight"] += 1.0
                    else:
                        word_graph.add_edge(node_path[-1], "END", weight=1.0)

            # Use k-shortest paths (iterating over a limited number of candidate paths) from START to END
            candidate_scores: List[Tuple[float, List[str]]] = []
            try:
                path_generator = nx.shortest_simple_paths(word_graph, source="START", target="END", weight="weight")
                candidate_count: int = 0
                max_candidates: int = 10
                for candidate_path in path_generator:
                    candidate_count += 1
                    # Compute total edge weight
                    total_edge_weight: float = 0.0
                    for idx in range(len(candidate_path) - 1):
                        u = candidate_path[idx]
                        v = candidate_path[idx + 1]
                        total_edge_weight += word_graph[u][v]["weight"]
                    # Extract token sequence (skip START and END)
                    token_sequence: List[str] = []
                    for node_id in candidate_path:
                        if node_id not in ("START", "END"):
                            token_sequence.append(word_graph.nodes[node_id]["word"])
                    # Compute n-gram probability sum using trigram model
                    ngram_prob_sum: float = calculate_ngram_probability(token_sequence, n=3)
                    adjusted_score: float = total_edge_weight / (ngram_prob_sum + epsilon)
                    candidate_scores.append((adjusted_score, candidate_path))
                    if candidate_count >= max_candidates:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                self.logger.error("No path found in word graph for a cluster: %s", str(e))
                candidate_scores = []

            # Select candidate with lowest adjusted score
            best_candidate: List[str] = []
            if candidate_scores:
                best_candidate = min(candidate_scores, key=lambda x: x[0])[1]
            else:
                # Fallback: use the sentence corresponding to the minimum index in the cluster.
                best_candidate = ["START", f"node_{cluster[0]}", "END"]

            # Convert best candidate path to summary text (skip START and END)
            summary_tokens: List[str] = []
            for node_id in best_candidate:
                if node_id in ("START", "END"):
                    continue
                summary_tokens.append(word_graph.nodes[node_id]["word"])
            # Ensure minimum summary length
            if len(summary_tokens) < self.min_summary_length:
                # Fallback to the original sentence with minimum index in the cluster.
                summary_text = sentences[min(cluster)]
            else:
                summary_text = tokens_to_text(summary_tokens)
            cluster_summaries.append(summary_text)
            self.logger.debug("Cluster summary generated: %s", summary_text)

        self.logger.info("Generated summaries for %d clusters.", len(cluster_summaries))
        return cluster_summaries

    def run_pipeline(self, data: List[Dict[str, Any]]) -> str:
        """
        Run the full GLIMMER pipeline on the provided multi-document input.
        The process includes:
            1. Combining sentences and raw text from the input.
            2. Constructing the sentence graph.
            3. Identifying semantic clusters.
            4. Summarizing each cluster with a word graph.
            5. Ordering and assembling the final summary.
            6. Truncating the summary to the desired output length as per configuration.
        
        Args:
            data: List of document set dictionaries (each containing keys "doc_id", "sentences", and "raw_text").
        
        Returns:
            The final multi-document summary as a string.
        """
        all_sentences: List[str] = []
        combined_raw_text: str = ""

        for document in data:
            sentences = document.get("sentences", [])
            raw_text = document.get("raw_text", "")
            all_sentences.extend(sentences)
            combined_raw_text += " " + raw_text

        combined_raw_text = combined_raw_text.strip()
        if not all_sentences:
            self.logger.error("No sentences found in the input data.")
            return ""

        # Construct sentence graph
        sentence_graph = self.construct_sentence_graph(all_sentences)

        # Perform semantic clustering using TTR-based method
        clusters = self.semantic_cluster_identification(sentence_graph, all_sentences, combined_raw_text)

        # Generate summary for each cluster via word graph-based summarization
        cluster_summaries = self.cluster_summarization(clusters, all_sentences)

        # Order cluster summaries by the minimum sentence index in their corresponding cluster
        # (Assuming clusters were sorted in cluster_summarization, we simply join them now)
        final_summary: str = ". ".join(cluster_summaries).strip()

        # Truncate final summary to desired output length (in tokens)
        final_tokens: List[str] = nltk.word_tokenize(final_summary)
        if len(final_tokens) > self.output_length:
            final_tokens = final_tokens[:self.output_length]
            final_summary = tokens_to_text(final_tokens)
        self.logger.info("Final summary generated with %d tokens.", len(final_tokens))
        return final_summary
