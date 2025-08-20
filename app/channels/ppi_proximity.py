# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.


"""
PPI proximity channel with Random Walk with Restart (RWR) for network medicine.
"""
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PPINetwork:
    """PPI network with centrality and RWR analysis capabilities."""

    def __init__(self, edges_file: str = "data_demo/ppi_edges.tsv"):
        self.graph = nx.Graph()
        self.adjacency_matrix = None
        self.node_to_index = {}
        self.index_to_node = {}
        self.edges_file = edges_file
        self._load_network()

    def _load_network(self):
        """Load PPI network from TSV file or create demo network."""
        edges_path = Path(self.edges_file)

        if not edges_path.exists():
            logger.warning(f"PPI edges file not found: {edges_path}")
            self._create_demo_network()
            return

        try:
            with open(edges_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num == 0 and line.startswith('protein1'):
                        continue  # Skip header

                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        protein1, protein2 = parts[0], parts[1]
                        # Add edge with optional confidence score
                        confidence = float(parts[2]) if len(parts) > 2 else 1.0
                        self.graph.add_edge(protein1, protein2, weight=confidence)

            logger.info(
                f"Loaded PPI network: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        except Exception as e:
            logger.error(f"Error loading PPI network: {e}")
            self._create_demo_network()

        self._prepare_adjacency_matrix()

    def _create_demo_network(self):
        """Create a robust demo network for testing."""
        logger.info("Creating demo PPI network")
        self.graph = nx.Graph()

        # Create a comprehensive demo network with known cancer genes and their interactions
        demo_edges = [
            # EGFR signaling pathway
            ("EGFR", "ERBB2", 0.95), ("EGFR", "GRB2", 0.90), ("EGFR", "SOS1", 0.85),
            ("EGFR", "STAT3", 0.80), ("EGFR", "PIK3CA", 0.75),

            # ERBB2 interactions
            ("ERBB2", "GRB2", 0.85), ("ERBB2", "PIK3CA", 0.80), ("ERBB2", "STAT3", 0.75),

            # RAS/MAPK pathway
            ("KRAS", "RAF1", 0.95), ("KRAS", "PIK3CA", 0.85), ("KRAS", "SOS1", 0.80),
            ("RAF1", "MAP2K1", 0.90), ("MAP2K1", "MAPK1", 0.90), ("MAPK1", "JUN", 0.85),
            ("BRAF", "MAP2K1", 0.95), ("BRAF", "RAF1", 0.75),

            # PI3K/AKT pathway
            ("PIK3CA", "AKT1", 0.90), ("AKT1", "MTOR", 0.85), ("AKT1", "GSK3B", 0.80),
            ("PTEN", "AKT1", 0.85), ("PTEN", "PIK3CA", 0.75),

            # p53 pathway
            ("TP53", "MDM2", 0.95), ("TP53", "CDKN1A", 0.90), ("TP53", "ATM", 0.85),
            ("TP53", "BRCA1", 0.80), ("MDM2", "ATM", 0.75),

            # BRCA interactions
            ("BRCA1", "BRCA2", 0.90), ("BRCA1", "ATM", 0.85), ("BRCA2", "RAD51", 0.85),

            # MET pathway
            ("MET", "GRB2", 0.85), ("MET", "SOS1", 0.80), ("MET", "PIK3CA", 0.75),

            # ALK interactions
            ("ALK", "GRB2", 0.80), ("ALK", "STAT3", 0.85), ("ALK", "PIK3CA", 0.75),

            # Additional hub connections
            ("GRB2", "SOS1", 0.90), ("GRB2", "PIK3CA", 0.80),
            ("STAT3", "JUN", 0.75), ("STAT3", "MYC", 0.80),

            # Cell cycle
            ("RB1", "E2F1", 0.90), ("RB1", "CDKN1A", 0.75), ("E2F1", "MYC", 0.80),

            # VHL pathway
            ("VHL", "HIF1A", 0.95), ("HIF1A", "VEGFA", 0.85),

            # Additional cancer-relevant interactions
            ("MYC", "MAX", 0.90), ("JUN", "FOS", 0.85), ("VEGFA", "KDR", 0.90)
        ]

        for node1, node2, weight in demo_edges:
            self.graph.add_edge(node1, node2, weight=weight)

        logger.info(
            f"Created demo PPI network: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        self._prepare_adjacency_matrix()

    def _prepare_adjacency_matrix(self):
        """Prepare adjacency matrix for RWR calculations."""
        nodes = list(self.graph.nodes())
        self.node_to_index = {node: i for i, node in enumerate(nodes)}
        self.index_to_node = {i: node for i, node in enumerate(nodes)}

        n = len(nodes)
        if n == 0:
            return

        # Create weighted adjacency matrix
        adjacency = np.zeros((n, n))
        for edge in self.graph.edges(data=True):
            i = self.node_to_index[edge[0]]
            j = self.node_to_index[edge[1]]
            weight = edge[2].get('weight', 1.0)
            adjacency[i, j] = weight
            adjacency[j, i] = weight

        self.adjacency_matrix = csr_matrix(adjacency)

    def get_disease_seed_genes(self, disease_genes: List[str], n_seeds: int = 20) -> Set[str]:
        """
        Get seed genes for RWR from disease-associated genes.

        Args:
            disease_genes: List of disease-associated gene symbols
            n_seeds: Maximum number of seeds to use

        Returns:
            Set of seed gene symbols present in the network
        """
        # Find disease genes present in network
        network_genes = set(self.graph.nodes())
        seed_candidates = [gene for gene in disease_genes if gene in network_genes]

        if not seed_candidates:
            logger.warning("No disease genes found in network, using degree hubs as seeds")
            # Fall back to highest degree nodes
            degree_dict = dict(self.graph.degree())
            top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:n_seeds]
            return {node for node, _ in top_hubs}

        # Take top N seeds (prioritize by presence in network)
        seeds = set(seed_candidates[:n_seeds])
        logger.info(f"Using {len(seeds)} disease genes as RWR seeds: {list(seeds)[:5]}...")

        return seeds

    def random_walk_with_restart(self, seed_genes: Set[str], restart_prob: float = 0.3,
                                 max_iter: int = 100, tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Perform Random Walk with Restart (RWR) on the network.

        Args:
            seed_genes: Set of seed gene symbols to start random walk
            restart_prob: Probability of restarting to seed nodes
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Dict mapping gene symbols to steady-state probabilities
        """
        if self.adjacency_matrix is None or len(seed_genes) == 0:
            return {}

        n = self.adjacency_matrix.shape[0]

        # Create seed vector
        seed_vector = np.zeros(n)
        seed_indices = []
        for gene in seed_genes:
            if gene in self.node_to_index:
                idx = self.node_to_index[gene]
                seed_indices.append(idx)
                seed_vector[idx] = 1.0

        if len(seed_indices) == 0:
            logger.warning("No seed genes found in network")
            return {}

        # Normalize seed vector
        seed_vector = seed_vector / np.sum(seed_vector)

        # Create column-normalized adjacency matrix (transition matrix)
        adjacency = self.adjacency_matrix.toarray()
        column_sums = np.sum(adjacency, axis=0)
        # Avoid division by zero
        column_sums[column_sums == 0] = 1.0
        transition_matrix = adjacency / column_sums[np.newaxis, :]

        # Initialize probability vector
        prob_vector = seed_vector.copy()

        # Iterative random walk
        for iteration in range(max_iter):
            new_prob = (1 - restart_prob) * transition_matrix.dot(prob_vector) + restart_prob * seed_vector

            # Check convergence
            diff = norm(new_prob - prob_vector, ord=1)
            if diff < tolerance:
                logger.info(f"RWR converged after {iteration + 1} iterations (diff: {diff:.2e})")
                break

            prob_vector = new_prob
        else:
            logger.warning(f"RWR did not converge after {max_iter} iterations")

        # Convert back to gene symbol mapping
        result = {}
        for i, prob in enumerate(prob_vector):
            gene = self.index_to_node[i]
            result[gene] = float(prob)

        return result

    def compute_centrality_scores(self, target: str) -> Dict[str, float]:
        """Compute various centrality scores for a target."""
        if target not in self.graph:
            return {"degree": 0.0, "betweenness": 0.0, "closeness": 0.0}

        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph).get(target, 0.0)

        # Betweenness centrality (expensive for large graphs)
        if self.graph.number_of_nodes() < 1000:
            betweenness_cent = nx.betweenness_centrality(
                self.graph,
                k=min(100, self.graph.number_of_nodes())
            ).get(target, 0.0)
        else:
            betweenness_cent = 0.0

        # Closeness centrality
        if nx.is_connected(self.graph):
            closeness_cent = nx.closeness_centrality(self.graph).get(target, 0.0)
        else:
            # For disconnected graphs, compute on the component containing the target
            if target in self.graph:
                component = nx.node_connected_component(self.graph, target)
                subgraph = self.graph.subgraph(component)
                closeness_cent = nx.closeness_centrality(subgraph).get(target, 0.0)
            else:
                closeness_cent = 0.0

        return {
            "degree": degree_cent,
            "betweenness": betweenness_cent,
            "closeness": closeness_cent
        }


# Global network instance
ppi_network = PPINetwork()


def compute_ppi_proximity(target: str, disease_genes: Optional[List[str]] = None,
                          rwr_enabled: bool = True) -> Tuple[float, List[str]]:
    """
    Compute PPI proximity score using centrality measures or RWR.

    Args:
        target: Target gene symbol
        disease_genes: List of disease-associated genes for RWR seeding
        rwr_enabled: Whether to use RWR (True) or fall back to centrality (False)

    Returns:
        (proximity_score, evidence_references)
    """
    evidence_refs = []
    evidence_refs.append("STRING:v12.0")

    # Check if target is in network
    if target not in ppi_network.graph:
        logger.warning(f"Target {target} not found in PPI network")
        evidence_refs.append(f"PPI_status:not_in_network")
        return 0.2, evidence_refs

    if rwr_enabled and disease_genes:
        try:
            # Use RWR approach
            seed_genes = ppi_network.get_disease_seed_genes(disease_genes, n_seeds=20)

            if seed_genes:
                rwr_scores = ppi_network.random_walk_with_restart(seed_genes)

                if rwr_scores:
                    # Get target's RWR score
                    target_rwr_score = rwr_scores.get(target, 0.0)

                    # Min-max normalize across all scores
                    all_scores = list(rwr_scores.values())
                    if len(all_scores) > 1:
                        min_score = min(all_scores)
                        max_score = max(all_scores)
                        if max_score > min_score:
                            normalized_score = (target_rwr_score - min_score) / (max_score - min_score)
                        else:
                            normalized_score = 0.5
                    else:
                        normalized_score = target_rwr_score

                    # Ensure reasonable bounds
                    normalized_score = max(0.1, min(1.0, normalized_score))

                    evidence_refs.append(f"RWR_score:{target_rwr_score:.4f}")
                    evidence_refs.append(f"RWR_seeds:{len(seed_genes)}")
                    evidence_refs.append("RWR:seeded_by_disease_genes")

                    logger.info(f"RWR score for {target}: {normalized_score:.3f} (raw: {target_rwr_score:.4f})")
                    return normalized_score, evidence_refs

        except Exception as e:
            logger.error(f"RWR computation failed for {target}: {e}")
            evidence_refs.append(f"RWR_error:{str(e)[:50]}")

    # Fall back to centrality-based scoring
    logger.info(f"Using centrality-based scoring for {target}")
    centrality_scores = ppi_network.compute_centrality_scores(target)

    # Combine centrality measures (weighted average)
    combined_score = (
            0.5 * centrality_scores["degree"] +
            0.3 * centrality_scores["betweenness"] +
            0.2 * centrality_scores["closeness"]
    )

    # Ensure minimum score
    combined_score = max(0.1, combined_score)

    evidence_refs.append(f"Centrality_degree:{centrality_scores['degree']:.3f}")
    evidence_refs.append(f"Centrality_betweenness:{centrality_scores['betweenness']:.3f}")
    evidence_refs.append(f"Centrality_closeness:{centrality_scores['closeness']:.3f}")
    evidence_refs.append("Centrality_method:fallback")

    logger.info(f"Centrality score for {target}: {combined_score:.3f}")
    return combined_score, evidence_refs


def get_ppi_network_stats() -> Dict:
    """Get PPI network statistics."""
    return {
        "nodes": ppi_network.graph.number_of_nodes(),
        "edges": ppi_network.graph.number_of_edges(),
        "density": nx.density(ppi_network.graph),
        "connected_components": nx.number_connected_components(ppi_network.graph),
        "largest_component_size": len(
            max(nx.connected_components(ppi_network.graph), key=len)
        ) if ppi_network.graph.number_of_nodes() > 0 else 0
    }


def get_network_neighbors(target: str, max_neighbors: int = 10) -> List[str]:
    """Get network neighbors of a target gene."""
    if target not in ppi_network.graph:
        return []

    neighbors = list(ppi_network.graph.neighbors(target))
    return neighbors[:max_neighbors]