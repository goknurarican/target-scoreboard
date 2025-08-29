# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
PPI proximity channel - Phase 1B Production with STRING-DB integration.
Real protein-protein interaction data with RWR and centrality analysis.
"""
import asyncio
import logging
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from typing import Dict, List, Tuple, Optional, Set

from ..schemas import ChannelScore, EvidenceRef, DataQualityFlags, PPINetwork, get_utc_now
from ..validation import get_validator
from ..data_access.stringdb import get_stringdb_client

logger = logging.getLogger(__name__)


class PPIProximityChannel:
    """
    Production PPI proximity channel using real STRING-DB data.
    """

    def __init__(self):
        self.validator = get_validator()
        self.channel_name = "ppi"
        self._fallback_network = None

    async def compute_score(self, gene: str, disease_genes: Optional[List[str]] = None) -> ChannelScore:
        """
        Compute PPI proximity score using STRING-DB data + network analysis.

        Args:
            gene: Target gene symbol
            disease_genes: Disease-associated genes for RWR seeding

        Returns:
            ChannelScore with proximity metrics and evidence
        """
        evidence_refs = []
        components = {}
        quality_flags = DataQualityFlags()

        try:
            # Fetch PPI network from STRING-DB
            stringdb_client = await get_stringdb_client()
            ppi_network = await stringdb_client.fetch_ppi(gene, limit=50)

            # Validate network quality
            network_quality = await stringdb_client.validate_network_quality(ppi_network)
            quality_flags = network_quality

            # Check if network is usable
            if len(ppi_network.edges) == 0:
                logger.warning(f"Empty PPI network for {gene}")
                return ChannelScore(
                    name=self.channel_name,
                    score=None,
                    status="data_missing",
                    components={"network_size": 0},
                    evidence=[],
                    quality=DataQualityFlags(partial=True, notes="No PPI interactions found")
                )

            # Build NetworkX graph from STRING data
            nx_graph = self._build_networkx_graph(ppi_network)

            # Compute proximity score
            if disease_genes and len(disease_genes) > 1:
                # Use RWR if we have disease context
                proximity_score = await self._compute_rwr_score(gene, disease_genes, nx_graph)
                method = "rwr"
            else:
                # Fall back to centrality analysis
                proximity_score = self._compute_centrality_score(gene, nx_graph)
                method = "centrality"

            # Build components
            components = {
               "network_size": float(len(ppi_network.edges)),
                "avg_confidence": float(
                    sum(edge.confidence for edge in ppi_network.edges) / len(ppi_network.edges))
                                                   }
            # metinleri kalite notlarına ekle
            quality_flags.notes = f"method={method}; string_version={ppi_network.source}"

            # Add centrality metrics as components
            centrality_scores = self._compute_centrality_metrics(gene, nx_graph)
            components.update(centrality_scores)

            # Convert STRING evidence to EvidenceRef
            for edge in ppi_network.edges[:5]:  # Include top 5 edges as evidence
                for evidence in edge.evidence:
                    evidence_refs.append(evidence)

            # Add summary evidence
            string_evidence = EvidenceRef(
                source="stringdb",
                title=f"STRING PPI network: {len(ppi_network.edges)} interactions",
                url=f"https://string-db.org/network/{gene}",
                source_quality="high",
                timestamp=get_utc_now()
            )
            evidence_refs.append(string_evidence)

            logger.info(
                f"PPI proximity computed for {gene}",
                extra={
                    "gene": gene,
                    "score": proximity_score,
                    "network_size": len(ppi_network.edges),
                    "method": method,
                    "avg_confidence": components["avg_confidence"]
                }
            )

            return ChannelScore(
                name=self.channel_name,
                score=proximity_score,
                status="ok",
                components=components,
                evidence=evidence_refs,
                quality=quality_flags
            )

        except Exception as e:
            logger.error(f"PPI proximity error for {gene}: {e}")

            # Try fallback network if available
            return ChannelScore(
                               name=self.channel_name, score = None, status = "error",
                          components = {}, evidence = [],
                           quality = DataQualityFlags(notes=f"Channel error: {str(e)[:100]}")
                                                   )

    def _build_networkx_graph(self, ppi_network: PPINetwork) -> nx.Graph:
        """Convert PPINetwork to NetworkX graph."""
        graph = nx.Graph()

        for edge in ppi_network.edges:
            graph.add_edge(
                edge.source_gene,
                edge.partner,
                weight=edge.confidence,
                confidence=edge.confidence
            )

        return graph

    async def _compute_rwr_score(self, target: str, disease_genes: List[str], graph: nx.Graph) -> float:
        """Compute Random Walk with Restart score."""
        try:
            if target not in graph:
                return 0.2

            # Prepare disease seed genes (present in network)
            network_genes = set(graph.nodes())
            seed_genes = set(gene for gene in disease_genes if gene in network_genes)

            if not seed_genes:
                logger.warning("No disease genes found in network, using centrality fallback")
                return self._compute_centrality_score(target, graph)

            # Prepare adjacency matrix
            nodes = list(graph.nodes())
            node_to_index = {node: i for i, node in enumerate(nodes)}
            n = len(nodes)

            if n == 0:
                return 0.2

            # Build adjacency matrix
            adjacency = np.zeros((n, n))
            for edge in graph.edges(data=True):
                i = node_to_index[edge[0]]
                j = node_to_index[edge[1]]
                weight = edge[2].get('confidence', 1.0)
                adjacency[i, j] = weight
                adjacency[j, i] = weight

            # Create seed vector
            seed_vector = np.zeros(n)
            for gene in seed_genes:
                if gene in node_to_index:
                    seed_vector[node_to_index[gene]] = 1.0

            if np.sum(seed_vector) == 0:
                return 0.2

            seed_vector = seed_vector / np.sum(seed_vector)

            # Create transition matrix
            column_sums = np.sum(adjacency, axis=0)
            column_sums[column_sums == 0] = 1.0
            transition_matrix = adjacency / column_sums[np.newaxis, :]

            # RWR iteration
            restart_prob = 0.3
            max_iter = 100
            tolerance = 1e-6
            prob_vector = seed_vector.copy()

            for iteration in range(max_iter):
                new_prob = (1 - restart_prob) * transition_matrix.dot(prob_vector) + restart_prob * seed_vector
                diff = np.linalg.norm(new_prob - prob_vector, ord=1)

                if diff < tolerance:
                    break
                prob_vector = new_prob

            # Get target's RWR score
            if target not in node_to_index:
                return 0.2

            target_rwr_score = prob_vector[node_to_index[target]]

            # Normalize against all scores
            all_scores = prob_vector[prob_vector > 0]
            if len(all_scores) > 1:
                min_score = np.min(all_scores)
                max_score = np.max(all_scores)
                if max_score > min_score:
                    normalized_score = (target_rwr_score - min_score) / (max_score - min_score)
                else:
                    normalized_score = 0.5
            else:
                normalized_score = target_rwr_score

            return max(0.1, min(1.0, normalized_score))

        except Exception as e:
            logger.error(f"RWR computation failed for {target}: {e}")
            return self._compute_centrality_score(target, graph)

    def _compute_centrality_score(self, target: str, graph: nx.Graph) -> float:
        """Compute centrality-based proximity score."""
        if target not in graph:
            return 0.2

        try:
            # Degree centrality
            degree_cent = nx.degree_centrality(graph).get(target, 0.0)

            # Betweenness centrality (for smaller networks)
            betweenness_cent = 0.0
            if graph.number_of_nodes() < 500:
                betweenness_cent = nx.betweenness_centrality(
                    graph, k=min(100, graph.number_of_nodes())
                ).get(target, 0.0)

            # Closeness centrality
            closeness_cent = 0.0
            if nx.is_connected(graph):
                closeness_cent = nx.closeness_centrality(graph).get(target, 0.0)
            else:
                # Compute on target's component
                if target in graph:
                    component = nx.node_connected_component(graph, target)
                    subgraph = graph.subgraph(component)
                    closeness_cent = nx.closeness_centrality(subgraph).get(target, 0.0)

            # Weighted combination
            combined_score = (
                    0.5 * degree_cent +
                    0.3 * betweenness_cent +
                    0.2 * closeness_cent
            )

            return max(0.1, min(1.0, combined_score))

        except Exception as e:
            logger.error(f"Centrality computation failed for {target}: {e}")
            return 0.2

    def _compute_centrality_metrics(self, target: str, graph: nx.Graph) -> Dict[str, float]:
        """Compute detailed centrality metrics for components."""
        if target not in graph:
            return {"degree": 0.0, "betweenness": 0.0, "closeness": 0.0}

        try:
            metrics = {}

            # Degree centrality
            metrics["degree"] = nx.degree_centrality(graph).get(target, 0.0)

            # Betweenness (for reasonable sized networks)
            if graph.number_of_nodes() < 500:
                metrics["betweenness"] = nx.betweenness_centrality(
                    graph, k=min(100, graph.number_of_nodes())
                ).get(target, 0.0)
            else:
                metrics["betweenness"] = 0.0

            # Closeness
            if nx.is_connected(graph):
                metrics["closeness"] = nx.closeness_centrality(graph).get(target, 0.0)
            else:
                if target in graph:
                    component = nx.node_connected_component(graph, target)
                    subgraph = graph.subgraph(component)
                    metrics["closeness"] = nx.closeness_centrality(subgraph).get(target, 0.0)
                else:
                    metrics["closeness"] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Centrality metrics failed for {target}: {e}")
            return {"degree": 0.0, "betweenness": 0.0, "closeness": 0.0}

    async def _compute_fallback_score(self, target: str, disease_genes: Optional[List[str]] = None) -> float:
        """Compute score using fallback demo network."""
        if self._fallback_network is None:
            self._fallback_network = self._create_demo_network()

        return self._compute_centrality_score(target, self._fallback_network)

    def _create_demo_network(self) -> nx.Graph:
        """Create demo network as fallback."""
        graph = nx.Graph()

        # Known cancer gene interactions
        demo_edges = [
            ("EGFR", "ERBB2", 0.95), ("EGFR", "GRB2", 0.90), ("EGFR", "SOS1", 0.85),
            ("KRAS", "RAF1", 0.95), ("KRAS", "PIK3CA", 0.85), ("BRAF", "MAP2K1", 0.95),
            ("PIK3CA", "AKT1", 0.90), ("PTEN", "AKT1", 0.85), ("TP53", "MDM2", 0.95),
            ("MET", "GRB2", 0.85), ("ALK", "STAT3", 0.85)
        ]

        for node1, node2, weight in demo_edges:
            graph.add_edge(node1, node2, weight=weight)

        return graph


# ========================
# Global channel instance
# ========================

_ppi_channel: Optional[PPIProximityChannel] = None


async def get_ppi_channel() -> PPIProximityChannel:
    """Get global PPI proximity channel instance."""
    global _ppi_channel
    if _ppi_channel is None:
        _ppi_channel = PPIProximityChannel()
    return _ppi_channel


# ========================
# Legacy compatibility functions
# ========================

async def compute_ppi_proximity(target: str, disease_genes: Optional[List[str]] = None, rwr_enabled: bool = True) -> \
Tuple[float, List[str]]:
    """
    Legacy compatibility wrapper for existing scoring.py integration.

    Args:
        target: Target gene symbol
        disease_genes: Disease-associated genes for RWR
        rwr_enabled: Whether to use RWR (always True in production)

    Returns:
        (proximity_score, evidence_references) - compatible with existing code
    """
    try:
        ppi_channel = await get_ppi_channel()
        channel_result = await ppi_channel.compute_score(target, disease_genes)

        # Convert to legacy format
        if channel_result.status == "ok" and channel_result.score is not None:
            score = channel_result.score

            # Convert evidence refs to legacy string format
            evidence_strings = []
            for evidence in channel_result.evidence:
                evidence_strings.append(f"Source:{evidence.source}")
                if evidence.title:
                    evidence_strings.append(f"Evidence:{evidence.title[:50]}")

            # Add component info
            for comp_name, comp_value in channel_result.components.items():
                if isinstance(comp_value, (int, float)):
                    evidence_strings.append(f"{comp_name}:{comp_value:.3f}")
                else:
                    evidence_strings.append(f"{comp_name}:{comp_value}")

            return score, evidence_strings

        elif channel_result.status == "data_missing":
            logger.warning(f"No PPI data for {target}")
            return 0.2, ["Status:data_missing", "Network:empty"]

        else:  # error status
            logger.error(f"PPI channel error for {target}")
            return 0.2, [f"Status:error"]

    except Exception as e:
        logger.error(f"Legacy PPI score computation failed: {e}")
        return 0.2, [f"Error:{str(e)[:50]}"]


def get_ppi_network_stats() -> Dict:
    """Get PPI network statistics (placeholder for legacy compatibility)."""
    return {
        "notes": "Network stats now computed per-request from STRING-DB",
        "source": "STRING-DB API",
        "version": "12.0"
    }


def get_network_neighbors(target: str, max_neighbors: int = 10) -> List[str]:
    """Get network neighbors (placeholder - needs async refactor)."""
    return []  # TODO: Implement async version