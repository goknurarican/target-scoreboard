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
import os
PPI_DISABLE_FALLBACK = os.getenv("PPI_DISABLE_FALLBACK", "true").lower() == "true"

logger = logging.getLogger(__name__)
# --- helpers for FE graph preview -------------------------------------------
from typing import Iterable, Tuple, Dict, Any, List, Optional

def _normalize_neighbor(n, target_symbol: str) -> Optional[Tuple[str, float]]:
    """
    Komşu girdilerini (str / tuple / dict) normalize edip (symbol, confidence) döndür.
    """
    try:
        if n is None:
            return None
        if isinstance(n, str):
            sym = n.strip().upper()
            if not sym or sym == target_symbol:
                return None
            return sym, 1.0
        if isinstance(n, (list, tuple)) and len(n) >= 1:
            sym = str(n[0]).strip().upper()
            conf = float(n[1]) if len(n) > 1 and n[1] is not None else 1.0
            if not sym or sym == target_symbol:
                return None
            return sym, max(0.0, min(1.0, conf))
        if isinstance(n, dict):
            sym = str(
                n.get("gene") or n.get("partner") or n.get("symbol") or
                n.get("id") or n.get("name") or ""
            ).strip().upper()
            conf = float(n.get("confidence") or n.get("score") or n.get("weight") or 1.0)
            if not sym or sym == target_symbol:
                return None
            return sym, max(0.0, min(1.0, conf))
    except Exception:
        return None
    return None


def _edge_iter(edge_candidates: Optional[Iterable[Any]]) -> List[Tuple[str, str, float]]:
    """
    Çeşitli edge şekillerini (model/dict) (source, target, confidence) üçlüsüne çevir.
    """
    if not edge_candidates:
        return []
    out = []
    for e in edge_candidates:
        try:
            if hasattr(e, "source_gene") and hasattr(e, "partner"):
                s = str(getattr(e, "source_gene")).upper()
                t = str(getattr(e, "partner")).upper()
                c = float(getattr(e, "confidence", 1.0) or 1.0)
            elif isinstance(e, dict):
                s = str(
                    e.get("source_gene") or e.get("source") or e.get("from") or
                    e.get("a") or e.get("u") or e.get("source_node") or ""
                ).upper()
                t = str(
                    e.get("partner") or e.get("target") or e.get("to") or
                    e.get("b") or e.get("v") or e.get("target_node") or ""
                ).upper()
                c = float(e.get("confidence") or e.get("score") or e.get("weight") or 1.0)
            else:
                continue
            if not s or not t or s == t:
                continue
            out.append((s, t, max(0.0, min(1.0, c))))
        except Exception:
            continue
    return out


def _build_graph_preview(
    target_symbol: str,
    neighbors: Iterable[Any],
    edge_candidates: Optional[Iterable[Any]] = None,
    max_nodes: int = 50,
    max_edges: int = 100,
) -> Dict[str, Any]:
    """
    FE'nin beklediği küçük graph objesini üret: {nodes: [{id}], links: [{source,target,confidence}]}
    """
    nodes: List[Dict[str, str]] = []
    links: List[Dict[str, Any]] = []

    if not target_symbol:
        return {"nodes": [], "links": []}

    target_symbol = target_symbol.upper()
    seen = set()

    def add_node(sym: str) -> bool:
        sym = sym.upper()
        if sym not in seen and len(nodes) < max_nodes:
            nodes.append({"id": sym})
            seen.add(sym)
            return True
        return False

    add_node(target_symbol)

    # 1-hop komşular
    neigh_pairs: List[Tuple[str, float]] = []
    for n in neighbors or []:
        nn = _normalize_neighbor(n, target_symbol)
        if not nn:
            continue
        sym, conf = nn
        if add_node(sym):
            neigh_pairs.append((sym, conf))

    # Aday edge'ler (varsa)
    edges = _edge_iter(edge_candidates)
    if edges:
        for s, t, c in edges:
            if s in seen and t in seen:
                links.append({"source": s, "target": t, "confidence": c})
                if len(links) >= max_edges:
                    break

    # Hiç edge yoksa yıldız (target ↔ komşu) fallback
    if not links:
        for sym, conf in neigh_pairs:
            links.append({"source": target_symbol, "target": sym, "confidence": conf})
            if len(links) >= max_edges:
                break

    return {"nodes": nodes, "links": links}


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
        """
        # DEBUG LOG EKLE
        logger.info(f"PPI compute_score: gene={gene}, disease_genes={disease_genes}")

        evidence_refs = []
        components = {}
        quality_flags = DataQualityFlags()

        try:
            # DEBUG LOG EKLE
            logger.info(f"Getting StringDB client for {gene}")

            # Fetch PPI network from STRING-DB
            stringdb_client = await get_stringdb_client()

            # DEBUG LOG EKLE
            logger.info(f"Calling fetch_ppi for {gene}")

            ppi_network = await stringdb_client.fetch_ppi(gene, limit=50)

            # DEBUG LOG EKLE
            logger.info(f"fetch_ppi returned network with {len(ppi_network.edges)} edges for {gene}")
            # Fetch PPI network from STRING-DB

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

            if PPI_DISABLE_FALLBACK:
                return ChannelScore(

                    name=self.channel_name,

                    score=None,

                    status="error",

                    components={"error": str(e)[:120]},

                    evidence=[],

                    quality=DataQualityFlags(partial=True, notes="STRING error; fallback disabled")

                )

            # (fallback açık ise - ama prod’da kapalı tutuyoruz)

            try:

                fallback_score = await self._compute_fallback_score(gene, disease_genes)

                logger.warning(f"Using fallback PPI score for {gene}: {fallback_score}")

                return ChannelScore(

                    name=self.channel_name,

                    score=fallback_score,

                    status="ok",

                    components={"method": "fallback", "network_size": 0},

                    evidence=[EvidenceRef(

                        source="fallback",

                        title="Fallback PPI score - STRING API unavailable",

                        timestamp=get_utc_now()

                    )],

                    quality=DataQualityFlags(partial=True, notes="Using fallback network")

                )

            except Exception as fallback_error:

                logger.error(f"Fallback also failed for {gene}: {fallback_error}")

                return ChannelScore(

                    name=self.channel_name,

                    score=None,

                    status="error",

                    components={"error": str(fallback_error)[:120]},

                    evidence=[],

                    quality=DataQualityFlags(notes="Both STRING and fallback failed")

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
            logger.info(f"RWR: target={target}, disease_genes={disease_genes}")

            if target not in graph:
                logger.warning(f"RWR: Target {target} not in graph")
                return 0.2

            # Prepare disease seed genes (present in network)
            network_genes = set(graph.nodes())
            logger.info(f"RWR: Network has {len(network_genes)} genes: {list(network_genes)[:10]}...")

            seed_genes = {g for g in disease_genes if g in network_genes and g != target}
            logger.info(f"RWR: Found {len(seed_genes)} seed genes in network: {seed_genes}")

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

            # NEW percentile-based normalization
            all_scores = prob_vector[prob_vector > 0]
            if len(all_scores) > 1:
                # Calculate target's percentile (0-1 range)
                target_percentile = np.sum(all_scores <= target_rwr_score) / len(all_scores)

                # More granular score mapping - use actual percentile value
                if target_percentile >= 0.95:  # Top 5%
                    normalized_score = 0.75 + (target_percentile - 0.95) * 3.0  # 0.75-0.9
                elif target_percentile >= 0.85:  # Top 15%
                    normalized_score = 0.65 + (target_percentile - 0.85) * 1.0  # 0.65-0.75
                elif target_percentile >= 0.70:  # Top 30%
                    normalized_score = 0.50 + (target_percentile - 0.70) * 1.0  # 0.50-0.65
                elif target_percentile >= 0.50:  # Top 50%
                    normalized_score = 0.35 + (target_percentile - 0.50) * 0.75  # 0.35-0.50
                else:  # Bottom 50%
                    normalized_score = 0.20 + target_percentile * 0.30  # 0.20-0.35

                logger.info(
                    f"RWR {target}: raw_score={target_rwr_score:.4f}, percentile={target_percentile:.3f}, final={normalized_score:.3f}")
            else:
                normalized_score = target_rwr_score
                logger.info(
                    f"RWR {target}: single_node, raw_score={target_rwr_score:.4f}, final={normalized_score:.3f}")

            return max(0.2, min(0.9, normalized_score))
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
            logger.info(f"Centrality {target}: degree={degree_cent:.3f}")

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

            if combined_score > 0.8:  # Çok yüksek centrality
                final_score = 0.7 + (combined_score - 0.8) * 0.5  # 0.7-0.9 arası
            elif combined_score > 0.5:  # Orta centrality
                final_score = 0.4 + (combined_score - 0.5) * 1.0  # 0.4-0.7 arası
            else:  # Düşük centrality
                final_score = 0.2 + combined_score * 0.4  # 0.2-0.4 arası

            final_score = max(0.2, min(0.9, final_score))  # 0.2-0.9 arası sınırla
            logger.info(
                f"Centrality {target}: degree={degree_cent:.3f}, betweenness={betweenness_cent:.3f}, closeness={closeness_cent:.3f}")
            logger.info(f"Centrality {target}: combined={combined_score:.3f}, final={final_score:.3f}")

            return final_score

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