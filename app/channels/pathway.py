# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Pathway enrichment channel - Phase 1B Production with ChannelScore format.
Uses local pathway data with enhanced scoring and evidence tracking.
"""
import json
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..schemas import ChannelScore, EvidenceRef, DataQualityFlags, get_utc_now
from ..validation import get_validator

logger = logging.getLogger(__name__)

# Configuration
PATHWAY_DATA_FILE = "data_demo/pathways.json"
REACTOME_VERSION = "2024"
MIN_PATHWAY_SIZE = 5
MAX_PATHWAY_SIZE = 500


class PathwayChannel:
    """
    Production pathway enrichment channel with structured scoring.
    """

    def __init__(self, pathway_file: str = PATHWAY_DATA_FILE):
        self.pathway_file = Path(pathway_file)
        self.validator = get_validator()
        self.channel_name = "pathway_enrichment"

        # Data structures
        self.pathways_data = {}  # target -> [pathways]
        self.pathway_to_targets = defaultdict(set)  # pathway -> {targets}

        self._load_pathways()

    def _load_pathways(self):
        """Load pathway annotations from JSON file."""
        try:
            if self.pathway_file.exists():
                with open(self.pathway_file, 'r') as f:
                    self.pathways_data = json.load(f)

                # Build reverse mapping: pathway -> set of targets
                for target, pathways in self.pathways_data.items():
                    for pathway in pathways:
                        self.pathway_to_targets[pathway].add(target)

                logger.info(
                    f"Loaded pathway data: {len(self.pathways_data)} targets, {len(self.pathway_to_targets)} pathways")
            else:
                logger.warning(f"Pathway file {self.pathway_file} not found, creating minimal data")
                self._create_minimal_pathway_data()

        except Exception as e:
            logger.error(f"Error loading pathway data: {e}")
            self._create_minimal_pathway_data()

    def _create_minimal_pathway_data(self):
        """Create minimal pathway data for core cancer genes."""
        self.pathways_data = {
            "EGFR": ["EGFR signaling", "RTK signaling", "Cell growth"],
            "ERBB2": ["ERBB signaling", "RTK signaling", "Cell proliferation"],
            "KRAS": ["RAS signaling", "MAPK cascade", "Cell proliferation"],
            "BRAF": ["MAPK cascade", "RAF signaling", "Cell growth"],
            "PIK3CA": ["PI3K-AKT signaling", "Cell survival", "mTOR signaling"],
            "PTEN": ["PI3K-AKT signaling", "Cell cycle regulation", "Apoptosis"],
            "TP53": ["p53 pathway", "DNA damage response", "Apoptosis", "Cell cycle"],
            "MET": ["MET signaling", "RTK signaling", "Cell migration"],
            "ALK": ["ALK signaling", "RTK signaling", "Cell growth"]
        }

        # Build reverse mapping
        self.pathway_to_targets = defaultdict(set)
        for target, pathways in self.pathways_data.items():
            for pathway in pathways:
                self.pathway_to_targets[pathway].add(target)

        logger.info("Created minimal pathway data")

    async def compute_score(self, gene: str, targets_context: Optional[List[str]] = None) -> ChannelScore:
        """
        Compute pathway enrichment score for a gene.

        Args:
            gene: Target gene symbol
            targets_context: Other targets in analysis for context enrichment

        Returns:
            ChannelScore with pathway enrichment metrics
        """
        evidence_refs = []
        components = {}
        quality_flags = DataQualityFlags()

        try:
            # Get pathways for target
            target_pathways = self.pathways_data.get(gene, [])

            if not target_pathways:
                logger.warning(f"No pathway data for {gene}")
                return ChannelScore(
                    name=self.channel_name,
                    score=None,
                    status="data_missing",
                    components={"pathway_count": 0},
                    evidence=[],
                    quality=DataQualityFlags(partial=True, notes="No pathway annotations found")
                )

            # Compute individual pathway specificity scores
            pathway_scores = []
            pathway_details = []

            for pathway in target_pathways:
                pathway_targets = self.pathway_to_targets.get(pathway, set())
                pathway_size = len(pathway_targets)

                # Skip overly large or small pathways
                if pathway_size < MIN_PATHWAY_SIZE:
                    quality_flags.partial = True
                    continue
                elif pathway_size > MAX_PATHWAY_SIZE:
                    continue

                # Specificity score (smaller pathways = more specific = higher score)
                specificity_score = 1.0 / np.log2(max(2, pathway_size))
                pathway_scores.append(specificity_score)

                pathway_details.append({
                    "name": pathway,
                    "size": pathway_size,
                    "specificity": specificity_score
                })

            if not pathway_scores:
                return ChannelScore(
                    name=self.channel_name,
                    score=None,
                    status="data_missing",
                    components={"pathway_count": len(target_pathways), "valid_pathways": 0},
                    evidence=[],
                    quality=DataQualityFlags(partial=True, notes="No valid pathways after filtering")
                )

            # Base enrichment score
            base_score = np.mean(pathway_scores)

            # Context enrichment bonus
            context_bonus = 0.0
            if targets_context and len(targets_context) > 1:
                context_bonus = self._compute_context_enrichment(gene, targets_context)

            final_score = min(1.0, base_score + context_bonus)

            # Build components
            components = {
                "pathway_count": len(target_pathways),
                "valid_pathways": len(pathway_scores),
                "base_score": base_score,
                "context_bonus": context_bonus,
                "final_score": final_score,
                "avg_pathway_size": np.mean([p["size"] for p in pathway_details])
            }

            # Build evidence references
            evidence_refs.append(EvidenceRef(
                source="reactome",
                title=f"Pathway annotations: {len(target_pathways)} pathways",
                url=f"https://reactome.org/content/query?q={gene}",
                source_quality="high",
                timestamp=get_utc_now()
            ))

            # Add top pathway details
            sorted_pathways = sorted(pathway_details, key=lambda x: x["specificity"], reverse=True)
            for pathway_info in sorted_pathways[:3]:
                evidence_refs.append(EvidenceRef(
                    source="reactome",
                    title=f"{pathway_info['name']} (size: {pathway_info['size']})",
                    url=f"https://reactome.org/content/query?q={pathway_info['name']}",
                    source_quality="medium",
                    timestamp=get_utc_now()
                ))

            logger.info(
                f"Pathway enrichment computed for {gene}",
                extra={
                    "gene": gene,
                    "score": final_score,
                    "pathway_count": len(target_pathways),
                    "valid_pathways": len(pathway_scores),
                    "context_bonus": context_bonus
                }
            )

            return ChannelScore(
                name=self.channel_name,
                score=final_score,
                status="ok",
                components=components,
                evidence=evidence_refs,
                quality=quality_flags
            )

        except Exception as e:
            logger.error(f"Pathway channel error for {gene}: {e}")

            return ChannelScore(
                name=self.channel_name,
                score=None,
                status="error",
                components={"error": str(e)[:100]},
                evidence=[],
                quality=DataQualityFlags(notes=f"Channel error: {str(e)[:100]}")
            )

    def _compute_context_enrichment(self, gene: str, targets_context: List[str]) -> float:
        """
        Compute pathway enrichment bonus based on target context.

        Args:
            gene: Target gene symbol
            targets_context: List of all targets in analysis

        Returns:
            Context enrichment bonus (0-0.2)
        """
        try:
            gene_pathways = set(self.pathways_data.get(gene, []))
            if not gene_pathways:
                return 0.0

            # Check pathway overlap with other targets
            overlap_scores = []

            for other_target in targets_context:
                if other_target == gene:
                    continue

                other_pathways = set(self.pathways_data.get(other_target, []))
                if other_pathways:
                    # Jaccard similarity
                    intersection = len(gene_pathways & other_pathways)
                    union = len(gene_pathways | other_pathways)

                    if union > 0:
                        jaccard = intersection / union
                        overlap_scores.append(jaccard)

            if overlap_scores:
                # Average overlap with context targets
                avg_overlap = np.mean(overlap_scores)
                # Convert to bonus (max 0.2)
                context_bonus = min(0.2, avg_overlap * 0.5)

                logger.debug(f"Context enrichment for {gene}: {context_bonus:.3f} (avg overlap: {avg_overlap:.3f})")
                return context_bonus

            return 0.0

        except Exception as e:
            logger.error(f"Context enrichment computation failed for {gene}: {e}")
            return 0.0

    def get_pathway_targets(self, pathway: str) -> Set[str]:
        """Get targets in a specific pathway."""
        return self.pathway_to_targets.get(pathway, set())

    def get_target_pathways(self, target: str) -> List[str]:
        """Get pathways for a specific target."""
        return self.pathways_data.get(target, [])


# ========================
# Global channel instance
# ========================

_pathway_channel: Optional[PathwayChannel] = None


async def get_pathway_channel() -> PathwayChannel:
    """Get global pathway channel instance."""
    global _pathway_channel
    if _pathway_channel is None:
        _pathway_channel = PathwayChannel()
    return _pathway_channel


# ========================
# Legacy compatibility functions
# ========================

async def compute_pathway_enrichment(target: str, targets_context: List[str] = None) -> Tuple[float, List[str]]:
    """
    Legacy compatibility wrapper for existing scoring.py integration.

    Args:
        target: Target gene symbol
        targets_context: Other targets in analysis for context

    Returns:
        (enrichment_score, evidence_references)
    """
    try:
        pathway_channel = await get_pathway_channel()
        channel_result = await pathway_channel.compute_score(target, targets_context)

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
            logger.warning(f"No pathway data for {target}")
            return 0.1, ["Status:data_missing", f"Reactome:{REACTOME_VERSION}"]

        else:  # error status
            logger.error(f"Pathway channel error for {target}")
            return 0.1, [f"Status:error"]

    except Exception as e:
        logger.error(f"Legacy pathway enrichment computation failed: {e}")
        return 0.1, [f"Error:{str(e)[:50]}"]


def compute_pathway_enrichment_batch(targets: List[str]) -> Dict[str, Tuple[float, List[str]]]:
    """
    Compute pathway enrichment scores for multiple targets (needs async refactor).
    """
    # TODO: Implement async batch processing
    results = {}
    for target in targets:
        # Placeholder - needs async implementation
        results[target] = (0.5, [f"Batch_placeholder:{target}"])
    return results


def get_pathway_summary(targets: List[str]) -> Dict[str, any]:
    """Get pathway analysis summary (needs async refactor)."""
    # TODO: Implement async version
    return {
        "note": "Pathway summary needs async refactor",
        "targets": len(targets)
    }


# Legacy global instances for compatibility
class LegacyPathwayAnalyzer:
    """Legacy analyzer for backward compatibility."""

    def __init__(self):
        self.pathways_data = {}
        self.pathway_to_targets = defaultdict(set)

    def get_target_pathways(self, target: str) -> List[str]:
        """Legacy method."""
        return self.pathways_data.get(target, [])

    def get_pathway_targets(self, pathway: str) -> Set[str]:
        """Legacy method."""
        return self.pathway_to_targets.get(pathway, set())

    def compute_pathway_enrichment(self, targets: List[str]) -> Dict[str, float]:
        """Legacy enrichment computation."""
        return {}  # Placeholder


pathway_analyzer = LegacyPathwayAnalyzer()