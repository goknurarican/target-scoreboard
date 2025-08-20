# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.


import json
from pathlib import Path
from typing import Tuple, List, Dict, Set
import numpy as np
from collections import defaultdict


class PathwayAnalyzer:
    def __init__(self, pathway_file: str = "data_demo/pathways.json"):
        self.pathway_file = Path(pathway_file)
        self.pathways_data = {}
        self.pathway_to_targets = defaultdict(set)
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

                print(
                    f"Loaded pathway data: {len(self.pathways_data)} targets, {len(self.pathway_to_targets)} pathways")
            else:
                print(f"Pathway file {self.pathway_file} not found, using empty data")

        except Exception as e:
            print(f"Error loading pathway data: {e}")
            self.pathways_data = {}
            self.pathway_to_targets = defaultdict(set)

    def get_target_pathways(self, target: str) -> List[str]:
        """Get list of pathways for a target."""
        return self.pathways_data.get(target, [])

    def get_pathway_targets(self, pathway: str) -> Set[str]:
        """Get set of targets in a pathway."""
        return self.pathway_to_targets.get(pathway, set())

    def compute_pathway_enrichment(self, targets: List[str]) -> Dict[str, float]:
        """Compute pathway enrichment scores for a set of targets."""
        if not targets:
            return {}

        # Get all pathways mentioned by the target set
        all_pathways = set()
        target_pathway_map = {}

        for target in targets:
            target_pathways = self.get_target_pathways(target)
            target_pathway_map[target] = set(target_pathways)
            all_pathways.update(target_pathways)

        # Calculate enrichment for each pathway
        pathway_scores = {}

        for pathway in all_pathways:
            # How many of our targets are in this pathway?
            targets_in_pathway = sum(1 for t in targets if pathway in target_pathway_map.get(t, set()))

            # Total targets in this pathway (from all data)
            total_targets_in_pathway = len(self.get_pathway_targets(pathway))

            # Enrichment score: fraction of our targets in this pathway
            if total_targets_in_pathway > 0:
                enrichment = targets_in_pathway / len(targets)
                # Weight by pathway size (smaller pathways get higher scores)
                size_weight = 1.0 / np.log2(max(2, total_targets_in_pathway))
                pathway_scores[pathway] = enrichment * size_weight
            else:
                pathway_scores[pathway] = 0.0

        return pathway_scores


# Global pathway analyzer instance
pathway_analyzer = PathwayAnalyzer()


def compute_pathway_enrichment(target: str, targets_context: List[str] = None) -> Tuple[float, List[str]]:
    """
    Compute pathway enrichment score for a single target.

    Args:
        target: Target gene symbol
        targets_context: Other targets in the same analysis (for context)

    Returns:
        (score, evidence_refs): Score in 0..1 range and list of evidence references
    """

    try:
        # Get pathways for this target
        target_pathways = pathway_analyzer.get_target_pathways(target)

        if not target_pathways:
            return 0.1, [f"Reactome:2024 (no_pathways_{target})"]

        # Simple scoring: number of pathways normalized by max pathways seen
        # This is a toy implementation - real version would use statistical enrichment

        # Get pathway sizes and compute a basic score
        pathway_scores = []
        evidence_refs = ["Reactome:2024 (demo)"]

        for pathway in target_pathways:
            pathway_targets = pathway_analyzer.get_pathway_targets(pathway)
            pathway_size = len(pathway_targets)

            # Smaller, more specific pathways get higher scores
            if pathway_size > 0:
                specificity_score = 1.0 / np.log2(max(2, pathway_size))
                pathway_scores.append(specificity_score)

                # Add pathway to evidence
                evidence_refs.append(f"{pathway}:size_{pathway_size}")

        if pathway_scores:
            # Average of pathway specificity scores
            final_score = np.mean(pathway_scores)
            # Normalize to 0-1 range (assuming max specificity score ~1.0)
            normalized_score = max(0.0, min(1.0, final_score))
        else:
            normalized_score = 0.1

        # If we have context targets, check for pathway overlap
        if targets_context and len(targets_context) > 1:
            enrichment_scores = pathway_analyzer.compute_pathway_enrichment(targets_context)

            # Boost score if target's pathways are enriched in the context
            target_pathway_set = set(target_pathways)
            context_boost = 0.0

            for pathway, enrich_score in enrichment_scores.items():
                if pathway in target_pathway_set:
                    context_boost += enrich_score

            if context_boost > 0:
                # Add up to 20% boost for pathway enrichment
                boost_factor = min(0.2, context_boost / len(target_pathway_set))
                normalized_score = min(1.0, normalized_score + boost_factor)
                evidence_refs.append(f"Context_enrichment:{context_boost:.3f}")

        return normalized_score, evidence_refs

    except Exception as e:
        print(f"Error computing pathway enrichment for {target}: {e}")
        return 0.1, [f"Reactome:error_{target}"]


def compute_pathway_enrichment_batch(targets: List[str]) -> Dict[str, Tuple[float, List[str]]]:
    """
    Compute pathway enrichment scores for multiple targets at once.

    Args:
        targets: List of target gene symbols

    Returns:
        Dictionary mapping target -> (score, evidence_refs)
    """
    results = {}

    # Compute pathway enrichment for the entire set
    if targets:
        enrichment_scores = pathway_analyzer.compute_pathway_enrichment(targets)

        # Get individual scores with context
        for target in targets:
            score, refs = compute_pathway_enrichment(target, targets_context=targets)
            results[target] = (score, refs)
    else:
        # Empty target list
        for target in targets:
            results[target] = (0.0, ["Reactome:2024 (empty_target_list)"])

    return results


def get_pathway_summary(targets: List[str]) -> Dict[str, any]:
    """
    Get a summary of pathway analysis for a set of targets.

    Returns:
        Dictionary with pathway statistics and top enriched pathways
    """
    if not targets:
        return {"error": "No targets provided"}

    try:
        # Get enrichment scores
        enrichment_scores = pathway_analyzer.compute_pathway_enrichment(targets)

        # Get all pathways mentioned by targets
        all_target_pathways = set()
        for target in targets:
            target_pathways = pathway_analyzer.get_target_pathways(target)
            all_target_pathways.update(target_pathways)

        # Sort pathways by enrichment score
        sorted_pathways = sorted(
            enrichment_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "total_pathways": len(all_target_pathways),
            "enriched_pathways": len([s for s in enrichment_scores.values() if s > 0.1]),
            "top_pathways": sorted_pathways[:5],  # Top 5
            "pathway_coverage": {
                target: len(pathway_analyzer.get_target_pathways(target))
                for target in targets
            }
        }

    except Exception as e:
        return {"error": f"Error in pathway summary: {e}"}