# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.


"""
Genetics channel for target scoring.
Uses Open Targets association scores for genetic evidence.
"""
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def compute_genetics_score(disease: str, target: str, ot_data: dict) -> Tuple[float, List[str]]:
    """
    Compute genetics association score from Open Targets data with enhanced fallbacks.

    Args:
        disease: Disease EFO identifier
        target: Target gene symbol
        ot_data: Open Targets API response data

    Returns:
        (genetics_score, evidence_references)
    """
    try:
        evidence_refs = []

        # Check if we have valid OT data
        if not ot_data or ot_data.get("error") or "genetics" not in ot_data:
            logger.info(f"Using demo genetics scoring for {target}")
            return _get_demo_genetics_score(target, evidence_refs)

        # Extract genetics score from OT data
        genetics_score = ot_data.get("genetics", 0.0)
        overall_score = ot_data.get("overall", 0.0)
        evidence_count = ot_data.get("evidence_count", 0)
        release = ot_data.get("release", "2024.06")

        # Normalize to 0-1 range and ensure minimum score
        normalized_score = max(0.1, min(1.0, genetics_score))

        # Build evidence references
        evidence_refs.append(f"OpenTargets:{release}")
        evidence_refs.append(f"OT_genetics:{genetics_score:.3f}")

        if overall_score > 0.5:
            evidence_refs.append(f"OT_overall:{overall_score:.3f}")

        if evidence_count > 0:
            evidence_refs.append(f"Evidence_count:{evidence_count}")

        # Add confidence level based on evidence
        if evidence_count >= 50:
            evidence_refs.append("Confidence:high")
        elif evidence_count >= 10:
            evidence_refs.append("Confidence:medium")
        else:
            evidence_refs.append("Confidence:low")

        logger.info(f"OT genetics score for {target}: {normalized_score:.3f} (evidence: {evidence_count})")
        return normalized_score, evidence_refs

    except Exception as e:
        logger.error(f"Error computing genetics score for {target}: {e}")
        evidence_refs = [f"Genetics_error:{str(e)[:50]}"]
        return _get_demo_genetics_score(target, evidence_refs)


def _get_demo_genetics_score(target: str, evidence_refs: List[str]) -> Tuple[float, List[str]]:
    """
    Generate realistic demo genetics scores based on known target biology.

    Args:
        target: Target gene symbol
        evidence_refs: Evidence references list to append to

    Returns:
        (demo_score, updated_evidence_refs)
    """
    # Enhanced demo scores based on real genetic associations
    demo_scores = {
        # Strong genetic associations (oncogenes with driver mutations)
        "EGFR": 0.89,  # Strong NSCLC driver
        "ALK": 0.94,   # Strong fusion driver
        "ERBB2": 0.82, # HER2 amplifications
        "BRAF": 0.85,  # V600E mutations
        "KIT": 0.88,   # GIST driver
        "ABL1": 0.91,  # BCR-ABL fusions

        # Moderate genetic associations
        "MET": 0.71,   # Amplifications and mutations
        "KRAS": 0.78,  # Common but complex
        "PIK3CA": 0.76, # Hotspot mutations
        "PTEN": 0.69,  # Loss of function
        "RET": 0.83,   # Fusion driver

        # Tumor suppressors (strong but different pattern)
        "TP53": 0.92,  # Most commonly mutated
        "RB1": 0.79,   # Retinoblastoma gene
        "BRCA1": 0.88, # Hereditary breast cancer
        "BRCA2": 0.87, # Hereditary breast cancer
        "VHL": 0.86,   # Von Hippel-Lindau
        "APC": 0.84,   # Colorectal cancer

        # Moderate associations
        "PDGFRA": 0.68,
        "FLT3": 0.74,
        "IDH1": 0.71,
        "IDH2": 0.69,
        "NRAS": 0.65,

        # Lower but significant
        "SRC": 0.58,
        "JAK2": 0.72,
        "STAT3": 0.45,
        "MYC": 0.63,
        "CCND1": 0.57
    }

    # Get score with some variability for unlisted targets
    if target in demo_scores:
        base_score = demo_scores[target]
        evidence_refs.append(f"Demo_genetics:{base_score:.2f}")
        evidence_refs.append("Known_cancer_gene:curated")
    else:
        # Generate score based on target characteristics
        base_score = _estimate_genetics_score_by_name(target)
        evidence_refs.append(f"Demo_genetics:{base_score:.2f}")
        evidence_refs.append("Estimated_score:heuristic")

    evidence_refs.append("Data_source:demo_fallback")

    return base_score, evidence_refs


def _estimate_genetics_score_by_name(target: str) -> float:
    """
    Estimate genetics score based on gene name patterns and known biology.

    Args:
        target: Target gene symbol

    Returns:
        Estimated genetics score (0.1-0.8)
    """
    target_upper = target.upper()

    # Oncogene patterns
    if any(pattern in target_upper for pattern in ['ERB', 'EGFR', 'ALK', 'RET', 'MET']):
        return 0.75

    # Kinase patterns
    if any(pattern in target_upper for pattern in ['KIN', 'CDK', 'PLK', 'AUR']):
        return 0.65

    # Tumor suppressor patterns
    if any(pattern in target_upper for pattern in ['P53', 'RB', 'PTEN', 'VHL']):
        return 0.80

    # Transcription factor patterns
    if any(pattern in target_upper for pattern in ['MYC', 'FOS', 'JUN', 'E2F']):
        return 0.55

    # Metabolic enzyme patterns
    if any(pattern in target_upper for pattern in ['IDH', 'SDH', 'FH']):
        return 0.60

    # Default for unknown genes
    return 0.35


def get_genetics_explanation(score: float, evidence_refs: List[str]) -> str:
    """
    Generate human-readable explanation for genetics score.

    Args:
        score: Genetics score (0-1)
        evidence_refs: List of evidence references

    Returns:
        Explanation string
    """
    if score >= 0.85:
        return f"Very strong genetic association with disease (score: {score:.3f}). This target has well-documented driver mutations or genetic alterations."
    elif score >= 0.70:
        return f"Strong genetic association (score: {score:.3f}). Frequent mutations or alterations observed in disease."
    elif score >= 0.50:
        return f"Moderate genetic association (score: {score:.3f}). Some genetic evidence supports disease involvement."
    elif score >= 0.30:
        return f"Weak genetic association (score: {score:.3f}). Limited genetic evidence available."
    else:
        return f"Minimal genetic association (score: {score:.3f}). Little to no genetic evidence found."


def validate_genetics_inputs(disease: str, target: str) -> Tuple[bool, str]:
    """
    Validate inputs for genetics scoring.

    Args:
        disease: Disease identifier
        target: Target identifier

    Returns:
        (is_valid, error_message)
    """
    if not disease:
        return False, "Disease identifier is required"

    if not target:
        return False, "Target identifier is required"

    # Basic format validation
    if not disease.startswith(("EFO_", "MONDO_", "HP_", "DOID_")):
        logger.warning(f"Disease ID format not recognized: {disease}")

    if len(target) < 2:
        return False, "Target identifier too short"

    # Check for valid gene symbol format
    if not target.replace('_', '').replace('-', '').isalnum():
        return False, "Target identifier contains invalid characters"

    return True, ""


def get_genetics_data_summary() -> dict:
    """
    Get summary of genetics scoring capabilities.

    Returns:
        Dictionary with scoring system information
    """
    return {
        "primary_source": "Open Targets Platform",
        "fallback_mode": "Curated demo scores",
        "score_range": "0.1 - 1.0",
        "evidence_types": [
            "GWAS associations",
            "Rare disease mutations",
            "Somatic mutations",
            "Copy number variations",
            "Structural variants"
        ],
        "demo_targets_covered": 25,
        "confidence_levels": ["low", "medium", "high"]
    }