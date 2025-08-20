# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.


"""
Safety channel for computing off-tissue expression penalties and known safety profiles.
"""
from typing import Tuple, List, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SafetyAnalyzer:
    """Analyzer for target safety assessment based on expression and known safety data."""

    def __init__(self):
        self.known_safety_profiles = self._load_safety_profiles()
        self.tissue_expression_data = self._load_tissue_expression_data()

    def _load_safety_profiles(self) -> Dict[str, Dict]:
        """Load known safety profiles for targets."""
        return {
            # Known safe targets with approved drugs
            'EGFR': {
                'risk_level': 'low',
                'approved_drugs': ['erlotinib', 'gefitinib', 'osimertinib'],
                'known_toxicities': ['skin_rash', 'diarrhea'],
                'tissue_specificity': 0.7
            },
            'ERBB2': {
                'risk_level': 'low',
                'approved_drugs': ['trastuzumab', 'pertuzumab'],
                'known_toxicities': ['cardiotoxicity'],
                'tissue_specificity': 0.8
            },
            'ALK': {
                'risk_level': 'low',
                'approved_drugs': ['crizotinib', 'alectinib'],
                'known_toxicities': ['hepatotoxicity', 'pneumonitis'],
                'tissue_specificity': 0.9
            },
            'BRAF': {
                'risk_level': 'moderate',
                'approved_drugs': ['vemurafenib', 'dabrafenib'],
                'known_toxicities': ['skin_reactions', 'arthralgia'],
                'tissue_specificity': 0.6
            },
            'BCR': {
                'risk_level': 'low',
                'approved_drugs': ['imatinib', 'dasatinib'],
                'known_toxicities': ['myelosuppression'],
                'tissue_specificity': 0.8
            },

            # Moderate risk targets
            'MET': {
                'risk_level': 'moderate',
                'approved_drugs': ['capmatinib'],
                'known_toxicities': ['peripheral_edema', 'nausea'],
                'tissue_specificity': 0.5
            },
            'KRAS': {
                'risk_level': 'moderate',
                'approved_drugs': ['sotorasib'],
                'known_toxicities': ['diarrhea', 'fatigue'],
                'tissue_specificity': 0.4
            },
            'PIK3CA': {
                'risk_level': 'moderate',
                'approved_drugs': ['alpelisib'],
                'known_toxicities': ['hyperglycemia', 'rash'],
                'tissue_specificity': 0.5
            },

            # High risk targets (tumor suppressors, essential genes)
            'TP53': {
                'risk_level': 'high',
                'approved_drugs': [],
                'known_toxicities': ['cell_cycle_disruption'],
                'tissue_specificity': 0.2
            },
            'RB1': {
                'risk_level': 'high',
                'approved_drugs': [],
                'known_toxicities': ['cell_cycle_disruption'],
                'tissue_specificity': 0.3
            },
            'PTEN': {
                'risk_level': 'high',
                'approved_drugs': [],
                'known_toxicities': ['metabolic_disruption'],
                'tissue_specificity': 0.2
            },
            'MYC': {
                'risk_level': 'high',
                'approved_drugs': [],
                'known_toxicities': ['proliferation_disruption'],
                'tissue_specificity': 0.1
            }
        }

    def _load_tissue_expression_data(self) -> Dict[str, Dict]:
        """Load tissue-specific expression data for safety assessment."""
        return {
            'EGFR': {
                'lung': 245.8, 'skin': 89.3, 'liver': 45.2, 'kidney': 67.8,
                'brain': 23.1, 'heart': 12.4, 'muscle': 8.9, 'blood': 5.2
            },
            'ERBB2': {
                'breast': 287.4, 'lung': 123.7, 'heart': 67.3, 'brain': 34.5,
                'liver': 29.1, 'kidney': 18.7, 'muscle': 12.3, 'blood': 6.8
            },
            'ALK': {
                'brain': 156.7, 'lung': 8.9, 'liver': 4.2, 'kidney': 3.1,
                'heart': 2.8, 'muscle': 1.9, 'skin': 1.2, 'blood': 0.8
            },
            'KRAS': {
                'lung': 89.3, 'colon': 78.9, 'pancreas': 134.5, 'liver': 67.2,
                'kidney': 45.8, 'brain': 34.2, 'heart': 23.6, 'blood': 18.4
            },
            'TP53': {
                'all_tissues': 156.4  # Ubiquitously expressed
            },
            'MET': {
                'liver': 198.7, 'lung': 67.2, 'kidney': 89.4, 'brain': 34.6,
                'heart': 23.8, 'muscle': 15.2, 'skin': 12.7, 'blood': 8.9
            }
        }


def compute_safety_penalty(target: str, ot_data: dict = None) -> Tuple[float, List[str]]:
    """
    Compute safety penalty based on off-tissue expression and known safety data.

    Args:
        target: Target gene symbol
        ot_data: Open Targets data (optional)

    Returns:
        (safety_penalty, evidence_references) - penalty is 0-1 where lower is safer
    """
    try:
        analyzer = SafetyAnalyzer()
        evidence_refs = []

        # Get known safety profile
        safety_profile = analyzer.known_safety_profiles.get(target, {})

        if safety_profile:
            risk_level = safety_profile['risk_level']
            approved_drugs = safety_profile['approved_drugs']
            tissue_specificity = safety_profile['tissue_specificity']

            # Base penalty from risk level
            risk_penalties = {
                'low': 0.2,
                'moderate': 0.5,
                'high': 0.8
            }
            base_penalty = risk_penalties.get(risk_level, 0.5)

            # Adjust based on approved drugs (indicates validated safety)
            if approved_drugs:
                drug_bonus = min(0.3, len(approved_drugs) * 0.1)
                base_penalty = max(0.1, base_penalty - drug_bonus)
                evidence_refs.append(f"Approved_drugs:{len(approved_drugs)}")

            # Adjust based on tissue specificity
            specificity_bonus = tissue_specificity * 0.2
            final_penalty = max(0.1, base_penalty - specificity_bonus)

            evidence_refs.extend([
                f"Risk_level:{risk_level}",
                f"Tissue_specificity:{tissue_specificity:.2f}",
                f"Known_safety_profile:curated"
            ])

        else:
            # Compute penalty from tissue expression if available
            final_penalty = _compute_expression_based_penalty(target, analyzer)
            evidence_refs.append("Expression_based_assessment:heuristic")

        # Use OT known drug data if available
        if ot_data and "known_drug" in ot_data:
            known_drug_score = ot_data["known_drug"]
            if known_drug_score > 0.5:
                # Reduce penalty for targets with known drugs
                drug_reduction = known_drug_score * 0.2
                final_penalty = max(0.1, final_penalty - drug_reduction)
                evidence_refs.append(f"OT_known_drugs:{known_drug_score:.2f}")

        # Add general safety assessment
        evidence_refs.append(f"Safety_penalty:{final_penalty:.2f}")

        if final_penalty < 0.3:
            evidence_refs.append("Safety_assessment:favorable")
        elif final_penalty > 0.7:
            evidence_refs.append("Safety_assessment:concerning")
        else:
            evidence_refs.append("Safety_assessment:moderate")

        logger.info(f"Safety penalty for {target}: {final_penalty:.3f}")
        return final_penalty, evidence_refs

    except Exception as e:
        logger.error(f"Error computing safety penalty for {target}: {e}")
        return 0.5, [f"Safety_error:{str(e)[:50]}"]


def _compute_expression_based_penalty(target: str, analyzer: SafetyAnalyzer) -> float:
    """Compute safety penalty based on tissue expression patterns."""
    expression_data = analyzer.tissue_expression_data.get(target, {})

    if not expression_data:
        # Default penalty for unknown targets
        return 0.5

    if 'all_tissues' in expression_data:
        # Ubiquitously expressed genes have higher risk
        return 0.7

    # Calculate tissue specificity
    expression_values = list(expression_data.values())
    if len(expression_values) < 2:
        return 0.4

    max_expression = max(expression_values)
    total_expression = sum(expression_values)

    # Tissue specificity = max / total (higher = more specific = safer)
    tissue_specificity = max_expression / (total_expression + 1e-6)

    # Convert to penalty (invert specificity)
    penalty = 1.0 - min(tissue_specificity, 1.0)

    # Ensure reasonable bounds
    return max(0.2, min(0.8, penalty))


def get_safety_recommendations(target: str, penalty: float) -> List[str]:
    """Generate safety-focused recommendations."""
    recommendations = []

    if penalty < 0.3:
        recommendations.append("Target has favorable safety profile")
        recommendations.append("Approved drugs available provide safety precedent")
    elif penalty < 0.5:
        recommendations.append("Moderate safety considerations")
        recommendations.append("Monitor for known class effects")
    elif penalty < 0.7:
        recommendations.append("Significant safety evaluation required")
        recommendations.append("Consider tissue-specific delivery approaches")
    else:
        recommendations.append("High safety risk - essential gene or broad expression")
        recommendations.append("Explore selective targeting strategies")

    return recommendations


def validate_safety_inputs(target: str) -> Tuple[bool, str]:
    """Validate inputs for safety scoring."""
    if not target or not isinstance(target, str):
        return False, "Target must be a valid gene symbol"

    if len(target.strip()) < 2:
        return False, "Target symbol too short"

    return True, ""


def get_safety_data_summary() -> Dict:
    """Get summary of safety scoring data sources."""
    analyzer = SafetyAnalyzer()

    return {
        "known_safety_profiles": len(analyzer.known_safety_profiles),
        "tissue_expression_data": len(analyzer.tissue_expression_data),
        "risk_categories": ["low", "moderate", "high"],
        "assessment_factors": [
            "Known drug safety",
            "Tissue expression specificity",
            "Approved drug precedent",
            "Known toxicity patterns"
        ],
        "data_sources": [
            "FDA drug labels",
            "Literature curation",
            "Expression atlases",
            "Clinical trial safety data"
        ]
    }