# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Safety channel - Phase 1B Production with Expression Atlas integration.
Real tissue expression data for off-target safety assessment.
"""
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..schemas import ChannelScore, EvidenceRef, DataQualityFlags, get_utc_now
from ..validation import get_validator
from ..data_access.expression_atlas import get_expression_atlas_client

logger = logging.getLogger(__name__)

# Safety-critical tissues for off-target assessment
SAFETY_CRITICAL_TISSUES = [
    'heart', 'brain', 'liver', 'kidney', 'blood', 'bone_marrow',
    'lung', 'muscle', 'skin', 'immune_system'
]

# Known safety profiles (curated from drug development experience)
KNOWN_SAFETY_PROFILES = {
    # Low risk - approved drugs with established safety
    'EGFR': {
        'risk_level': 'low',
        'approved_drugs': ['erlotinib', 'gefitinib', 'osimertinib'],
        'known_toxicities': ['skin_rash', 'diarrhea'],
        'tissue_specificity_expected': 0.7
    },
    'ERBB2': {
        'risk_level': 'low',
        'approved_drugs': ['trastuzumab', 'pertuzumab'],
        'known_toxicities': ['cardiotoxicity'],
        'tissue_specificity_expected': 0.8
    },
    'ALK': {
        'risk_level': 'low',
        'approved_drugs': ['crizotinib', 'alectinib'],
        'known_toxicities': ['hepatotoxicity', 'pneumonitis'],
        'tissue_specificity_expected': 0.9
    },
    'BRAF': {
        'risk_level': 'moderate',
        'approved_drugs': ['vemurafenib', 'dabrafenib'],
        'known_toxicities': ['skin_reactions', 'arthralgia'],
        'tissue_specificity_expected': 0.6
    },

    # Moderate risk
    'MET': {
        'risk_level': 'moderate',
        'approved_drugs': ['capmatinib'],
        'known_toxicities': ['peripheral_edema', 'nausea'],
        'tissue_specificity_expected': 0.5
    },
    'KRAS': {
        'risk_level': 'moderate',
        'approved_drugs': ['sotorasib'],
        'known_toxicities': ['diarrhea', 'fatigue'],
        'tissue_specificity_expected': 0.4
    },

    # High risk - essential genes or broad expression
    'TP53': {
        'risk_level': 'high',
        'approved_drugs': [],
        'known_toxicities': ['cell_cycle_disruption'],
        'tissue_specificity_expected': 0.2
    },
    'RB1': {
        'risk_level': 'high',
        'approved_drugs': [],
        'known_toxicities': ['cell_cycle_disruption'],
        'tissue_specificity_expected': 0.3
    },
    'PTEN': {
        'risk_level': 'high',
        'approved_drugs': [],
        'known_toxicities': ['metabolic_disruption'],
        'tissue_specificity_expected': 0.2
    }
}


class SafetyChannel:
    """
    Production safety channel using real tissue expression data.
    """

    def __init__(self):
        self.validator = get_validator()
        self.channel_name = "safety"

    def _convert_quality_to_score(self, quality_level: str) -> float:
        """Convert quality level string to numeric score."""
        quality_mapping = {
            "high": 1.0,
            "medium": 0.5,
            "low": 0.2,
            "unknown": 0.1
        }
        return quality_mapping.get(quality_level.lower(), 0.5)
    async def compute_score(self, gene: str, disease_context: Optional[str] = None) -> ChannelScore:
        """
        Compute safety penalty score using Expression Atlas tissue data.

        Args:
            gene: Target gene symbol
            disease_context: Disease context for tissue relevance

        Returns:
            ChannelScore with safety penalty (higher = more concerning)
        """
        evidence_refs = []
        components = {}
        quality_flags = DataQualityFlags()

        try:
            # Get known safety profile if available
            safety_profile = KNOWN_SAFETY_PROFILES.get(gene, {})

            # Fetch tissue expression data
            atlas_client = await get_expression_atlas_client()
            tissue_expression = await atlas_client.get_tissue_specificity(gene)

            if not tissue_expression and not safety_profile:
                logger.warning(f"No safety data available for {gene}")
                return ChannelScore(
                    name=self.channel_name,
                    score=None,
                    status="data_missing",
                    components={},
                    evidence=[],
                    quality=DataQualityFlags(partial=True, notes="No expression or safety profile data")
                )

            # Compute safety penalty components
            expression_penalty = await self._compute_expression_penalty(gene, tissue_expression)
            profile_penalty = self._compute_profile_penalty(safety_profile)

            # Combine penalties (weighted average)
            if tissue_expression and safety_profile:
                # Both sources available
                final_penalty = 0.6 * expression_penalty + 0.4 * profile_penalty
                data_quality = "high"
            elif safety_profile:
                # Only profile available
                final_penalty = profile_penalty
                data_quality = "medium"
                quality_flags.partial = True
            else:
                # Only expression available
                final_penalty = expression_penalty
                data_quality = "medium"
                quality_flags.partial = True

            # Build components
            components = {
                "expression_penalty": expression_penalty,
                "profile_penalty": profile_penalty,
                "final_penalty": final_penalty,
                "data_quality_score": self._convert_quality_to_score(data_quality),  # Float değer
                "data_quality_level": data_quality,  # String değer (artık Any type kabul ediliyor)
                "tissues_analyzed": len(tissue_expression) if tissue_expression else 0
            }

            # Build evidence references
            if tissue_expression:
                evidence_refs.append(EvidenceRef(
                    source="expression_atlas",
                    title=f"Tissue expression across {len(tissue_expression)} tissues",
                    url="https://www.ebi.ac.uk/gxa/",
                    source_quality="high",
                    timestamp=get_utc_now()
                ))

            if safety_profile:
                approved_drugs = safety_profile.get('approved_drugs', [])
                evidence_refs.append(EvidenceRef(
                    source="vantai_curated",
                    title=f"Safety profile: {safety_profile.get('risk_level', 'unknown')} risk, {len(approved_drugs)} approved drugs",
                    source_quality="high",
                    timestamp=get_utc_now()
                ))

            # Add tissue specificity evidence
            if tissue_expression:
                max_tissue = max(tissue_expression.items(), key=lambda x: x[1])
                evidence_refs.append(EvidenceRef(
                    source="expression_atlas",
                    title=f"Highest expression in {max_tissue[0]}: {max_tissue[1]:.1f} TPM",
                    url=f"https://www.ebi.ac.uk/gxa/genes/{gene}",
                    source_quality="high",
                    timestamp=get_utc_now()
                ))

            logger.info(
                f"Safety penalty computed for {gene}",
                extra={
                    "gene": gene,
                    "penalty": final_penalty,
                    "expression_tissues": len(tissue_expression) if tissue_expression else 0,
                    "has_safety_profile": bool(safety_profile),
                    "data_quality": data_quality
                }
            )

            return ChannelScore(
                name=self.channel_name,
                score=final_penalty,  # Note: this is a penalty, higher = worse
                status="ok",
                components=components,
                evidence=evidence_refs,
                quality=quality_flags
            )

        except Exception as e:
            logger.error(f"Safety channel error for {gene}: {e}")

            return ChannelScore(
                name=self.channel_name,
                score=None,
                status="error",
                components={"error": str(e)[:100]},
                evidence=[],
                quality=DataQualityFlags(notes=f"Channel error: {str(e)[:100]}")
            )

    async def _compute_expression_penalty(self, gene: str, tissue_expression: Dict[str, float]) -> float:
        """
        Compute safety penalty based on tissue expression patterns.

        Args:
            gene: Gene symbol
            tissue_expression: Dict mapping tissue to expression level

        Returns:
            Expression-based penalty (0-1, higher = more concerning)
        """
        if not tissue_expression:
            return 0.5  # Default penalty for unknown expression

        try:
            # Analyze expression across safety-critical tissues
            critical_expression = []
            total_expression = []

            for tissue, expression_level in tissue_expression.items():
                total_expression.append(expression_level)

                # Check if this is a safety-critical tissue
                tissue_lower = tissue.lower()
                is_critical = any(
                    critical_tissue in tissue_lower
                    for critical_tissue in SAFETY_CRITICAL_TISSUES
                )

                if is_critical:
                    critical_expression.append(expression_level)

            # Calculate tissue specificity metrics
            if len(total_expression) < 2:
                return 0.5

            max_expression = max(total_expression)
            mean_expression = np.mean(total_expression)

            # Tissue specificity = max / mean (higher = more specific = safer)
            if mean_expression > 0:
                specificity = max_expression / mean_expression
            else:
                specificity = 1.0

            # Penalty calculation
            # High specificity (> 5x) = lower penalty
            # Broad expression = higher penalty
            if specificity > 5.0:
                specificity_penalty = 0.2  # Very specific
            elif specificity > 3.0:
                specificity_penalty = 0.4  # Moderately specific
            elif specificity > 2.0:
                specificity_penalty = 0.6  # Some specificity
            else:
                specificity_penalty = 0.8  # Broadly expressed

            # Critical tissue penalty
            critical_penalty = 0.0
            if critical_expression:
                avg_critical = np.mean(critical_expression)
                # High expression in critical tissues increases penalty
                if avg_critical > 100:  # High TPM
                    critical_penalty = 0.3
                elif avg_critical > 50:
                    critical_penalty = 0.2
                elif avg_critical > 10:
                    critical_penalty = 0.1

            # Combine penalties
            total_penalty = min(1.0, specificity_penalty + critical_penalty)

            return max(0.1, total_penalty)

        except Exception as e:
            logger.error(f"Expression penalty computation failed for {gene}: {e}")
            return 0.5

    def _compute_profile_penalty(self, safety_profile: Dict) -> float:
        """
        Compute penalty based on known safety profile.

        Args:
            safety_profile: Known safety data dict

        Returns:
            Profile-based penalty (0-1)
        """
        if not safety_profile:
            return 0.5

        # Base penalty from risk level
        risk_level = safety_profile.get('risk_level', 'moderate')
        risk_penalties = {
            'low': 0.2,
            'moderate': 0.5,
            'high': 0.8
        }
        base_penalty = risk_penalties.get(risk_level, 0.5)

        # Approved drug bonus (indicates validated safety)
        approved_drugs = safety_profile.get('approved_drugs', [])
        if approved_drugs:
            drug_bonus = min(0.3, len(approved_drugs) * 0.1)
            base_penalty = max(0.1, base_penalty - drug_bonus)

        # Expected tissue specificity adjustment
        expected_specificity = safety_profile.get('tissue_specificity_expected', 0.5)
        specificity_bonus = expected_specificity * 0.2
        final_penalty = max(0.1, base_penalty - specificity_bonus)

        return min(1.0, final_penalty)


# ========================
# Global channel instance
# ========================

_safety_channel: Optional[SafetyChannel] = None


async def get_safety_channel() -> SafetyChannel:
    """Get global safety channel instance."""
    global _safety_channel
    if _safety_channel is None:
        _safety_channel = SafetyChannel()
    return _safety_channel


# ========================
# Legacy compatibility functions
# ========================

async def compute_safety_penalty(target: str, ot_data: dict = None) -> Tuple[float, List[str]]:
    """
    Legacy compatibility wrapper for existing scoring.py integration.

    Args:
        target: Target gene symbol
        ot_data: OpenTargets data (legacy parameter)

    Returns:
        (safety_penalty, evidence_references) - penalty is 0-1 where higher = more concerning
    """
    try:
        safety_channel = await get_safety_channel()

        # Extract disease context if available from ot_data
        disease_context = None
        if ot_data and isinstance(ot_data, dict):
            # Could extract disease info here if needed
            pass

        channel_result = await safety_channel.compute_score(target, disease_context)

        # Convert to legacy format
        if channel_result.status == "ok" and channel_result.score is not None:
            penalty = channel_result.score

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

            # Add safety assessment
            if penalty < 0.3:
                evidence_strings.append("Assessment:favorable")
            elif penalty > 0.7:
                evidence_strings.append("Assessment:concerning")
            else:
                evidence_strings.append("Assessment:moderate")

            return penalty, evidence_strings

        elif channel_result.status == "data_missing":
            logger.warning(f"No safety data for {target}")
            return 0.5, ["Status:data_missing", "Default:moderate_risk"]

        else:  # error status
            logger.error(f"Safety channel error for {target}")
            return 0.5, [f"Status:error", "Default:moderate_risk"]

    except Exception as e:
        logger.error(f"Legacy safety penalty computation failed: {e}")
        return 0.5, [f"Error:{str(e)[:50]}"]


def get_safety_recommendations(target: str, penalty: float) -> List[str]:
    """Generate safety-focused recommendations."""
    recommendations = []

    if penalty < 0.3:
        recommendations.append("Target has favorable safety profile")
        recommendations.append("Approved drugs provide safety precedent")
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
    return {
        "known_safety_profiles": len(KNOWN_SAFETY_PROFILES),
        "critical_tissues_monitored": len(SAFETY_CRITICAL_TISSUES),
        "risk_categories": ["low", "moderate", "high"],
        "data_sources": [
            "Expression Atlas tissue specificity",
            "FDA drug labels",
            "Literature curation",
            "Clinical trial safety data"
        ],
        "assessment_factors": [
            "Tissue expression specificity",
            "Safety-critical tissue expression",
            "Approved drug precedent",
            "Known toxicity patterns"
        ],
        "note": "Production data from Expression Atlas API"
    }


# Legacy compatibility for existing code
class SafetyAnalyzer:
    """Legacy analyzer for backward compatibility."""

    def __init__(self):
        self.known_safety_profiles = KNOWN_SAFETY_PROFILES


# Global instance for legacy compatibility
safety_analyzer = SafetyAnalyzer()