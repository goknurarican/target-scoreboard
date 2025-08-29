# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Modality fit channel - Phase 1B Production with real data integration.
Uses Expression Atlas + AlphaFold for E3 co-expression and structure confidence.
"""
import asyncio
import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional

from ..schemas import ChannelScore, EvidenceRef, DataQualityFlags, get_utc_now
from ..validation import get_validator
from ..data_access.expression_atlas import get_expression_atlas_client
from ..data_access.alphafold import get_alphafold_client

logger = logging.getLogger(__name__)

# E3 ligase list for co-expression analysis
E3_LIGASES = {
    'CRBN', 'VHL', 'DDB1', 'DCAF1', 'DCAF4', 'DCAF7', 'DCAF8', 'DCAF11', 'DCAF15',
    'MDM2', 'FBXW7', 'BTRC', 'SKP1', 'CUL1', 'CUL2', 'CUL3', 'CUL4A', 'CUL4B'
}

# Tissues relevant for safety assessment
SAFETY_CRITICAL_TISSUES = {
    'heart', 'brain', 'liver', 'kidney', 'blood', 'bone_marrow'
}

# Known ternary complex formation data (minimal curated set)
TERNARY_EVIDENCE = {
    'EGFR': 'reported',
    'ERBB2': 'reported',
    'ALK': 'reported',
    'BTK': 'reported',
    'BCL2': 'reported',
    'KRAS': 'weak',
    'MET': 'weak',
    'BRAF': 'weak',
    'TP53': 'challenging',
    'RB1': 'challenging',
    'MYC': 'challenging'
}


class ModalityFitChannel:
    """
    Production modality fit channel using real expression and structure data.
    """

    def __init__(self):
        self.validator = get_validator()
        self.channel_name = "modality_fit"

    async def compute_score(self, gene: str) -> ChannelScore:
        """
        Compute modality fit score using Expression Atlas + AlphaFold.

        Args:
            gene: Target gene symbol

        Returns:
            ChannelScore with modality subscores and evidence
        """
        evidence_refs = []
        components = {}
        quality_flags = DataQualityFlags()

        try:
            # Fetch data from multiple sources in parallel
            expression_task = self._fetch_expression_data(gene)
            structure_task = self._fetch_structure_data(gene)

            expression_data, structure_data = await asyncio.gather(
                expression_task, structure_task, return_exceptions=True
            )

            # Handle exceptions from parallel fetch
            if isinstance(expression_data, Exception):
                logger.warning(f"Expression data fetch failed for {gene}: {expression_data}")
                expression_data = {}
                quality_flags.partial = True

            if isinstance(structure_data, Exception):
                logger.warning(f"Structure data fetch failed for {gene}: {structure_data}")
                structure_data = None
                quality_flags.partial = True

            # Compute subscores
            e3_coexpr_score = await self._compute_e3_coexpression(gene, expression_data)
            ternary_score = self._compute_ternary_proxy(gene)
            structure_score = self._compute_structure_confidence(structure_data)

            # Compute modality-specific scores
            protac_score = self._compute_protac_score(e3_coexpr_score, ternary_score, structure_score)
            small_mol_score = self._compute_small_molecule_score(structure_score, expression_data)
            molecular_glue_score = self._compute_molecular_glue_score(ternary_score, structure_score)

            # Overall druggability
            overall_score = (
                    0.35 * e3_coexpr_score +
                    0.30 * ternary_score +
                    0.35 * structure_score
            )

            # Build components dict
            components = {
                "e3_coexpr": e3_coexpr_score,
                "ternary_proxy": ternary_score,
                "structure_confidence": structure_score,
                "overall_druggability": overall_score,
                "protac_degrader": protac_score,
                "small_molecule": small_mol_score,
                "molecular_glue": molecular_glue_score
            }

            # Build evidence references
            if expression_data:
                evidence_refs.append(EvidenceRef(
                    source="expression_atlas",
                    title=f"Expression data across {len(expression_data)} tissues",
                    url="https://www.ebi.ac.uk/gxa/",
                    source_quality="high",
                    timestamp=get_utc_now()
                ))

            if structure_data:
                evidence_refs.append(EvidenceRef(
                    source="alphafold",
                    title=f"Structure confidence: pLDDT={structure_data.plddt_mean:.1f}",
                    url=f"https://alphafold.ebi.ac.uk/entry/{gene}",
                    source_quality="high",
                    timestamp=get_utc_now()
                ))

            # Add ternary evidence
            ternary_level = TERNARY_EVIDENCE.get(gene, 'none')
            evidence_refs.append(EvidenceRef(
                source="vantai_curated",
                title=f"Ternary complex evidence: {ternary_level}",
                source_quality="medium",
                timestamp=get_utc_now()
            ))

            logger.info(
                f"Modality fit computed for {gene}",
                extra={
                    "gene": gene,
                    "overall_score": overall_score,
                    "protac_score": protac_score,
                    "expression_tissues": len(expression_data) if expression_data else 0,
                    "structure_available": structure_data is not None
                }
            )

            return ChannelScore(
                name=self.channel_name,
                score=overall_score,
                status="ok",
                components=components,
                evidence=evidence_refs,
                quality=quality_flags
            )

        except Exception as e:
            logger.error(f"Modality fit error for {gene}: {e}")

            return ChannelScore(
                name=self.channel_name,
                score=None,
                status="error",
                components={},
                evidence=[],
                quality=DataQualityFlags(notes=f"Channel error: {str(e)[:100]}"),
            )

    async def _fetch_expression_data(self, gene: str) -> Dict[str, float]:
        """Fetch expression data from Expression Atlas."""
        try:
            atlas_client = await get_expression_atlas_client()
            tissue_expression = await atlas_client.get_tissue_specificity(gene)

            logger.info(f"Expression data fetched for {gene}: {len(tissue_expression)} tissues")
            return tissue_expression

        except Exception as e:
            logger.error(f"Expression fetch failed for {gene}: {e}")
            return {}

    async def _fetch_structure_data(self, gene: str):
        """Fetch structure confidence from AlphaFold."""
        try:
            alphafold_client = await get_alphafold_client()
            structure_confidence = await alphafold_client.get_confidence_scores(gene)

            if structure_confidence:
                # GÜVENLI ATTRIBUTE ACCESS
                plddt_mean = getattr(structure_confidence, 'plddt_mean', None)
                logger.info(f"Structure data fetched for {gene}: pLDDT={plddt_mean}")
            else:
                logger.info(f"No structure data available for {gene}")

            return structure_confidence

        except Exception as e:
            logger.error(f"Structure fetch failed for {gene}: {e}")
            # Return None instead of raising - let channel handle missing data gracefully
            return None
    async def _compute_e3_coexpression(self, gene: str, expression_data: Dict[str, float]) -> float:
        """
        Compute E3 ligase co-expression score using real expression data.

        Args:
            gene: Target gene symbol
            expression_data: Dict mapping tissue to expression level

        Returns:
            Co-expression score (0-1)
        """
        if not expression_data:
            logger.warning(f"No expression data for E3 co-expression analysis: {gene}")
            return 0.3

        try:
            # Get E3 ligase expression data for comparison
            atlas_client = await get_expression_atlas_client()

            coexpression_scores = []
            tissues_analyzed = []

            for tissue, target_expr in expression_data.items():
                if target_expr <= 0:
                    continue

                tissue_e3_scores = []

                # Sample a few key E3 ligases for co-expression analysis
                key_e3_ligases = ['CRBN', 'VHL', 'MDM2', 'DCAF1', 'DCAF4']

                for e3_ligase in key_e3_ligases:
                    try:
                        e3_expression = await atlas_client.get_expression(e3_ligase, tissue)

                        if e3_expression and len(e3_expression) > 0:
                            e3_value = e3_expression[0].value

                            if e3_value > 0:
                                # Compute correlation-like score
                                log_target = np.log2(target_expr + 1)
                                log_e3 = np.log2(e3_value + 1)

                                # Similarity based on log-space difference
                                diff = abs(log_target - log_e3)
                                similarity = 1 / (1 + np.exp((diff - 3) / 2))  # Sigmoid
                                tissue_e3_scores.append(similarity)

                    except Exception as e:
                        logger.debug(f"E3 expression fetch failed for {e3_ligase} in {tissue}: {e}")
                        continue

                if tissue_e3_scores:
                    tissue_avg = np.mean(tissue_e3_scores)
                    coexpression_scores.append(tissue_avg)
                    tissues_analyzed.append(tissue)

            if coexpression_scores:
                final_score = np.mean(coexpression_scores)

                # Consistency bonus across tissues
                if len(coexpression_scores) > 1:
                    consistency = 1 - (np.std(coexpression_scores) / (np.mean(coexpression_scores) + 1e-6))
                    final_score *= (1 + 0.1 * consistency)

                logger.info(f"E3 co-expression for {gene}: {final_score:.3f} across {len(tissues_analyzed)} tissues")
                return max(0.1, min(1.0, final_score))
            else:
                logger.warning(f"No E3 co-expression data computed for {gene}")
                return 0.3

        except Exception as e:
            logger.error(f"E3 co-expression computation failed for {gene}: {e}")
            return 0.3

    def _compute_ternary_proxy(self, gene: str) -> float:
        """
        Compute ternary complex formation proxy score.

        Args:
            gene: Target gene symbol

        Returns:
            Ternary formation score (0-1)
        """
        evidence_level = TERNARY_EVIDENCE.get(gene, 'none')

        score_mapping = {
            'none': 0.25,
            'weak': 0.50,
            'challenging': 0.35,
            'reported': 0.85,
            'validated': 0.95
        }

        base_score = score_mapping.get(evidence_level, 0.25)

        # Bonus for known druggable targets
        if gene in {'EGFR', 'ERBB2', 'ALK', 'BTK', 'BCL2'}:
            base_score = min(1.0, base_score + 0.1)

        return base_score

    def _compute_structure_confidence(self, structure_data) -> float:
        """
        Compute structure-based druggability score.
        """
        if not structure_data:
            return 0.4  # Default for missing structure

        try:
            # GÜVENLI ATTRIBUTE ACCESS
            plddt_mean = getattr(structure_data, 'plddt_mean', None)
            pae_mean = getattr(structure_data, 'pae_mean', None)

            if plddt_mean is None:
                return 0.4

            # pLDDT-based scoring
            if plddt_mean >= 90:
                plddt_score = 0.9
            elif plddt_mean >= 70:
                plddt_score = 0.7
            elif plddt_mean >= 50:
                plddt_score = 0.5
            else:
                plddt_score = 0.3

            # PAE adjustment if available
            if pae_mean is not None:
                if pae_mean <= 5.0:
                    pae_bonus = 0.1
                elif pae_mean <= 10.0:
                    pae_bonus = 0.05
                else:
                    pae_bonus = 0.0

                plddt_score = min(1.0, plddt_score + pae_bonus)

            return plddt_score

        except Exception as e:
            logger.error(f"Structure confidence computation failed: {e}")
            return 0.4
    def _compute_protac_score(self, e3_coexpr: float, ternary: float, structure: float) -> float:
        """Compute PROTAC/degrader suitability score."""
        return (
                0.45 * e3_coexpr +
                0.35 * ternary +
                0.20 * structure
        )

    def _compute_small_molecule_score(self, structure: float, expression_data: Dict[str, float]) -> float:
        """Compute small molecule druggability score."""
        # Base on structure confidence
        base_score = 0.7 * structure

        # Adjust based on expression levels (moderate expression preferred)
        if expression_data:
            avg_expression = np.mean(list(expression_data.values()))

            # Sweet spot around log2(50-200) TPM
            log_expr = np.log2(avg_expression + 1)
            if 5.5 <= log_expr <= 7.5:  # ~50-200 TPM
                expression_bonus = 0.3
            elif 4.0 <= log_expr <= 9.0:  # ~16-512 TPM
                expression_bonus = 0.2
            else:
                expression_bonus = 0.1

            base_score += expression_bonus
        else:
            base_score += 0.2  # Default bonus

        return min(1.0, base_score)

    def _compute_molecular_glue_score(self, ternary: float, structure: float) -> float:
        """Compute molecular glue potential score."""
        return (
                0.6 * ternary +
                0.4 * structure
        )


# ========================
# Global channel instance
# ========================

_modality_channel: Optional[ModalityFitChannel] = None


async def get_modality_channel() -> ModalityFitChannel:
    """Get global modality fit channel instance."""
    global _modality_channel
    if _modality_channel is None:
        _modality_channel = ModalityFitChannel()
    return _modality_channel


# ========================
# Legacy compatibility functions
# ========================

async def compute_modality_fit(target: str, ppi_graph: Optional[nx.Graph] = None) -> Tuple[Dict[str, float], List[str]]:
    """
    Legacy compatibility wrapper for existing scoring.py integration.

    Args:
        target: Target gene symbol
        ppi_graph: PPI graph (legacy parameter, not used in new implementation)

    Returns:
        (modality_scores_dict, evidence_references)
    """
    try:
        modality_channel = await get_modality_channel()
        channel_result = await modality_channel.compute_score(target)

        # Convert to legacy format
        if channel_result.status == "ok" and channel_result.score is not None:
            # Extract scores from components
            scores = channel_result.components.copy()

            # Convert evidence refs to legacy string format
            evidence_strings = []
            for evidence in channel_result.evidence:
                evidence_strings.append(f"Source:{evidence.source}")
                if evidence.title:
                    evidence_strings.append(f"Evidence:{evidence.title[:50]}")

            # Add component info
            for comp_name, comp_value in scores.items():
                if isinstance(comp_value, (int, float)):
                    evidence_strings.append(f"{comp_name}:{comp_value:.3f}")

            return scores, evidence_strings

        elif channel_result.status == "data_missing":
            logger.warning(f"Insufficient modality data for {target}")
            # Return default scores
            default_scores = {
                "e3_coexpr": 0.3,
                "ternary_proxy": 0.3,
                "structure_confidence": 0.3,
                "overall_druggability": 0.3,
                "protac_degrader": 0.3,
                "small_molecule": 0.3,
                "molecular_glue": 0.3
            }
            return default_scores, ["Status:data_missing"]

        else:  # error status
            logger.error(f"Modality channel error for {target}")
            default_scores = {
                "e3_coexpr": 0.3,
                "ternary_proxy": 0.3,
                "structure_confidence": 0.3,
                "overall_druggability": 0.3,
                "protac_degrader": 0.3,
                "small_molecule": 0.3,
                "molecular_glue": 0.3
            }
            return default_scores, [f"Status:error"]

    except Exception as e:
        logger.error(f"Legacy modality score computation failed: {e}")
        default_scores = {
            "e3_coexpr": 0.3,
            "ternary_proxy": 0.3,
            "structure_confidence": 0.3,
            "overall_druggability": 0.3,
            "protac_degrader": 0.3,
            "small_molecule": 0.3,
            "molecular_glue": 0.3
        }
        return default_scores, [f"Error:{str(e)[:50]}"]


def get_modality_recommendations(target: str, ppi_graph: Optional[nx.Graph] = None) -> Dict:
    """
    Get detailed modality recommendations (async wrapper for legacy compatibility).

    Args:
        target: Target gene symbol
        ppi_graph: PPI graph (legacy parameter)

    Returns:
        Dict with recommendations and analysis
    """
    # This function needs to be async but keeping sync for legacy compatibility
    # Will be properly refactored when main.py endpoints are updated

    try:
        # Use fallback scoring for legacy compatibility
        modality_scores = {
            "overall_druggability": 0.5,
            "protac_degrader": 0.5,
            "small_molecule": 0.5,
            "molecular_glue": 0.4,
            "e3_coexpr": 0.5,
            "ternary_proxy": 0.5,
            "structure_confidence": 0.5
        }

        recommendations = {
            "target": target,
            "overall_druggability": modality_scores["overall_druggability"],
            "subscores": {
                "e3_coexpr": modality_scores["e3_coexpr"],
                "ternary_proxy": modality_scores["ternary_proxy"],
                "structure_confidence": modality_scores["structure_confidence"]
            },
            "modality_scores": {
                "protac_degrader": modality_scores["protac_degrader"],
                "small_molecule": modality_scores["small_molecule"],
                "molecular_glue": modality_scores["molecular_glue"]
            },
            "recommendations": [
                {
                    "modality": "PROTAC/Degrader",
                    "priority": "Medium",
                    "confidence": "Legacy fallback",
                    "rationale": "Legacy compatibility mode - use async API for full analysis"
                }
            ]
        }

        return recommendations

    except Exception as e:
        logger.error(f"Modality recommendations failed for {target}: {e}")
        return {
            "target": target,
            "error": str(e),
            "recommendations": []
        }


# Legacy global instances for compatibility
class LegacyModalityAnalyzer:
    """Legacy analyzer for backward compatibility."""

    def __init__(self):
        self.expression_data = {}
        self.ternary_data = TERNARY_EVIDENCE

    def compute_modality_scores(self, target: str, ppi_graph: Optional[nx.Graph] = None) -> Dict[str, float]:
        """Legacy modality score computation."""
        return {
            "e3_coexpr": 0.5,
            "ternary_proxy": TERNARY_EVIDENCE.get(target, 0.3),
            "ppi_hotspot": 0.5,
            "overall_druggability": 0.5,
            "protac_degrader": 0.5,
            "small_molecule": 0.5,
            "molecular_glue": 0.4,
            "adc_score": 0.4
        }


modality_analyzer = LegacyModalityAnalyzer()


def get_modality_data_summary() -> Dict:
    """Get summary of modality data sources."""
    return {
        "data_sources": ["Expression Atlas", "AlphaFold", "VantAI Curated"],
        "e3_ligases_tracked": len(E3_LIGASES),
        "ternary_evidence_targets": len(TERNARY_EVIDENCE),
        "analysis_methods": ["E3 co-expression", "Ternary complex evidence", "Structure confidence"],
        "modality_types": ["PROTAC/Degrader", "Small molecule", "Molecular glue"],
        "note": "Production data from Expression Atlas + AlphaFold APIs"
    }