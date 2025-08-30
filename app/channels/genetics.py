# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Genetics channel for target scoring - Phase 1C Production.
Uses OpenTargets + PubMed via data_access layer. No synthetic fallbacks.
"""
import logging
from typing import List, Optional

from ..schemas import ChannelScore, EvidenceRef, DataQualityFlags, get_utc_now
from ..validation import get_validator
from ..data_access.opentargets import get_ot_client

logger = logging.getLogger(__name__)


class GeneticsChannel:
    """
    Production genetics channel using real OpenTargets data.
    """

    def __init__(self):
        self.validator = get_validator()
        self.channel_name = "genetics"

    async def compute_score(self, gene: str, disease: str) -> ChannelScore:
        """
        Compute genetics association score from OpenTargets + PubMed.
        """
        evidence_refs: List[EvidenceRef] = []
        components = {}
        quality_flags = DataQualityFlags()

        try:
            ot_client = await get_ot_client()
            associations = await ot_client.fetch_gene_disease_associations(gene, disease)

            # Hiç association yoksa doğrudan data_missing
            if not associations:
                logger.warning(f"No genetic associations found for {gene}-{disease}")
                return ChannelScore(
                    name=self.channel_name,
                    score=None,
                    status="data_missing",
                    components={},
                    evidence=[],
                    quality=DataQualityFlags(partial=True, notes="No associations found (OT v4)")
                )

            # Validasyon
            for association in associations:
                vr = self.validator.validate("opentargets", association)
                if not vr.ok:
                    quality_flags.partial = True
                    n = "; ".join(vr.issues[:3])
                    quality_flags.notes = (quality_flags.notes + " | " if quality_flags.notes else "") + f"Validation: {n}"

            # Skorları topla
            overall_score = 0.0
            genetics_score = 0.0
            literature_score = 0.0
            total_evidence_count = 0

            for association in associations:
                if association.source == "opentargets":
                    overall_score = max(overall_score, float(association.score or 0.0))
                elif association.source == "opentargets_genetics":
                    genetics_score = max(genetics_score, float(association.score or 0.0))

                # Kanıtlar
                if association.evidence:
                    total_evidence_count += len(association.evidence)
                    evidence_refs.extend(association.evidence)

            # Evidence dedupe (pmid -> url -> title sırası)
            deduped = []
            seen = set()
            for ev in evidence_refs:
                key = (ev.source or "", ev.pmid or ev.url or ev.title or "")
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(ev)
            evidence_refs = deduped

            # Nihai skor: varsa genetics, yoksa overall
            final_genetics_score = genetics_score if genetics_score > 0 else overall_score

            # Bileşenler
            components = {
                "ot_overall": overall_score,
                "ot_genetics": genetics_score,
                "literature": literature_score,
                "evidence_count": total_evidence_count
            }

            # Güven seviyesi (basit sezgisel)
            if total_evidence_count >= 50:
                confidence_level = "high"
            elif total_evidence_count >= 10:
                confidence_level = "medium"
            else:
                confidence_level = "low"
                quality_flags.partial = True
            components["confidence_level"] = confidence_level

            # --- EN ÖNEMLİ KISIM ---
            # Sinyal var mı? (skor>0 veya kanıt>0)
            has_signal = (final_genetics_score and final_genetics_score > 0.0) or (total_evidence_count > 0)

            status = "ok" if has_signal else "data_missing"
            score_to_return = final_genetics_score if has_signal else None

            if not has_signal:
                # Nedenini de yaz
                quality_flags.notes = (
                    (quality_flags.notes + " | ") if quality_flags.notes else ""
                ) + "No OT genetic/overall signal for this disease (OT v4)"

            logger.info(
                f"Genetics score for {gene}-{disease}: "
                f"{'none' if score_to_return is None else f'{score_to_return:.3f}'} "
                f"(status={status}, evidences={total_evidence_count}, conf={confidence_level})",
                extra={
                    "gene": gene, "disease": disease, "score": score_to_return,
                    "evidence_count": total_evidence_count, "confidence": confidence_level, "status": status
                }
            )

            return ChannelScore(
                name=self.channel_name,
                score=score_to_return,
                status=status,
                components=components,
                evidence=evidence_refs,
                quality=quality_flags
            )

        except Exception as e:
            logger.error(f"Genetics channel error for {gene}-{disease}: {e}")
            return ChannelScore(
                name=self.channel_name,
                score=None,
                status="error",
                components={"error": str(e)[:100]},
                evidence=[],
                quality=DataQualityFlags(notes=f"Channel error: {str(e)[:100]}")
            )

    async def fetch_literature_evidence(self, gene: str, disease: str) -> List[EvidenceRef]:
        """
        Fetch additional literature evidence via PubMed.

        Args:
            gene: Gene symbol
            disease: Disease identifier

        Returns:
            List of EvidenceRef objects from literature
        """
        try:
            # TODO: Implement PubMedClient when created in Phase 1B
            # For now, return evidence from OpenTargets associations
            ot_client = await get_ot_client()
            evidence_refs = await ot_client.fetch_evidences(gene, disease)

            logger.info(f"Fetched {len(evidence_refs)} literature evidence for {gene}-{disease}")
            return evidence_refs

        except Exception as e:
            logger.error(f"Literature evidence fetch failed for {gene}-{disease}: {e}")
            return []


# ========================
# Global channel instance
# ========================

_genetics_channel: Optional[GeneticsChannel] = None


async def get_genetics_channel() -> GeneticsChannel:
    """Get global genetics channel instance."""
    global _genetics_channel
    if _genetics_channel is None:
        _genetics_channel = GeneticsChannel()
    return _genetics_channel


# ========================
# Legacy compatibility functions (for existing scoring.py)
# ========================

async def compute_genetics_score(disease: str, target: str, ot_data: dict) -> tuple[float, List[str]]:
    """
    Legacy compatibility wrapper for existing scoring.py integration.

    Args:
        disease: Disease EFO identifier
        target: Target gene symbol
        ot_data: OpenTargets API response (legacy format)

    Returns:
        (genetics_score, evidence_references) - compatible with existing code
    """
    try:
        # Use new production channel
        genetics_channel = await get_genetics_channel()
        channel_result = await genetics_channel.compute_score(target, disease)

        # Convert to legacy format
        if channel_result.status == "ok" and channel_result.score is not None:
            score = channel_result.score

            # Convert evidence refs to legacy string format
            evidence_strings = []
            for evidence in channel_result.evidence:
                if evidence.pmid:
                    evidence_strings.append(f"PMID:{evidence.pmid}")
                evidence_strings.append(f"Source:{evidence.source}")

            # Add component info
            for comp_name, comp_value in channel_result.components.items():
                if isinstance(comp_value, (int, float)):
                    evidence_strings.append(f"{comp_name}:{comp_value:.3f}")
                else:
                    evidence_strings.append(f"{comp_name}:{comp_value}")

            return score, evidence_strings

        elif channel_result.status == "data_missing":
            logger.warning(f"No genetics data for {target}-{disease}")
            return 0.0, ["Status:data_missing"]

        else:  # error status
            logger.error(f"Genetics channel error for {target}-{disease}")
            return 0.0, [f"Status:error"]

    except Exception as e:
        logger.error(f"Legacy genetics score computation failed: {e}")
        return 0.0, [f"Error:{str(e)[:50]}"]


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


def validate_genetics_inputs(disease: str, target: str) -> tuple[bool, str]:
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
        "primary_source": "Open Targets Platform (Production API)",
        "fallback_mode": "None (data_missing status returned)",
        "score_range": "0.0 - 1.0",
        "evidence_types": [
            "GWAS associations",
            "Rare disease mutations",
            "Somatic mutations",
            "Copy number variations",
            "Structural variants"
        ],
        "validation": "DataQualityValidator with staleness checks",
        "cache_ttl": "24 hours",
        "status_types": ["ok", "data_missing", "error"]
    }