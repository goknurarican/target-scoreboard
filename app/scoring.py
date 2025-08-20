"""
Complete scoring implementation with fixed explanation generation and proper error handling.
"""
from typing import List, Dict, Tuple, Optional
import asyncio
from datetime import datetime
import logging

from .schemas import TargetScore, TargetBreakdown, ScoreRequest, ModalityFitScores
from .data_access.opentargets import fetch_ot_association
from .channels.genetics import compute_genetics_score
from .channels.ppi_proximity import compute_ppi_proximity, ppi_network
from .channels.pathway import compute_pathway_enrichment
from .channels.modality_fit import compute_modality_fit
from .channels.safety import compute_safety_penalty

logger = logging.getLogger(__name__)

# Default weights configuration
DEFAULT_WEIGHTS = {
    "genetics": 0.35,
    "ppi": 0.25,
    "pathway": 0.20,
    "safety": 0.10,
    "modality_fit": 0.10
}


class ExplanationBuilder:
    """Enhanced builder for target scoring explanations with comprehensive evidence formatting."""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def build_evidence_refs(self, evidence_refs: List[str]) -> List[Dict[str, str]]:
        """Convert evidence references to clickable objects with proper URLs and categorization."""
        clickable_evidence = []

        for ref in evidence_refs:
            try:
                if "PMID:" in ref:
                    pmid = ref.split("PMID:")[1].split()[0]
                    clickable_evidence.append({
                        "label": f"PMID:{pmid}",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                        "type": "literature"
                    })
                elif "OpenTargets:" in ref or "OT-" in ref:
                    version = ref.split(":")[-1] if ":" in ref else "2024.06"
                    clickable_evidence.append({
                        "label": f"Open Targets {version}",
                        "url": "https://platform.opentargets.org/",
                        "type": "database"
                    })
                elif "STRING:" in ref:
                    clickable_evidence.append({
                        "label": "STRING Database",
                        "url": "https://string-db.org/",
                        "type": "database"
                    })
                elif "Reactome:" in ref:
                    clickable_evidence.append({
                        "label": "Reactome Pathways",
                        "url": "https://reactome.org/",
                        "type": "database"
                    })
                elif "VantAI:" in ref:
                    clickable_evidence.append({
                        "label": "VantAI Proprietary",
                        "url": "#",
                        "type": "proprietary"
                    })
                elif ref.startswith("OT_"):
                    clickable_evidence.append({
                        "label": "Open Targets",
                        "url": "https://platform.opentargets.org/",
                        "type": "database"
                    })
                elif "Demo_" in ref:
                    clickable_evidence.append({
                        "label": "Demo Data",
                        "url": "#",
                        "type": "demo"
                    })
                else:
                    # Handle other reference types with truncation
                    label = ref[:50] + "..." if len(ref) > 50 else ref
                    clickable_evidence.append({
                        "label": label,
                        "url": "#",
                        "type": "other"
                    })
            except Exception as e:
                logger.warning(f"Error processing evidence reference '{ref}': {e}")
                continue

        return clickable_evidence

    def compute_contributions(self, channel_scores: Dict[str, float]) -> List[Dict]:
        """Compute weighted contributions for each channel with enhanced availability tracking."""
        contributions = []

        for channel, weight in self.weights.items():
            score = channel_scores.get(channel, None)

            # Handle None scores and edge cases gracefully
            if score is None or (isinstance(score, float) and score < 0):
                available = False
                effective_score = 0.0
                contribution = 0.0
            else:
                available = True
                effective_score = float(score)

                # Handle safety inversion (safety is a penalty, so we invert it)
                if channel == "safety":
                    effective_score = 1.0 - effective_score

                # Ensure bounds
                effective_score = max(0.0, min(1.0, effective_score))
                contribution = weight * effective_score

            contributions.append({
                "channel": channel,
                "weight": weight,
                "score": score,  # Keep original score for display
                "contribution": contribution,
                "available": available
            })

        # Sort by contribution descending
        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        return contributions

    def build_explanation(self, target: str, channel_scores: Dict[str, float],
                          evidence_refs: List[str]) -> Dict:
        """Build complete explanation object with enhanced insights."""
        contributions = self.compute_contributions(channel_scores)
        clickable_evidence = self.build_evidence_refs(evidence_refs)

        # Calculate metrics
        total_weighted_score = sum(c["contribution"] for c in contributions)
        available_channels = sum(1 for c in contributions if c["available"])

        # Determine confidence level
        if available_channels >= 4:
            confidence_level = "high"
        elif available_channels >= 2:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Generate key insights
        key_insights = self._generate_insights(target, contributions, total_weighted_score)

        return {
            "target": target,
            "contributions": contributions,
            "evidence_refs": clickable_evidence,
            "total_weighted_score": total_weighted_score,
            "confidence_level": confidence_level,
            "key_insights": key_insights
        }

    def _generate_insights(self, target: str, contributions: List[Dict],
                           total_score: float) -> List[str]:
        """Generate key biological insights from scoring results."""
        insights = []

        # Overall assessment
        if total_score >= 0.7:
            insights.append(f"{target} is a high-priority target with strong multi-modal evidence")
        elif total_score >= 0.5:
            insights.append(f"{target} shows promising characteristics across multiple channels")
        else:
            insights.append(f"{target} may require alternative targeting approaches")

        # Channel-specific insights
        top_contrib = contributions[0] if contributions else None
        if top_contrib and top_contrib["available"]:
            channel_name = top_contrib["channel"].replace("_", " ").title()
            insights.append(f"Strongest evidence comes from {channel_name} analysis")

        # Safety assessment
        safety_contrib = next((c for c in contributions if c["channel"] == "safety"), None)
        if safety_contrib and safety_contrib["available"]:
            safety_score = safety_contrib["score"]
            if safety_score < 0.3:
                insights.append("Good safety profile with low off-tissue risks")
            elif safety_score > 0.7:
                insights.append("Safety considerations warrant careful evaluation")

        return insights


class RankingAnalyzer:
    """Enhanced analyzer for ranking changes between weight configurations."""

    def compute_rank_impact(self, target_scores: List[TargetScore],
                            current_weights: Dict[str, float]) -> List[Dict]:
        """Compute comprehensive ranking impact analysis."""

        # Current rankings (with current weights)
        current_ranking = {
            ts.target: i + 1 for i, ts in enumerate(
                sorted(target_scores, key=lambda x: x.total_score, reverse=True)
            )
        }

        # Recompute scores with default weights
        default_scores = []
        for ts in target_scores:
            default_score = self._recompute_with_default_weights(ts)
            default_scores.append({
                "target": ts.target,
                "score": default_score,
                "original_score": ts.total_score
            })

        # Default rankings (with default weights)
        default_ranking = {
            item["target"]: i + 1 for i, item in enumerate(
                sorted(default_scores, key=lambda x: x["score"], reverse=True)
            )
        }

        # Compute comprehensive deltas
        rank_impact = []
        for ts in target_scores:
            target = ts.target
            current_rank = current_ranking[target]
            default_rank = default_ranking[target]
            delta = default_rank - current_rank  # Positive = moved up, negative = moved down

            # Calculate score change
            original_score = ts.total_score
            default_item = next(item for item in default_scores if item["target"] == target)
            default_score = default_item["score"]
            score_change = original_score - default_score

            rank_impact.append({
                "target": target,
                "rank_baseline": default_rank,
                "rank_current": current_rank,
                "delta": delta,
                "movement": "up" if delta > 0 else "down" if delta < 0 else "unchanged",
                "score_change": score_change
            })

        return rank_impact

    def _recompute_with_default_weights(self, target_score: TargetScore) -> float:
        """Recompute score using default weights."""
        breakdown = target_score.breakdown

        # Extract channel scores safely
        channel_scores = {
            "genetics": breakdown.genetics,
            "ppi": breakdown.ppi_proximity,
            "pathway": breakdown.pathway_enrichment,
            "safety": breakdown.safety_off_tissue,
            "modality_fit": None
        }

        # Handle modality_fit score extraction
        if breakdown.modality_fit:
            if isinstance(breakdown.modality_fit, dict):
                channel_scores["modality_fit"] = breakdown.modality_fit.get("overall_druggability", 0.0)
            elif hasattr(breakdown.modality_fit, 'overall_druggability'):
                channel_scores["modality_fit"] = breakdown.modality_fit.overall_druggability
            else:
                channel_scores["modality_fit"] = float(breakdown.modality_fit)

        # Compute weighted score with default weights
        total_score = 0.0
        total_weight = 0.0

        for channel, weight in DEFAULT_WEIGHTS.items():
            if channel in channel_scores and channel_scores[channel] is not None:
                score = float(channel_scores[channel])

                # Handle safety inversion
                if channel == "safety":
                    score = 1.0 - score

                total_score += score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.1


class VersionManager:
    """Enhanced version and cache metadata manager."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset for new scoring batch."""
        self.versions = {
            "opentargets": "2024.06",
            "string": "v12.0",
            "reactome": "2024",
            "vantai": "proprietary_v2.0"
        }
        self.cache_stats = {
            "ot_calls": [],
            "total_calls": 0,
            "cached_calls": 0
        }

    def add_ot_call(self, cached: bool, fetch_ms: float, release: str = None):
        """Record an Open Targets API call with enhanced tracking."""
        self.cache_stats["ot_calls"].append((cached, fetch_ms))
        self.cache_stats["total_calls"] += 1
        if cached:
            self.cache_stats["cached_calls"] += 1

        if release and release != "unknown":
            # Clean up version string
            clean_release = release.replace("OT-", "").replace("OpenTargets:", "")
            self.versions["opentargets"] = clean_release

    def set_string_version(self, version: str):
        """Set STRING database version."""
        if version and version != "unknown":
            self.versions["string"] = version

    def set_reactome_version(self, version: str):
        """Set Reactome database version."""
        if version and version != "unknown":
            self.versions["reactome"] = version

    def get_data_version(self) -> str:
        """Build comprehensive data version string."""
        ot_ver = self.versions["opentargets"]
        string_ver = self.versions["string"]
        reactome_ver = self.versions["reactome"]

        return f"OT-{ot_ver} | STRING-{string_ver} | Reactome-{reactome_ver}"

    def get_cache_metadata(self) -> Dict:
        """Get comprehensive cache metadata."""
        if not self.cache_stats["ot_calls"]:
            return {
                "cached": False,
                "fetch_ms": 0.0,
                "cache_hit_rate": 0.0,
                "total_calls": 0
            }

        fetch_times = [fetch_ms for _, fetch_ms in self.cache_stats["ot_calls"]]
        cached_calls = [cached for cached, _ in self.cache_stats["ot_calls"]]

        return {
            "cached": any(cached_calls),
            "fetch_ms": sum(fetch_times),
            "cache_hit_rate": self.cache_stats["cached_calls"] / self.cache_stats["total_calls"],
            "total_calls": self.cache_stats["total_calls"],
            "min_fetch_ms": min(fetch_times) if fetch_times else 0.0,
            "max_fetch_ms": max(fetch_times) if fetch_times else 0.0,
            "avg_fetch_ms": sum(fetch_times) / len(fetch_times) if fetch_times else 0.0
        }


class TargetScorer:
    """Enhanced target scoring class with comprehensive explanation generation."""

    def __init__(self):
        self.default_weights = DEFAULT_WEIGHTS

    async def score_single_target(
        self,
        disease: str,
        target: str,
        weights: Dict[str, float],
        targets_context: List[str] = None,
        version_manager: VersionManager = None
    ) -> TargetScore:
        """Score a single target with comprehensive explanation generation."""

        breakdown = TargetBreakdown()
        all_evidence_refs: List[str] = []
        channel_scores: Dict[str, float] = {}
        warnings: List[str] = []

        try:
            ot_data = await fetch_ot_association(disease, target)
        except Exception as e:
            logger.warning(f"OpenTargets fetch failed for {target}: {e}")
            ot_data = {"release": "unknown", "cached": False}

            # Version metadata toplayıcıya bildir
            if version_manager:
                version_manager.add_ot_call(
                    cached=bool(ot_data.get("cached", False)),
                    fetch_ms=float(ot_data.get("fetch_ms", 0.0)),
                    release=str(ot_data.get("release", "OT-unknown"))
                )

            # 2) Genetics channel
            # Önce mevcut (eski şema ile çalışan) hesaplayıcıyı dene; şema uymazsa compact fallback'e geç
            genetics_score: float = 0.0
            genetics_refs: List[str] = []
            try:
                g_score, g_refs = compute_genetics_score(disease, target, ot_data)  # eski şema ile uyumluysa çalışır
                genetics_score = float(g_score or 0.0)
                genetics_refs = list(g_refs or [])
                # Eğer hesaplayıcı 0 döndürdüyse ve elimizde compact skor varsa, minimal fallback uygula
                if genetics_score == 0.0 and isinstance(ot_data, dict) and "genetics" in ot_data:
                    genetics_score = float(ot_data.get("genetics") or 0.0)
                    genetics_refs.append(f"OT_genetics:{genetics_score:.3f}")
                    warnings.append("Genetics fallback: compact OT score was used")
            except Exception as ex:
                # Tamamen fallback
                genetics_score = float(ot_data.get("genetics", 0.0)) if isinstance(ot_data, dict) else 0.0
                genetics_refs.append(f"OpenTargets:genetics={genetics_score:.3f}")
                warnings.append(f"Genetics enhanced scorer failed, fallback used ({str(ex)[:60]})")

            # OT metadata rozetlerini ekle
            if isinstance(ot_data, dict):
                if "release" in ot_data: genetics_refs.append(f"OpenTargets:{ot_data['release']}")  # örn 2025.06
                if "cached" in ot_data: genetics_refs.append(
                    "OpenTargets:cache_hit" if ot_data["cached"] else "OpenTargets:cache_miss")
                if "evidence_count" in ot_data: genetics_refs.append(
                    f"OpenTargets:evidence={int(ot_data['evidence_count'])}")

            breakdown.genetics = genetics_score
            channel_scores["genetics"] = genetics_score
            all_evidence_refs.extend(genetics_refs)

            # 3) PPI proximity (RWR/centrality karışımı — sizin compute_ppi_proximity bunu hallediyor)
            #disease_genes = targets_context or []
            #ppi_score, ppi_refs = compute_ppi_proximity(target, disease_genes=disease_genes, rwr_enabled=True)
            disease_genes = (targets_context or [])
            ppi_score, ppi_refs = compute_ppi_proximity(
                target,
                disease_genes=disease_genes,
                rwr_enabled=True
            )
            breakdown.ppi_proximity = float(ppi_score or 0.0)
            channel_scores["ppi"] = breakdown.ppi_proximity
            all_evidence_refs.extend(ppi_refs or [])

            # 4) Pathway enrichment
            pathway_score, pathway_refs = compute_pathway_enrichment(target, targets_context)
            breakdown.pathway_enrichment = float(pathway_score or 0.0)
            channel_scores["pathway"] = breakdown.pathway_enrichment
            all_evidence_refs.extend(pathway_refs or [])

            # 5) Safety (penalty: düşük daha iyi)
            try:
                safety_score, safety_refs = compute_safety_penalty(target, ot_data)
                breakdown.safety_off_tissue = float(safety_score if safety_score is not None else 0.5)
                channel_scores["safety"] = breakdown.safety_off_tissue
                all_evidence_refs.extend(safety_refs or [])
            except Exception as e:
                logger.warning(f"Safety scoring failed for {target}: {e}")
                breakdown.safety_off_tissue = 0.5
                channel_scores["safety"] = 0.5
                warnings.append(f"Safety scoring unavailable: {str(e)[:50]}")

            # 6) Modality fit (alt-bileşenler + overall)
            modality_scores, modality_refs = compute_modality_fit(target, ppi_network.graph)
            if isinstance(modality_scores, dict):
                breakdown.modality_fit = modality_scores
                modality_fit_score = float(modality_scores.get("overall_druggability", 0.0))
            else:
                # Eski tip objeyse, .get ile erişilebilen dict'e dönüştürmeye çalış
                try:
                    modality_fit_score = float(getattr(modality_scores, "overall_druggability", 0.0))
                    breakdown.modality_fit = {
                        "overall_druggability": modality_fit_score,
                        "protac_degrader": float(getattr(modality_scores, "protac_degrader", 0.0)),
                        "small_molecule": float(getattr(modality_scores, "small_molecule", 0.0)),
                        "e3_coexpr": float(getattr(modality_scores, "e3_coexpr", 0.0)),
                        "ternary_proxy": float(getattr(modality_scores, "ternary_proxy", 0.0)),
                        "ppi_hotspot": float(getattr(modality_scores, "ppi_hotspot", 0.0)),
                    }
                except Exception:
                    modality_fit_score = 0.0
                    breakdown.modality_fit = {"overall_druggability": 0.0}
            channel_scores["modality_fit"] = modality_fit_score
            all_evidence_refs.extend(modality_refs or [])

            # 7) Toplam skor
            total_score = self._compute_weighted_score(channel_scores, weights)

            # 8) Açıklama nesnesi
            explanation_builder = ExplanationBuilder(weights)
            explanation = explanation_builder.build_explanation(target, channel_scores, all_evidence_refs)

            # 9) Data version string (VersionManager kullan; yoksa sağlam default)
            data_version = (
                version_manager.get_data_version()
                if version_manager else "VantAI:proprietary | OT-unknown | STRING-v12.0 | Reactome-2024"
            )

            return TargetScore(
                target=target,
                total_score=float(total_score),
                breakdown=breakdown,
                evidence_refs=list(dict.fromkeys(all_evidence_refs)),  # uniq sırayı koru
                data_version=data_version,
                explanation=explanation,
                timestamp=datetime.now(),
                warnings=warnings or None
            )

        except Exception as e:
            logger.error(f"Critical error scoring target {target}: {e}")

            error_explanation = {
                "target": target,
                "contributions": [
                    {
                        "channel": ch,
                        "weight": wt,
                        "score": None,
                        "contribution": 0.0,
                        "available": False
                    } for ch, wt in (weights or {}).items()
                ],
                "evidence_refs": [],
                "total_weighted_score": 0.1,
                "confidence_level": "error",
                "key_insights": [f"Scoring failed: {str(e)[:100]}"]
            }

            return TargetScore(
                target=target,
                total_score=0.1,
                breakdown=TargetBreakdown(
                    genetics=0.1,
                    ppi_proximity=0.1,
                    pathway_enrichment=0.1,
                    safety_off_tissue=0.5,
                    modality_fit={
                        "e3_coexpr": 0.1,
                        "ternary_proxy": 0.1,
                        "ppi_hotspot": 0.1,
                        "overall_druggability": 0.1
                    }
                ),
                evidence_refs=[f"Error_scoring_{target}: {str(e)[:100]}"],
                data_version="Error_state",
                explanation=error_explanation,
                timestamp=datetime.now(),
                warnings=[f"Critical scoring error: {str(e)}"]
            )

    def _compute_weighted_score(self, channel_scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted total score with enhanced error handling."""
        total_score = 0.0
        total_weight = 0.0

        for channel, weight in (weights or {}).items():
            if channel in channel_scores and channel_scores[channel] is not None:
                score = float(channel_scores[channel])

                # Safety (penalty → invert)
                if channel == "safety":
                    score = 1.0 - score

                # Clamp
                score = max(0.0, min(1.0, score))

                total_score += score * float(weight)
                total_weight += float(weight)

        if total_weight > 0:
            normalized_score = total_score / total_weight
        else:
            logger.warning("No valid channel scores available; returning fallback score")
            normalized_score = 0.1

        return max(0.0, min(1.0, normalized_score))

    async def score_targets_batch(self, request: ScoreRequest) -> Tuple[List[TargetScore], Dict]:
        """Score multiple targets with comprehensive analysis."""
        if not request.targets:
            return [], {"data_version": "No_targets", "meta": {"cached": False, "fetch_ms": 0.0}}

        version_manager = VersionManager()
        final_weights = {**self.default_weights, **(request.weights or {})}

        # Paralel skorla
        tasks = [
            self.score_single_target(
                request.disease,
                t,
                final_weights,
                targets_context=request.targets,
                version_manager=version_manager
            )
            for t in request.targets
        ]
        target_scores = await asyncio.gather(*tasks)

        # Ranking impact
        ranking_analyzer = RankingAnalyzer()
        rank_impact = ranking_analyzer.compute_rank_impact(target_scores, final_weights)

        # Metadata
        metadata = {
            "data_version": version_manager.get_data_version(),
            "meta": version_manager.get_cache_metadata(),
            "rank_impact": rank_impact,
            "system_info": {
                "scoring_channels": len(final_weights),
                "targets_processed": len(target_scores),
                "successful_scores": sum(1 for ts in target_scores if ts.total_score > 0.1),
            },
        }

        return target_scores, metadata

    def validate_request(self, request: ScoreRequest) -> Tuple[bool, str]:
        """Enhanced request validation."""
        if not request.targets:
            return False, "No targets provided"
        if len(request.targets) > 50:
            return False, "Too many targets (max 50)"
        if not request.disease:
            return False, "Disease identifier required"

        # Weights
        for channel, weight in (request.weights or {}).items():
            if not isinstance(weight, (int, float)):
                return False, f"Weight for {channel} must be numeric"
            if not 0 <= float(weight) <= 1:
                return False, f"Weight for {channel} must be between 0 and 1"

        weight_sum = sum(float(w) for w in (request.weights or {}).values())
        if weight_sum > 1.2 or weight_sum < 0.8:
            return False, f"Weights should sum to ~1.0 (current sum: {weight_sum:.2f})"

        # Targets
        for t in request.targets:
            if not isinstance(t, str) or len(t.strip()) < 2:
                return False, f"Invalid target identifier: {t}"

        return True, ""


# Global scorer instance
target_scorer = TargetScorer()


async def score_targets(request: ScoreRequest) -> Tuple[List[TargetScore], Dict]:
    """Main scoring function for API endpoints."""
    return await target_scorer.score_targets_batch(request)


def validate_score_request(request: ScoreRequest) -> Tuple[bool, str]:
    """Validate scoring request."""
    return target_scorer.validate_request(request)