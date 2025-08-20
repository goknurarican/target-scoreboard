"""
Complete scoring implementation with fixed explanation generation and proper error handling.
"""
from typing import List, Dict, Tuple, Optional
import asyncio
from datetime import datetime
import logging
import numpy as np
from typing import Dict, List
from scipy.stats import dirichlet, kendalltau
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


def simulate_weight_perturbations(
        target_scores: List[TargetScore],
        base_weights: Dict[str, float],
        n_samples: int = 200,
        dirichlet_alpha: float = 80.0
) -> Dict:
    """
    Simulate weight uncertainty and compute rank stability metrics.

    Uses Dirichlet distribution to sample weight configurations around base weights,
    then analyzes ranking stability across perturbations.

    Args:
        target_scores: List of scored targets with breakdown information
        base_weights: Base weight configuration to perturb around
        n_samples: Number of weight samples to generate
        dirichlet_alpha: Concentration parameter for Dirichlet (higher = less variation)

    Returns:
        Dict containing:
        - stability: Per-target rank statistics (mode_rank, entropy, histogram)
        - kendall_tau_mean: Average rank correlation across samples
        - samples: Number of samples processed
        - weight_stats: Weight variation statistics
    """
    if not target_scores:
        return {
            "stability": {},
            "kendall_tau_mean": 0.0,
            "samples": 0,
            "weight_stats": {}
        }

    try:
        # Channel ordering for consistent indexing
        channels = ["genetics", "ppi", "pathway", "safety", "modality_fit"]

        # Extract base weights in channel order
        base_weight_vector = np.array([base_weights.get(ch, 0.0) for ch in channels])

        # Generate Dirichlet-sampled weight configurations
        # Scale base weights by alpha for concentration parameter
        alpha_vector = base_weight_vector * dirichlet_alpha
        alpha_vector = np.maximum(alpha_vector, 0.1)  # Prevent zero alphas

        sampled_weights = dirichlet.rvs(alpha_vector, size=n_samples)

        # Extract channel scores for each target
        target_channel_scores = []
        target_names = []

        for ts in target_scores:
            target_names.append(ts.target)
            breakdown = ts.breakdown

            # Extract scores safely with fallbacks
            scores = {
                "genetics": float(breakdown.genetics or 0.0),
                "ppi": float(breakdown.ppi_proximity or 0.0),
                "pathway": float(breakdown.pathway_enrichment or 0.0),
                "safety": float(breakdown.safety_off_tissue or 0.0),
                "modality_fit": 0.0
            }

            # Handle modality_fit extraction
            if breakdown.modality_fit:
                if isinstance(breakdown.modality_fit, dict):
                    scores["modality_fit"] = float(breakdown.modality_fit.get("overall_druggability", 0.0))
                else:
                    scores["modality_fit"] = float(getattr(breakdown.modality_fit, "overall_druggability", 0.0))

            # Convert to ordered array
            score_vector = np.array([scores[ch] for ch in channels])
            target_channel_scores.append(score_vector)

        target_channel_scores = np.array(target_channel_scores)  # Shape: (n_targets, n_channels)

        # Simulate rankings across weight samples
        all_rankings = []
        rank_matrices = np.zeros((len(target_names), n_samples), dtype=int)

        for sample_idx in range(n_samples):
            weight_sample = sampled_weights[sample_idx]

            # Compute scores for this weight configuration
            sample_scores = []
            for target_idx in range(len(target_names)):
                target_scores_vec = target_channel_scores[target_idx]

                # Apply safety inversion (safety is penalty)
                adjusted_scores = target_scores_vec.copy()
                safety_idx = channels.index("safety")
                adjusted_scores[safety_idx] = 1.0 - adjusted_scores[safety_idx]

                # Compute weighted score
                weighted_score = np.dot(adjusted_scores, weight_sample)
                sample_scores.append(weighted_score)

            # Rank targets (1-indexed, highest score = rank 1)
            sample_ranks = len(sample_scores) + 1 - np.argsort(np.argsort(sample_scores))
            rank_matrices[:, sample_idx] = sample_ranks
            all_rankings.append(sample_ranks)

        # Compute per-target stability metrics
        stability_results = {}
        for target_idx, target_name in enumerate(target_names):
            target_ranks = rank_matrices[target_idx, :]

            # Rank histogram
            unique_ranks, counts = np.unique(target_ranks, return_counts=True)
            histogram = {int(rank): int(count) for rank, count in zip(unique_ranks, counts)}

            # Mode rank (most frequent)
            mode_rank = int(unique_ranks[np.argmax(counts)])

            # Rank entropy (normalized)
            probabilities = counts / n_samples
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
            max_entropy = np.log2(len(target_names))  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            stability_results[target_name] = {
                "mode_rank": mode_rank,
                "entropy": float(normalized_entropy),
                "histogram": histogram,
                "rank_range": [int(np.min(target_ranks)), int(np.max(target_ranks))],
                "rank_std": float(np.std(target_ranks))
            }

        # Compute overall ranking agreement (Kendall's tau)
        kendall_taus = []
        base_ranking = rank_matrices[:, 0]  # Use first sample as reference

        for sample_idx in range(1, min(n_samples, 50)):  # Sample subset for efficiency
            sample_ranking = rank_matrices[:, sample_idx]
            try:
                tau, _ = kendalltau(base_ranking, sample_ranking)
                if not np.isnan(tau):
                    kendall_taus.append(tau)
            except Exception:
                continue

        kendall_tau_mean = float(np.mean(kendall_taus)) if kendall_taus else 0.0

        # Weight variation statistics
        weight_stats = {}
        for ch_idx, channel in enumerate(channels):
            channel_weights = sampled_weights[:, ch_idx]
            weight_stats[channel] = {
                "mean": float(np.mean(channel_weights)),
                "std": float(np.std(channel_weights)),
                "min": float(np.min(channel_weights)),
                "max": float(np.max(channel_weights)),
                "base": float(base_weight_vector[ch_idx])
            }

        return {
            "stability": stability_results,
            "kendall_tau_mean": kendall_tau_mean,
            "samples": n_samples,
            "weight_stats": weight_stats,
            "meta": {
                "dirichlet_alpha": dirichlet_alpha,
                "channels": channels,
                "successful_kendall_comparisons": len(kendall_taus)
            }
        }

    except Exception as e:
        logger.error(f"Weight perturbation simulation failed: {e}")
        return {
            "stability": {},
            "kendall_tau_mean": 0.0,
            "samples": 0,
            "error": str(e)
        }


"""
Channel ablation analysis function for app/scoring.py
Add this after the RankingAnalyzer class.
"""


def compute_channel_ablation(target_scores: List[TargetScore], weights: Dict[str, float]) -> List[Dict]:
    """
    Compute channel ablation analysis by removing each channel and measuring impact.

    For each channel, sets its weight to 0 and proportionally normalizes remaining
    weights to maintain sum=1.0, then recomputes scores to measure drop.

    Args:
        target_scores: List of scored targets with breakdown information
        weights: Current weight configuration

    Returns:
        List of ablation results per channel with score drops and rank changes
    """
    if not target_scores or not weights:
        return []

    try:
        channels = ["genetics", "ppi", "pathway", "safety", "modality_fit"]

        # Baseline rankings (current scores)
        baseline_scores = {}
        baseline_rankings = {}
        sorted_targets = sorted(target_scores, key=lambda x: x.total_score, reverse=True)

        for i, ts in enumerate(sorted_targets):
            baseline_scores[ts.target] = ts.total_score
            baseline_rankings[ts.target] = i + 1

        ablation_results = []

        for ablated_channel in channels:
            # Create ablated weight configuration
            ablated_weights = weights.copy()
            ablated_weights[ablated_channel] = 0.0

            # Normalize remaining weights to sum to 1.0
            remaining_weight = sum(w for ch, w in ablated_weights.items() if w > 0)
            if remaining_weight > 0:
                for ch in ablated_weights:
                    if ablated_weights[ch] > 0:
                        ablated_weights[ch] = ablated_weights[ch] / remaining_weight
            else:
                # All weights were zero, skip this ablation
                logger.warning(f"Cannot ablate {ablated_channel}: no remaining positive weights")
                continue

            # Recompute scores with ablated weights
            ablated_target_scores = []
            channel_deltas = []

            for ts in target_scores:
                breakdown = ts.breakdown

                # Extract channel scores safely
                channel_scores = {
                    "genetics": float(breakdown.genetics or 0.0),
                    "ppi": float(breakdown.ppi_proximity or 0.0),
                    "pathway": float(breakdown.pathway_enrichment or 0.0),
                    "safety": float(breakdown.safety_off_tissue or 0.0),
                    "modality_fit": 0.0
                }

                # Handle modality_fit extraction
                if breakdown.modality_fit:
                    if isinstance(breakdown.modality_fit, dict):
                        channel_scores["modality_fit"] = float(breakdown.modality_fit.get("overall_druggability", 0.0))
                    else:
                        channel_scores["modality_fit"] = float(
                            getattr(breakdown.modality_fit, "overall_druggability", 0.0))

                # Compute ablated score
                ablated_score = 0.0
                total_weight_used = 0.0

                for channel, weight in ablated_weights.items():
                    if weight > 0 and channel in channel_scores:
                        score = channel_scores[channel]

                        # Handle safety inversion
                        if channel == "safety":
                            score = 1.0 - score

                        # Ensure bounds
                        score = max(0.0, min(1.0, score))
                        ablated_score += score * weight
                        total_weight_used += weight

                # Normalize if needed
                if total_weight_used > 0:
                    ablated_score = ablated_score / total_weight_used
                else:
                    ablated_score = 0.1  # Fallback

                ablated_target_scores.append({
                    "target": ts.target,
                    "original_score": baseline_scores[ts.target],
                    "ablated_score": ablated_score
                })

            # Compute new rankings with ablated scores
            ablated_sorted = sorted(ablated_target_scores, key=lambda x: x["ablated_score"], reverse=True)
            ablated_rankings = {item["target"]: i + 1 for i, item in enumerate(ablated_sorted)}

            # Calculate deltas for each target
            for item in ablated_target_scores:
                target = item["target"]
                original_score = item["original_score"]
                ablated_score = item["ablated_score"]

                score_drop = original_score - ablated_score
                original_rank = baseline_rankings[target]
                ablated_rank = ablated_rankings[target]
                rank_delta = ablated_rank - original_rank  # Positive = rank got worse

                channel_deltas.append({
                    "target": target,
                    "score_drop": float(score_drop),
                    "rank_delta": int(rank_delta),
                    "original_score": float(original_score),
                    "ablated_score": float(ablated_score),
                    "original_rank": int(original_rank),
                    "ablated_rank": int(ablated_rank)
                })

            # Sort by score drop (highest impact first)
            channel_deltas.sort(key=lambda x: x["score_drop"], reverse=True)

            ablation_results.append({
                "channel": ablated_channel,
                "ablated_weights": ablated_weights,
                "delta": channel_deltas,
                "avg_score_drop": float(sum(d["score_drop"] for d in channel_deltas) / len(channel_deltas)),
                "max_score_drop": float(max(d["score_drop"] for d in channel_deltas)) if channel_deltas else 0.0,
                "targets_affected": int(sum(1 for d in channel_deltas if d["score_drop"] > 0.01))
            })

        # Sort ablation results by average impact
        ablation_results.sort(key=lambda x: x["avg_score_drop"], reverse=True)

        return ablation_results

    except Exception as e:
        logger.error(f"Channel ablation analysis failed: {e}")
        return []
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