# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.


"""
Modality fit channel with explicit subscores for PROTAC/degrader suitability.
"""
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModalityAnalyzer:
    """Analyzer for modality-specific druggability scores."""

    def __init__(self):
        self.expression_data = {}
        self.ternary_data = {}
        self.hotspot_data = {}
        self.e3_ligases = {'CRBN', 'VHL', 'DDB1', 'DCAF1', 'DCAF4', 'DCAF7', 'DCAF8', 'DCAF11', 'DCAF15'}
        self._load_data()

    def _load_data(self):
        """Load demo data files or create synthetic data."""
        self._load_expression_data()
        self._load_ternary_data()
        self._load_hotspot_data()

    def _load_expression_data(self):
        """Load expression data for E3 co-expression analysis."""
        expression_file = Path("data_demo/expression_demo.csv")

        if expression_file.exists():
            try:
                df = pd.read_csv(expression_file)
                # Organize by tissue and gene
                for _, row in df.iterrows():
                    tissue = row['tissue']
                    gene = row['gene']
                    tpm = row['tpm']

                    if tissue not in self.expression_data:
                        self.expression_data[tissue] = {}
                    self.expression_data[tissue][gene] = tpm

                logger.info(f"Loaded expression data from file: {len(self.expression_data)} tissues")
                return

            except Exception as e:
                logger.error(f"Error loading expression data: {e}")

        # Create enhanced demo expression data
        logger.info("Creating enhanced demo expression data")
        self._create_enhanced_expression_data()

    def _create_enhanced_expression_data(self):
        """Create comprehensive demo expression data with realistic patterns."""
        # Enhanced expression data with more tissues and realistic TPM values
        self.expression_data = {
            'lung': {
                'EGFR': 245.8, 'CRBN': 12.4, 'VHL': 34.2, 'DDB1': 28.9, 'DCAF1': 15.6,
                'KRAS': 89.3, 'ERBB2': 123.7, 'MET': 67.2, 'ALK': 8.9, 'BRAF': 45.6,
                'TP53': 156.4, 'PIK3CA': 78.9, 'PTEN': 42.1, 'RB1': 34.5, 'VHL': 34.2
            },
            'breast': {
                'EGFR': 189.2, 'CRBN': 15.8, 'VHL': 29.1, 'DDB1': 31.5, 'DCAF1': 18.2,
                'KRAS': 78.9, 'ERBB2': 287.4, 'MET': 52.3, 'ALK': 4.2, 'BRAF': 38.7,
                'TP53': 203.1, 'PIK3CA': 95.6, 'PTEN': 67.8, 'RB1': 28.9
            },
            'liver': {
                'EGFR': 98.7, 'CRBN': 22.1, 'VHL': 56.3, 'DDB1': 41.2, 'DCAF1': 25.4,
                'KRAS': 134.5, 'ERBB2': 45.8, 'MET': 198.7, 'ALK': 2.1, 'BRAF': 67.4,
                'TP53': 187.3, 'PIK3CA': 112.8, 'PTEN': 89.4
            },
            'brain': {
                'EGFR': 67.8, 'CRBN': 8.9, 'VHL': 18.7, 'DDB1': 19.4, 'DCAF1': 11.3,
                'KRAS': 45.2, 'ERBB2': 23.1, 'MET': 34.6, 'ALK': 12.7, 'BRAF': 89.5,
                'TP53': 145.6, 'PIK3CA': 67.3, 'PTEN': 123.8
            },
            'kidney': {
                'EGFR': 156.3, 'CRBN': 18.5, 'VHL': 198.7, 'DDB1': 35.6, 'DCAF1': 21.8,
                'KRAS': 92.4, 'ERBB2': 78.9, 'MET': 89.4, 'ALK': 5.6, 'BRAF': 34.2,
                'TP53': 178.4, 'PIK3CA': 87.6, 'PTEN': 156.7, 'VHL': 198.7
            }
        }

    def _load_ternary_data(self):
        """Load ternary complex formation data."""
        ternary_file = Path("data_demo/ternary_reports.csv")

        if ternary_file.exists():
            try:
                df = pd.read_csv(ternary_file)
                for _, row in df.iterrows():
                    target = row['target']
                    evidence_level = row['evidence_level']
                    self.ternary_data[target] = evidence_level

                logger.info(f"Loaded ternary data from file: {len(self.ternary_data)} targets")
                return

            except Exception as e:
                logger.error(f"Error loading ternary data: {e}")

        # Create enhanced demo ternary data
        logger.info("Creating enhanced demo ternary data")
        self._create_enhanced_ternary_data()

    def _create_enhanced_ternary_data(self):
        """Create comprehensive demo ternary data based on known biology."""
        # Enhanced ternary data with more targets and realistic evidence levels
        self.ternary_data = {
            # Known PROTAC targets with strong evidence
            'EGFR': 'reported',
            'ERBB2': 'reported',
            'ALK': 'reported',
            'BTK': 'reported',
            'BCL2': 'reported',

            # Targets with weak/moderate evidence
            'KRAS': 'weak',
            'MET': 'weak',
            'BRAF': 'weak',
            'PIK3CA': 'weak',
            'PTEN': 'weak',

            # Challenging targets
            'TP53': 'none',
            'RB1': 'none',
            'MYC': 'none',

            # E3 ligases themselves
            'VHL': 'reported',
            'CRBN': 'reported',
            'MDM2': 'reported'
        }

    def _load_hotspot_data(self):
        """Load PPI hotspot data."""
        hotspot_file = Path("data_demo/hotspot_edges.csv")

        if hotspot_file.exists():
            try:
                df = pd.read_csv(hotspot_file)
                for _, row in df.iterrows():
                    target = row['target']
                    partner = row['partner']
                    hotspot_score = row['hotspot_score']

                    if target not in self.hotspot_data:
                        self.hotspot_data[target] = []
                    self.hotspot_data[target].append({
                        'partner': partner,
                        'score': hotspot_score,
                        'type': row.get('interaction_type', 'binding')
                    })

                logger.info(f"Loaded hotspot data from file: {len(self.hotspot_data)} targets")
                return

            except Exception as e:
                logger.error(f"Error loading hotspot data: {e}")

        # Create enhanced demo hotspot data
        logger.info("Creating enhanced demo hotspot data")
        self._create_enhanced_hotspot_data()

    def _create_enhanced_hotspot_data(self):
        """Create comprehensive demo hotspot data."""
        # Enhanced hotspot data with more realistic protein-protein interaction scores
        self.hotspot_data = {
            'EGFR': [
                {'partner': 'GRB2', 'score': 0.85, 'type': 'binding'},
                {'partner': 'ERBB2', 'score': 0.91, 'type': 'dimerization'},
                {'partner': 'SOS1', 'score': 0.74, 'type': 'complex'},
                {'partner': 'STAT3', 'score': 0.68, 'type': 'activation'}
            ],
            'ERBB2': [
                {'partner': 'EGFR', 'score': 0.91, 'type': 'dimerization'},
                {'partner': 'GRB2', 'score': 0.67, 'type': 'binding'},
                {'partner': 'PIK3CA', 'score': 0.79, 'type': 'activation'}
            ],
            'KRAS': [
                {'partner': 'RAF1', 'score': 0.78, 'type': 'binding'},
                {'partner': 'PIK3CA', 'score': 0.72, 'type': 'activation'},
                {'partner': 'SOS1', 'score': 0.83, 'type': 'complex'}
            ],
            'MET': [
                {'partner': 'GRB2', 'score': 0.76, 'type': 'binding'},
                {'partner': 'SOS1', 'score': 0.69, 'type': 'complex'},
                {'partner': 'PIK3CA', 'score': 0.74, 'type': 'activation'}
            ],
            'ALK': [
                {'partner': 'GRB2', 'score': 0.71, 'type': 'binding'},
                {'partner': 'STAT3', 'score': 0.84, 'type': 'activation'},
                {'partner': 'PIK3CA', 'score': 0.66, 'type': 'pathway'}
            ],
            'BRAF': [
                {'partner': 'RAF1', 'score': 0.73, 'type': 'complex'},
                {'partner': 'MAP2K1', 'score': 0.89, 'type': 'phosphorylation'},
                {'partner': 'KSR1', 'score': 0.65, 'type': 'scaffold'}
            ],
            'TP53': [
                {'partner': 'MDM2', 'score': 0.94, 'type': 'binding'},
                {'partner': 'ATM', 'score': 0.82, 'type': 'phosphorylation'},
                {'partner': 'BRCA1', 'score': 0.76, 'type': 'complex'}
            ],
            'PIK3CA': [
                {'partner': 'AKT1', 'score': 0.87, 'type': 'phosphorylation'},
                {'partner': 'PTEN', 'score': 0.79, 'type': 'antagonism'},
                {'partner': 'MTOR', 'score': 0.71, 'type': 'pathway'}
            ]
        }

    def compute_e3_coexpression(self, target: str) -> float:
        """
        Compute E3 ligase co-expression score using enhanced correlation analysis.

        Args:
            target: Target gene symbol

        Returns:
            Co-expression score (0-1)
        """
        if not self.expression_data:
            return 0.3

        coexpression_scores = []

        for tissue, expression in self.expression_data.items():
            if target not in expression:
                continue

            target_expr = expression[target]
            if target_expr <= 0:
                continue

            # Calculate correlation with E3 ligases
            tissue_scores = []
            for e3_ligase in self.e3_ligases:
                if e3_ligase in expression and expression[e3_ligase] > 0:
                    e3_expr = expression[e3_ligase]

                    # Improved co-expression metric using log-space correlation
                    log_target = np.log2(target_expr + 1)
                    log_e3 = np.log2(e3_expr + 1)

                    # Pearson-like correlation in log space
                    ratio_diff = abs(log_target - log_e3)
                    max_diff = 10  # Expected maximum log difference

                    # Convert to similarity score with sigmoid-like function
                    similarity = 1 / (1 + np.exp((ratio_diff - 4) / 2))
                    tissue_scores.append(similarity)

            if tissue_scores:
                # Weight by number of E3 ligases detected
                tissue_avg = np.mean(tissue_scores)
                # Boost score based on number of co-expressed E3 ligases
                ligase_bonus = min(0.2, len(tissue_scores) * 0.05)
                tissue_final = min(1.0, tissue_avg + ligase_bonus)
                coexpression_scores.append(tissue_final)

        if not coexpression_scores:
            return 0.3

        # Average across tissues with variance penalty
        final_score = np.mean(coexpression_scores)

        # Boost targets that are consistently co-expressed across tissues
        if len(coexpression_scores) > 1:
            consistency_bonus = 1 - (np.std(coexpression_scores) / np.mean(coexpression_scores))
            final_score = min(1.0, final_score * (1 + 0.1 * consistency_bonus))

        return max(0.1, min(1.0, final_score))

    def compute_ternary_proxy(self, target: str) -> float:
        """
        Compute ternary complex formation proxy score with enhanced scoring.

        Args:
            target: Target gene symbol

        Returns:
            Ternary formation score (0-1)
        """
        evidence_level = self.ternary_data.get(target, 'none')

        # Enhanced score mapping with more granular distinctions
        score_mapping = {
            'none': 0.25,
            'weak': 0.55,
            'moderate': 0.75,
            'reported': 0.90,
            'validated': 0.95
        }

        base_score = score_mapping.get(evidence_level, 0.25)

        # Bonus for known drug targets (likely to have better ternary formation)
        drug_target_bonus = 0.0
        known_drug_targets = {'EGFR', 'ERBB2', 'ALK', 'BTK', 'BCL2', 'ABL1'}
        if target in known_drug_targets:
            drug_target_bonus = 0.1

        return min(1.0, base_score + drug_target_bonus)

    def compute_ppi_hotspot(self, target: str, ppi_graph: Optional[nx.Graph] = None) -> float:
        """
        Compute PPI hotspot score with enhanced interface analysis.

        Args:
            target: Target gene symbol
            ppi_graph: PPI network graph (optional)

        Returns:
            Hotspot score (0-1)
        """
        # Use precomputed hotspot data if available
        if target in self.hotspot_data:
            hotspot_interactions = self.hotspot_data[target]

            if hotspot_interactions:
                # Weighted scoring based on interaction types
                type_weights = {
                    'binding': 1.0,
                    'dimerization': 1.2,
                    'complex': 1.1,
                    'activation': 0.9,
                    'phosphorylation': 0.95
                }

                weighted_scores = []
                for interaction in hotspot_interactions:
                    base_score = interaction['score']
                    interaction_type = interaction.get('type', 'binding')
                    weight = type_weights.get(interaction_type, 1.0)
                    weighted_scores.append(base_score * weight)

                # Use top 3 interactions for final score
                top_scores = sorted(weighted_scores, reverse=True)[:3]
                final_score = np.mean(top_scores) if top_scores else 0.2

                # Bonus for having multiple high-quality interactions
                if len(top_scores) >= 3:
                    final_score = min(1.0, final_score * 1.1)

                return max(0.1, min(1.0, final_score))

        # Fall back to PPI degree-based heuristic with enhanced scoring
        if ppi_graph and target in ppi_graph:
            degree = ppi_graph.degree(target)

            # Enhanced degree-based scoring
            if degree >= 10:
                score = 0.8  # High degree hub
            elif degree >= 5:
                score = 0.6  # Medium hub
            elif degree >= 2:
                score = 0.4  # Connected
            else:
                score = 0.2  # Poorly connected

            # Adjust based on neighbor quality (if they are also hubs)
            neighbors = list(ppi_graph.neighbors(target))
            neighbor_degrees = [ppi_graph.degree(n) for n in neighbors]
            if neighbor_degrees:
                avg_neighbor_degree = np.mean(neighbor_degrees)
                if avg_neighbor_degree > 5:
                    score = min(1.0, score * 1.2)  # Connected to important nodes

            return score

        return 0.2  # Default low score

    def compute_modality_scores(self, target: str, ppi_graph: Optional[nx.Graph] = None) -> Dict[str, float]:
        """
        Compute all modality subscores for a target with enhanced algorithms.

        Args:
            target: Target gene symbol
            ppi_graph: PPI network graph

        Returns:
            Dict with subscore and overall scores
        """
        # Compute enhanced subscores
        e3_coexpr = self.compute_e3_coexpression(target)
        ternary_proxy = self.compute_ternary_proxy(target)
        ppi_hotspot = self.compute_ppi_hotspot(target, ppi_graph)

        # Enhanced overall druggability with adaptive weighting
        overall_druggability = (
                0.4 * e3_coexpr +
                0.35 * ternary_proxy +
                0.25 * ppi_hotspot
        )

        # PROTAC/degrader score (emphasizes E3 and ternary)
        protac_degrader = (
                0.5 * e3_coexpr +
                0.4 * ternary_proxy +
                0.1 * ppi_hotspot
        )

        # Small molecule score (emphasizes hotspots)
        small_molecule = (
                0.15 * e3_coexpr +
                0.15 * ternary_proxy +
                0.7 * ppi_hotspot
        )

        # Molecular glue score (balanced approach)
        molecular_glue = (
                0.3 * e3_coexpr +
                0.5 * ternary_proxy +
                0.2 * ppi_hotspot
        )

        # Antibody-drug conjugate score (different weighting)
        adc_score = (
                0.1 * e3_coexpr +
                0.2 * ternary_proxy +
                0.7 * ppi_hotspot
        )

        return {
            "e3_coexpr": e3_coexpr,
            "ternary_proxy": ternary_proxy,
            "ppi_hotspot": ppi_hotspot,
            "overall_druggability": overall_druggability,
            "protac_degrader": protac_degrader,
            "small_molecule": small_molecule,
            "molecular_glue": molecular_glue,
            "adc_score": adc_score
        }


def get_modality_recommendations(target: str, ppi_graph: Optional[nx.Graph] = None) -> Dict:
    """
    Get detailed modality recommendations for a target with enhanced analysis.

    Args:
        target: Target gene symbol
        ppi_graph: PPI network graph

    Returns:
        Dict with recommendations and detailed analysis
    """
    modality_scores, _ = compute_modality_fit(target, ppi_graph)

    recommendations = {
        "target": target,
        "overall_druggability": modality_scores["overall_druggability"],
        "subscores": {
            "e3_coexpr": modality_scores["e3_coexpr"],
            "ternary_proxy": modality_scores["ternary_proxy"],
            "ppi_hotspot": modality_scores["ppi_hotspot"]
        },
        "modality_scores": {
            "protac_degrader": modality_scores["protac_degrader"],
            "small_molecule": modality_scores["small_molecule"],
            "molecular_glue": modality_scores["molecular_glue"],
            "adc_score": modality_scores["adc_score"]
        },
        "recommendations": []
    }

    # Enhanced recommendation logic
    if modality_scores["protac_degrader"] > 0.7:
        recommendations["recommendations"].append({
            "modality": "PROTAC/Degrader",
            "priority": "High",
            "confidence": "Strong",
            "rationale": "Excellent E3 ligase co-expression and strong ternary complex evidence"
        })
    elif modality_scores["protac_degrader"] > 0.5:
        recommendations["recommendations"].append({
            "modality": "PROTAC/Degrader",
            "priority": "Medium",
            "confidence": "Moderate",
            "rationale": "Good degrader potential, optimization may be needed"
        })

    if modality_scores["small_molecule"] > 0.6:
        recommendations["recommendations"].append({
            "modality": "Small Molecule",
            "priority": "High",
            "confidence": "Strong",
            "rationale": "Strong PPI hotspot score indicates druggable binding sites"
        })
    elif modality_scores["small_molecule"] > 0.4:
        recommendations["recommendations"].append({
            "modality": "Small Molecule",
            "priority": "Medium",
            "confidence": "Moderate",
            "rationale": "Moderate binding site druggability"
        })

    if modality_scores["molecular_glue"] > 0.6:
        recommendations["recommendations"].append({
            "modality": "Molecular Glue",
            "priority": "Medium",
            "confidence": "Emerging",
            "rationale": "Good ternary complex formation potential for glue development"
        })

    if modality_scores["adc_score"] > 0.5:
        recommendations["recommendations"].append({
            "modality": "Antibody-Drug Conjugate",
            "priority": "Consider",
            "confidence": "Alternative",
            "rationale": "Surface accessibility may enable ADC targeting"
        })

    if modality_scores["overall_druggability"] < 0.4:
        recommendations["recommendations"].append({
            "modality": "Alternative Approaches",
            "priority": "Required",
            "confidence": "Limited",
            "rationale": "Low conventional druggability suggests need for novel modalities"
        })

    return recommendations


def get_modality_data_summary() -> Dict:
    """Get comprehensive summary of modality data sources."""
    return {
        "expression_tissues": len(modality_analyzer.expression_data),
        "expression_genes": len(set().union(*[genes.keys() for genes in modality_analyzer.expression_data.values()])),
        "ternary_targets": len(modality_analyzer.ternary_data),
        "hotspot_targets": len(modality_analyzer.hotspot_data),
        "total_hotspot_interactions": sum(len(interactions) for interactions in modality_analyzer.hotspot_data.values()),
        "e3_ligases_tracked": len(modality_analyzer.e3_ligases),
        "data_coverage": {
            "high_confidence_ternary": len([t for t, level in modality_analyzer.ternary_data.items() if level == 'reported']),
            "targets_with_hotspots": len(modality_analyzer.hotspot_data),
            "multi_tissue_expression": len([
                gene for gene in set().union(*[genes.keys() for genes in modality_analyzer.expression_data.values()])
                if sum(1 for tissue_data in modality_analyzer.expression_data.values() if gene in tissue_data) > 1
            ])
        }
    }


# Global analyzer instance
modality_analyzer = ModalityAnalyzer()


def compute_modality_fit(target: str, ppi_graph: Optional[nx.Graph] = None) -> Tuple[Dict[str, float], List[str]]:
    """
    Compute modality fit scores for a target with comprehensive evidence tracking.

    Args:
        target: Target gene symbol
        ppi_graph: PPI network graph

    Returns:
        (modality_scores_dict, evidence_references)
    """
    evidence_refs = []

    try:
        # Compute all subscores using enhanced algorithms
        modality_scores = modality_analyzer.compute_modality_scores(target, ppi_graph)

        # Build comprehensive evidence references
        evidence_refs.append("VantAI:proprietary_v2.0")
        evidence_refs.append(f"E3_coexpr:{modality_scores['e3_coexpr']:.3f}")
        evidence_refs.append(f"Ternary_proxy:{modality_scores['ternary_proxy']:.3f}")
        evidence_refs.append(f"PPI_hotspot:{modality_scores['ppi_hotspot']:.3f}")

        # Add specific evidence details with enhanced tracking
        if target in modality_analyzer.ternary_data:
            evidence_level = modality_analyzer.ternary_data[target]
            evidence_refs.append(f"Ternary_evidence:{evidence_level}")

            if evidence_level in ['reported', 'validated']:
                evidence_refs.append("PROTAC_literature:available")

        if target in modality_analyzer.hotspot_data:
            hotspot_count = len(modality_analyzer.hotspot_data[target])
            evidence_refs.append(f"Hotspot_interactions:{hotspot_count}")

            # Add interaction type diversity
            interaction_types = set(h.get('type', 'binding') for h in modality_analyzer.hotspot_data[target])
            evidence_refs.append(f"Interaction_types:{len(interaction_types)}")

        # E3 ligase co-expression details with tissue specificity
        tissue_count = len([t for t in modality_analyzer.expression_data.keys()
                            if target in modality_analyzer.expression_data[t]])
        if tissue_count > 0:
            evidence_refs.append(f"Expression_tissues:{tissue_count}")

            # Calculate expression breadth
            if tissue_count >= 4:
                evidence_refs.append("Expression_breadth:broad")
            elif tissue_count >= 2:
                evidence_refs.append("Expression_breadth:moderate")
            else:
                evidence_refs.append("Expression_breadth:limited")

        # Add confidence indicators
        if modality_scores["overall_druggability"] > 0.7:
            evidence_refs.append("Confidence:high")
        elif modality_scores["overall_druggability"] > 0.4:
            evidence_refs.append("Confidence:medium")
        else:
            evidence_refs.append("Confidence:low")

        logger.info(f"Enhanced modality scores for {target}: overall={modality_scores['overall_druggability']:.3f}")

        return modality_scores, evidence_refs

    except Exception as e:
        logger.error(f"Error computing modality fit for {target}: {e}")

        # Return comprehensive default scores on error
        default_scores = {
            "e3_coexpr": 0.3,
            "ternary_proxy": 0.3,
            "ppi_hotspot": 0.3,
            "overall_druggability": 0.3,
            "protac_degrader": 0.3,
            "small_molecule": 0.3,
            "molecular_glue": 0.3,
            "adc_score": 0.3
        }

        error_refs = [f"Modality_error:{str(e)[:50]}", "VantAI:error_fallback"]
        return default_scores, error_refs