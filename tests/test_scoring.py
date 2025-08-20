import pytest
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import ScoreRequest, TargetBreakdown
from app.scoring import target_scorer, validate_score_request
from app.channels.genetics import compute_genetics_score
from app.channels.ppi_proximity import compute_ppi_proximity
from app.channels.pathway import compute_pathway_enrichment
from app.channels.modality_fit import compute_modality_fit


class TestScoring:

    def test_validate_score_request_valid(self):
        """Test request validation with valid request."""
        request = ScoreRequest(
            disease="EFO_0000305",
            targets=["EGFR", "ERBB2"],
            weights={"genetics": 0.6, "ppi": 0.4}
        )

        is_valid, error_msg = validate_score_request(request)
        assert is_valid
        assert error_msg == ""

    def test_validate_score_request_empty_targets(self):
        """Test request validation with empty targets."""
        request = ScoreRequest(
            disease="EFO_0000305",
            targets=[],
            weights={"genetics": 1.0}
        )

        is_valid, error_msg = validate_score_request(request)
        assert not is_valid
        assert "No targets provided" in error_msg

    def test_validate_score_request_too_many_targets(self):
        """Test request validation with too many targets."""
        targets = [f"TARGET_{i}" for i in range(60)]
        request = ScoreRequest(
            disease="EFO_0000305",
            targets=targets,
            weights={"genetics": 1.0}
        )

        is_valid, error_msg = validate_score_request(request)
        assert not is_valid
        assert "Too many targets" in error_msg

    def test_validate_score_request_invalid_weights(self):
        """Test request validation with invalid weights."""
        request = ScoreRequest(
            disease="EFO_0000305",
            targets=["EGFR"],
            weights={"genetics": 1.5}  # > 1.0
        )

        is_valid, error_msg = validate_score_request(request)
        assert not is_valid
        assert "must be between 0 and 1" in error_msg

    def test_validate_score_request_weights_sum(self):
        """Test request validation with weights that don't sum to ~1.0."""
        request = ScoreRequest(
            disease="EFO_0000305",
            targets=["EGFR"],
            weights={"genetics": 0.1, "ppi": 0.1}  # Sum = 0.2
        )

        is_valid, error_msg = validate_score_request(request)
        assert not is_valid
        assert "should sum to ~1.0" in error_msg

    @pytest.mark.asyncio
    async def test_score_single_target(self):
        """Test scoring a single target."""
        weights = {"genetics": 0.5, "ppi": 0.3, "pathway": 0.2}

        target_score = await target_scorer.score_single_target(
            disease="EFO_0000305",
            target="EGFR",
            weights=weights
        )

        assert target_score.target == "EGFR"
        assert 0 <= target_score.total_score <= 1
        assert target_score.breakdown is not None
        assert len(target_score.evidence_refs) > 0
        assert target_score.data_version is not None

        # Check breakdown components
        breakdown = target_score.breakdown
        assert breakdown.genetics is not None
        assert 0 <= breakdown.genetics <= 1
        assert breakdown.ppi_proximity is not None
        assert 0 <= breakdown.ppi_proximity <= 1

    @pytest.mark.asyncio
    async def test_score_targets_batch(self):
        """Test batch scoring of multiple targets."""
        request = ScoreRequest(
            disease="EFO_0000305",
            targets=["EGFR", "ERBB2", "MET"],
            weights={"genetics": 0.6, "ppi": 0.4}
        )

        target_scores = await target_scorer.score_targets_batch(request)

        assert len(target_scores) == 3

        for target_score in target_scores:
            assert target_score.target in ["EGFR", "ERBB2", "MET"]
            assert 0 <= target_score.total_score <= 1
            assert target_score.breakdown is not None
            assert len(target_score.evidence_refs) > 0

    def test_weighted_score_computation(self):
        """Test weighted score computation logic."""
        channel_scores = {
            "genetics": 0.8,
            "ppi": 0.6,
            "pathway": 0.4,
            "modality_fit": 0.7,
            "safety": 0.3  # This should be inverted (1.0 - 0.3 = 0.7)
        }

        weights = {
            "genetics": 0.4,
            "ppi": 0.3,
            "pathway": 0.2,
            "modality_fit": 0.1,
            "safety": 0.0  # No safety weight
        }

        total_score = target_scorer._compute_weighted_score(channel_scores, weights)

        # Expected: 0.8*0.4 + 0.6*0.3 + 0.4*0.2 + 0.7*0.1 = 0.32 + 0.18 + 0.08 + 0.07 = 0.65
        expected = 0.65
        assert abs(total_score - expected) < 0.01

    def test_weighted_score_with_safety(self):
        """Test weighted score computation with safety component."""
        channel_scores = {
            "genetics": 0.8,
            "safety": 0.3  # High safety concern
        }

        weights = {
            "genetics": 0.7,
            "safety": 0.3
        }

        total_score = target_scorer._compute_weighted_score(channel_scores, weights)

        # Expected: 0.8*0.7 + (1.0-0.3)*0.3 = 0.56 + 0.21 = 0.77
        expected = 0.77
        assert abs(total_score - expected) < 0.01

    def test_weighted_score_missing_channels(self):
        """Test weighted score computation with missing channels."""
        channel_scores = {
            "genetics": 0.8,
            # Missing other channels
        }

        weights = {
            "genetics": 0.5,
            "ppi": 0.3,
            "pathway": 0.2
        }

        total_score = target_scorer._compute_weighted_score(channel_scores, weights)

        # Should use only genetics with normalized weight
        # total_score = 0.8 * (0.5 / 0.5) = 0.8
        assert abs(total_score - 0.8) < 0.01

    def test_scoring_summary(self):
        """Test scoring summary generation."""
        # Create mock target scores
        mock_scores = [
            type('TargetScore', (), {
                'target': 'EGFR',
                'total_score': 0.8,
                'breakdown': type('Breakdown', (), {
                    'genetics': 0.7,
                    'ppi_proximity': 0.6
                })(),
                'evidence_refs': ['OT:test', 'STRING:test'],
                'data_version': 'Test:v1.0'
            })(),
            type('TargetScore', (), {
                'target': 'ERBB2',
                'total_score': 0.6,
                'breakdown': type('Breakdown', (), {
                    'genetics': 0.5,
                    'ppi_proximity': 0.4
                })(),
                'evidence_refs': ['OT:test'],
                'data_version': 'Test:v1.0'
            })()
        ]

        summary = target_scorer.get_scoring_summary(mock_scores)

        assert summary['total_targets'] == 2
        assert summary['score_statistics']['mean_total_score'] == 0.7
        assert summary['score_statistics']['max_total_score'] == 0.8
        assert summary['top_targets'][0]['target'] == 'EGFR'
        assert 'OT' in summary['evidence_sources']


class TestGeneticsChannel:

    def test_compute_genetics_score_valid_data(self):
        """Test genetics scoring with valid Open Targets data."""
        ot_data = {
            "genetics_score": 0.75,
            "overall_score": 0.6,
            "meta": {"version": "25.06"},
            "association": {
                "evidenceCount": {
                    "total": 150,
                    "datatype": [
                        {"id": "genetic_association", "count": 25}
                    ]
                }
            }
        }

        score, refs = compute_genetics_score("EFO_0000305", "EGFR", ot_data)

        assert 0 <= score <= 1
        assert score == 0.75  # Should use genetics_score directly
        assert len(refs) > 0
        assert any("OpenTargets" in ref for ref in refs)
        assert any("Evidence_count:150" in ref for ref in refs)

    def test_compute_genetics_score_missing_genetics(self):
        """Test genetics scoring when genetics_score is missing."""
        ot_data = {
            "genetics_score": None,
            "overall_score": 0.6,
            "meta": {"version": "25.06"}
        }

        score, refs = compute_genetics_score("EFO_0000305", "EGFR", ot_data)

        assert 0 <= score <= 1
        assert score == 0.6 * 0.7  # Should use overall_score with conservative factor
        assert len(refs) > 0


class TestPPIChannel:

    def test_compute_ppi_proximity(self):
        """Test PPI proximity scoring."""
        score, refs = compute_ppi_proximity("EGFR")

        assert 0 <= score <= 1
        assert len(refs) > 0
        assert any("STRING" in ref for ref in refs)

    def test_compute_ppi_proximity_unknown_target(self):
        """Test PPI proximity for unknown target."""
        score, refs = compute_ppi_proximity("UNKNOWN_TARGET_12345")

        assert 0 <= score <= 1
        assert len(refs) > 0


class TestPathwayChannel:

    def test_compute_pathway_enrichment(self):
        """Test pathway enrichment scoring."""
        score, refs = compute_pathway_enrichment("EGFR")

        assert 0 <= score <= 1
        assert len(refs) > 0
        assert any("Reactome" in ref for ref in refs)

    def test_compute_pathway_enrichment_with_context(self):
        """Test pathway enrichment with target context."""
        targets_context = ["EGFR", "ERBB2", "MET"]
        score, refs = compute_pathway_enrichment("EGFR", targets_context)

        assert 0 <= score <= 1
        assert len(refs) > 0


class TestModalityFitChannel:

    def test_compute_modality_fit(self):
        """Test modality fit scoring."""
        score_dict, refs = compute_modality_fit("EGFR")

        assert isinstance(score_dict, dict)
        assert len(refs) > 0

        # Check expected keys
        expected_keys = ["e3_coexpr", "ternary_proxy", "ppi_hotspot", "protac_degrader", "small_molecule",
                         "overall_druggability"]
        for key in expected_keys:
            assert key in score_dict
            assert 0 <= score_dict[key] <= 1

    def test_compute_modality_fit_unknown_target(self):
        """Test modality fit for unknown target."""
        score_dict, refs = compute_modality_fit("UNKNOWN_TARGET_12345")

        assert isinstance(score_dict, dict)
        assert len(refs) > 0
        # Scores should be low but not zero for unknown targets
        assert all(0 <= score <= 0.5 for score in score_dict.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])