import pytest
import asyncio
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.schemas import ScoreRequest

# Create test client
client = TestClient(app)


class TestAPI:

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "VantAI Target Scoreboard"

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "VantAI Target Scoreboard API"
        assert "endpoints" in data
        assert "example_usage" in data

    def test_score_endpoint_valid_request(self):
        """Test scoring endpoint with valid request."""
        request_data = {
            "disease": "EFO_0000305",
            "targets": ["EGFR", "ERBB2"],
            "weights": {
                "genetics": 0.4,
                "ppi": 0.3,
                "pathway": 0.2,
                "safety": 0.05,
                "modality_fit": 0.05
            }
        }

        response = client.post("/score", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "targets" in data
        assert len(data["targets"]) == 2
        assert "processing_time_ms" in data
        assert "request_summary" in data

        # Check target structure
        for target_score in data["targets"]:
            assert "target" in target_score
            assert "total_score" in target_score
            assert "breakdown" in target_score
            assert "evidence_refs" in target_score
            assert "data_version" in target_score

            # Check score range
            assert 0 <= target_score["total_score"] <= 1

            # Check breakdown structure
            breakdown = target_score["breakdown"]
            assert "genetics" in breakdown
            assert "ppi_proximity" in breakdown
            assert "pathway_enrichment" in breakdown
            assert "modality_fit" in breakdown

    def test_score_endpoint_empty_targets(self):
        """Test scoring endpoint with empty targets list."""
        request_data = {
            "disease": "EFO_0000305",
            "targets": [],
            "weights": {"genetics": 1.0}
        }

        response = client.post("/score", json=request_data)
        assert response.status_code == 400

    def test_score_endpoint_invalid_weights(self):
        """Test scoring endpoint with invalid weights."""
        request_data = {
            "disease": "EFO_0000305",
            "targets": ["EGFR"],
            "weights": {
                "genetics": 1.5,  # Invalid: > 1.0
                "ppi": 0.3
            }
        }

        response = client.post("/score", json=request_data)
        assert response.status_code == 400

    def test_score_endpoint_too_many_targets(self):
        """Test scoring endpoint with too many targets."""
        targets = [f"TARGET_{i}" for i in range(60)]  # More than 50

        request_data = {
            "disease": "EFO_0000305",
            "targets": targets,
            "weights": {"genetics": 1.0}
        }

        response = client.post("/score", json=request_data)
        assert response.status_code == 400

    def test_modality_recommendations_endpoint(self):
        """Test modality recommendations endpoint."""
        response = client.get("/targets/EGFR/modality-recommendations")
        assert response.status_code == 200

        data = response.json()
        assert "target" in data
        assert "recommendations" in data
        assert data["target"] == "EGFR"

    def test_system_summary_endpoint(self):
        """Test system summary endpoint."""
        response = client.get("/summary")
        assert response.status_code == 200

        data = response.json()
        assert "system_info" in data
        assert "data_sources" in data
        assert "scoring_channels" in data
        assert "default_weights" in data
        assert "capabilities" in data

    def test_score_endpoint_single_target(self):
        """Test scoring with single target."""
        request_data = {
            "disease": "EFO_0000305",
            "targets": ["EGFR"],
            "weights": {
                "genetics": 0.5,
                "ppi": 0.3,
                "pathway": 0.2
            }
        }

        response = client.post("/score", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["targets"]) == 1

        target_score = data["targets"][0]
        assert target_score["target"] == "EGFR"
        assert isinstance(target_score["total_score"], float)
        assert isinstance(target_score["evidence_refs"], list)
        assert len(target_score["evidence_refs"]) > 0

    def test_score_endpoint_default_weights(self):
        """Test scoring with default weights (not all channels specified)."""
        request_data = {
            "disease": "EFO_0000305",
            "targets": ["EGFR", "MET"],
            "weights": {
                "genetics": 0.6,
                "ppi": 0.4
                # Other weights should use defaults
            }
        }

        response = client.post("/score", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["targets"]) == 2

    def test_score_response_consistency(self):
        """Test that multiple calls return consistent results."""
        request_data = {
            "disease": "EFO_0000305",
            "targets": ["EGFR"],
            "weights": {"genetics": 1.0}
        }

        # Make two identical requests
        response1 = client.post("/score", json=request_data)
        response2 = client.post("/score", json=request_data)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Scores should be identical (since we're using cached/deterministic data)
        score1 = data1["targets"][0]["total_score"]
        score2 = data2["targets"][0]["total_score"]
        assert abs(score1 - score2) < 0.001  # Allow for small floating point differences


# Additional tests for edge cases
class TestAPIEdgeCases:

    def test_malformed_json(self):
        """Test API with malformed JSON."""
        response = client.post(
            "/score",
            data="{'invalid': 'json'",  # Missing quotes, etc.
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity

    def test_missing_required_fields(self):
        """Test API with missing required fields."""
        request_data = {
            "targets": ["EGFR"]
            # Missing disease field
        }

        response = client.post("/score", json=request_data)
        assert response.status_code == 422

    def test_unknown_target(self):
        """Test scoring with unknown/invalid target."""
        request_data = {
            "disease": "EFO_0000305",
            "targets": ["UNKNOWN_GENE_12345"],
            "weights": {"genetics": 1.0}
        }

        response = client.post("/score", json=request_data)
        # Should still return 200 but with low/default scores
        assert response.status_code == 200

        data = response.json()
        target_score = data["targets"][0]
        assert target_score["target"] == "UNKNOWN_GENE_12345"
        # Score should be low for unknown targets
        assert target_score["total_score"] <= 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])