"""
ChEMBL Integration for Drug-Target Bioactivity Data
Demonstrates API integration and bioactivity analysis capabilities
"""
from random import random

import httpx
import asyncio
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import json
import time


class ChEMBLClient:
    def __init__(self, base_url="https://www.ebi.ac.uk/chembl/api/data"):
        self.base_url = base_url
        self.cache_dir = Path("data_demo/cache/chembl")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def get_target_bioactivity(self, target_symbol: str) -> Dict:
        """
        Fetch bioactivity data for a target from ChEMBL.
        Shows ability to integrate external APIs and process biological data.
        """
        cache_file = self.cache_dir / f"{target_symbol}_bioactivity.json"

        # Check cache first
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First, get target ChEMBL ID
                target_response = await client.get(
                    f"{self.base_url}/target/search.json",
                    params={"q": target_symbol, "format": "json"}
                )

                if target_response.status_code != 200:
                    return self._get_mock_bioactivity_data(target_symbol)

                target_data = target_response.json()
                if not target_data.get("targets"):
                    return self._get_mock_bioactivity_data(target_symbol)

                chembl_id = target_data["targets"][0]["target_chembl_id"]

                # Get bioactivity data
                bioactivity_response = await client.get(
                    f"{self.base_url}/activity.json",
                    params={
                        "target_chembl_id": chembl_id,
                        "standard_type__in": "IC50,EC50,Ki,Kd",
                        "pchembl_value__isnull": "false",
                        "limit": 100,
                        "format": "json"
                    }
                )

                if bioactivity_response.status_code == 200:
                    bioactivity_data = bioactivity_response.json()
                    processed_data = self._process_bioactivity_data(
                        target_symbol, chembl_id, bioactivity_data
                    )
                else:
                    processed_data = self._get_mock_bioactivity_data(target_symbol)

        except Exception as e:
            print(f"ChEMBL API error for {target_symbol}: {e}")
            processed_data = self._get_mock_bioactivity_data(target_symbol)

        # Cache the result
        with open(cache_file, 'w') as f:
            json.dump(processed_data, f, indent=2)

        return processed_data

    def _process_bioactivity_data(self, target_symbol: str, chembl_id: str, raw_data: Dict) -> Dict:
        """Process raw ChEMBL bioactivity data into actionable metrics."""
        activities = raw_data.get("activities", [])

        if not activities:
            return self._get_mock_bioactivity_data(target_symbol)

        # Analyze bioactivity patterns
        pchembl_values = []
        assay_types = set()
        compound_count = len(set(act.get("molecule_chembl_id") for act in activities))

        for activity in activities:
            if activity.get("pchembl_value"):
                pchembl_values.append(float(activity["pchembl_value"]))
                assay_types.add(activity.get("standard_type", "unknown"))

        # Calculate druggability metrics
        if pchembl_values:
            avg_potency = sum(pchembl_values) / len(pchembl_values)
            max_potency = max(pchembl_values)
            potent_compounds = len([p for p in pchembl_values if p >= 6.0])  # pIC50 >= 6 (IC50 <= 1µM)
        else:
            avg_potency = max_potency = potent_compounds = 0

        # Druggability score based on bioactivity data
        druggability_score = min(1.0, (
                (avg_potency / 10.0) * 0.4 +  # Average potency contribution
                (potent_compounds / max(10, compound_count)) * 0.4 +  # Fraction of potent compounds
                (len(assay_types) / 5.0) * 0.2  # Assay diversity
        ))

        return {
            "target_symbol": target_symbol,
            "chembl_id": chembl_id,
            "compound_count": compound_count,
            "assay_count": len(activities),
            "avg_potency_pchembl": round(avg_potency, 2) if avg_potency else 0,
            "max_potency_pchembl": round(max_potency, 2) if max_potency else 0,
            "potent_compounds": potent_compounds,
            "assay_types": list(assay_types),
            "druggability_score": round(druggability_score, 3),
            "data_quality": "high" if len(activities) > 20 else "medium" if len(activities) > 5 else "low",
            "last_updated": time.time()
        }

    def _get_mock_bioactivity_data(self, target_symbol: str) -> Dict:
        """Generate realistic mock data for demo purposes."""
        import random

        # Known druggable targets get better scores
        known_druggable = ["EGFR", "ERBB2", "MET", "ALK", "BRAF", "PIK3CA"]
        base_score = 0.8 if target_symbol in known_druggable else 0.4

        compound_count = random.randint(15, 150)
        assay_count = random.randint(20, 200)
        avg_potency = random.uniform(4.5, 8.5)

        return {
            "target_symbol": target_symbol,
            "chembl_id": f"CHEMBL_{random.randint(1000, 9999)}",
            "compound_count": compound_count,
            "assay_count": assay_count,
            "avg_potency_pchembl": round(avg_potency, 2),
            "max_potency_pchembl": round(avg_potency + random.uniform(0.5, 2.0), 2),
            "potent_compounds": random.randint(5, compound_count // 3),
            "assay_types": ["IC50", "EC50", "Ki", "Kd"][:random.randint(2, 4)],
            "druggability_score": round(min(1.0, base_score + random.uniform(-0.2, 0.2)), 3),
            "data_quality": random.choice(["high", "medium", "low"]),
            "last_updated": time.time(),
            "source": "mock_data"
        }

    async def get_compound_similarity_analysis(self, target_symbol: str) -> Dict:
        """
        Analyze chemical space and compound similarity for target.
        Demonstrates advanced data analysis capabilities.
        """
        bioactivity_data = await self.get_target_bioactivity(target_symbol)

        # Mock similarity analysis (in real implementation, would use RDKit)
        similarity_clusters = random.randint(3, 8)
        scaffold_diversity = random.uniform(0.3, 0.9)

        return {
            "target": target_symbol,
            "chemical_space": {
                "similarity_clusters": similarity_clusters,
                "scaffold_diversity": round(scaffold_diversity, 3),
                "lead_like_compounds": random.randint(5, 30),
                "drug_like_compounds": random.randint(10, 50)
            },
            "novelty_score": round(1.0 - (similarity_clusters / 10.0), 3),
            "chemical_tractability": round(scaffold_diversity * 0.7 + 0.3, 3)
        }


# Global ChEMBL client
chembl_client = ChEMBLClient()


async def get_target_druggability_data(target_symbol: str) -> Dict:
    """Get comprehensive druggability data from ChEMBL."""
    bioactivity = await chembl_client.get_target_bioactivity(target_symbol)
    similarity = await chembl_client.get_compound_similarity_analysis(target_symbol)

    return {
        "bioactivity": bioactivity,
        "chemical_space": similarity
    }