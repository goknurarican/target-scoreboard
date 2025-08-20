"""
Patent Landscape Analysis for Target Assessment
Demonstrates ability to integrate qualitative business intelligence with quantitative analysis
"""
from pathlib import Path

import httpx
import asyncio
from typing import Dict, List, Optional
import json
import time
from datetime import datetime, timedelta
import random


class PatentLandscapeAnalyzer:
    def __init__(self):
        self.cache_dir = Path("data_demo/cache/patents")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Mock patent database for demo
        self.mock_patent_data = self._initialize_mock_patents()

    def _initialize_mock_patents(self) -> Dict:
        """Initialize realistic patent landscape data for demo."""
        return {
            "EGFR": {
                "total_patents": 450,
                "active_patents": 180,
                "recent_filings": 25,
                "major_assignees": ["Roche", "AstraZeneca", "Pfizer", "Merck", "Novartis"],
                "key_areas": ["Small molecule inhibitors", "Antibodies", "ADCs", "Resistance mechanisms"],
                "freedom_to_operate": 0.3,  # Lower = more crowded
                "competitive_intensity": 0.8,
                "white_space_score": 0.2
            },
            "ERBB2": {
                "total_patents": 380,
                "active_patents": 150,
                "recent_filings": 18,
                "major_assignees": ["Roche", "Seattle Genetics", "Daiichi Sankyo", "GSK"],
                "key_areas": ["HER2 antibodies", "ADCs", "Bispecific antibodies", "Small molecules"],
                "freedom_to_operate": 0.4,
                "competitive_intensity": 0.7,
                "white_space_score": 0.3
            },
            "MET": {
                "total_patents": 220,
                "active_patents": 95,
                "recent_filings": 12,
                "major_assignees": ["Merck", "Novartis", "Amgen", "BMS", "Incyte"],
                "key_areas": ["Small molecule inhibitors", "Antibodies", "Hepatocyte growth factor"],
                "freedom_to_operate": 0.6,
                "competitive_intensity": 0.5,
                "white_space_score": 0.6
            },
            "ALK": {
                "total_patents": 180,
                "active_patents": 85,
                "recent_filings": 15,
                "major_assignees": ["Pfizer", "Roche", "Novartis", "Takeda", "Ariad"],
                "key_areas": ["Kinase inhibitors", "Resistance mutations", "Combination therapy"],
                "freedom_to_operate": 0.5,
                "competitive_intensity": 0.6,
                "white_space_score": 0.4
            },
            "KRAS": {
                "total_patents": 320,
                "active_patents": 140,
                "recent_filings": 35,
                "major_assignees": ["Amgen", "Mirati", "Revolution Medicines", "Boehringer", "Wellcome Trust"],
                "key_areas": ["G12C inhibitors", "SHP2 inhibitors", "Protein degraders", "Allosteric inhibitors"],
                "freedom_to_operate": 0.4,
                "competitive_intensity": 0.9,
                "white_space_score": 0.3
            }
        }

    async def analyze_patent_landscape(self, target_symbol: str) -> Dict:
        """
        Comprehensive patent landscape analysis for target.
        Shows IP intelligence and strategic thinking.
        """
        cache_file = self.cache_dir / f"{target_symbol}_patent_landscape.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Get base patent data
        patent_data = self.mock_patent_data.get(target_symbol, self._generate_mock_patent_data(target_symbol))

        # Analyze competitive landscape
        competitive_analysis = self._analyze_competitive_landscape(target_symbol, patent_data)

        # Identify white space opportunities
        white_space_analysis = self._identify_white_space(target_symbol, patent_data)

        # Risk assessment
        ip_risk_assessment = self._assess_ip_risks(target_symbol, patent_data)

        result = {
            "target": target_symbol,
            "patent_overview": patent_data,
            "competitive_landscape": competitive_analysis,
            "white_space_opportunities": white_space_analysis,
            "ip_risk_assessment": ip_risk_assessment,
            "strategic_recommendations": self._generate_strategic_recommendations(patent_data),
            "last_updated": time.time()
        }

        # Cache result
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def _analyze_competitive_landscape(self, target: str, patent_data: Dict) -> Dict:
        """Analyze competitive intensity and key players."""
        major_assignees = patent_data.get("major_assignees", [])

        # Calculate market concentration (Herfindahl index simulation)
        total_patents = patent_data.get("total_patents", 100)
        market_shares = [random.uniform(0.1, 0.3) for _ in major_assignees[:3]]
        market_shares.extend([random.uniform(0.05, 0.15) for _ in major_assignees[3:]])
        market_shares = [min(share, 1.0) for share in market_shares]

        concentration_index = sum(share ** 2 for share in market_shares)

        return {
            "market_leaders": major_assignees[:3],
            "emerging_players": major_assignees[3:],
            "market_concentration": round(concentration_index, 3),
            "competitive_intensity": patent_data.get("competitive_intensity", 0.5),
            "recent_activity": {
                "new_entrants": random.randint(2, 5),
                "increased_filing_rate": patent_data.get("recent_filings", 10) > 20,
                "patent_challenges": random.randint(0, 3)
            }
        }

    def _identify_white_space(self, target: str, patent_data: Dict) -> Dict:
        """Identify potential white space opportunities."""
        key_areas = patent_data.get("key_areas", [])

        # Simulate white space analysis
        white_space_areas = []
        emerging_modalities = ["PROTACs", "Molecular glues", "Allosteric modulators", "RNA therapeutics"]

        for modality in emerging_modalities:
            if modality not in " ".join(key_areas):
                white_space_areas.append({
                    "area": modality,
                    "opportunity_score": random.uniform(0.4, 0.9),
                    "technical_feasibility": random.uniform(0.3, 0.8),
                    "patent_freedom": random.uniform(0.6, 0.95)
                })

        return {
            "white_space_score": patent_data.get("white_space_score", 0.5),
            "opportunities": white_space_areas,
            "underexplored_mechanisms": [
                                            area for area in emerging_modalities
                                            if area not in " ".join(key_areas)
                                        ][:3],
            "geographical_gaps": ["Asia-Pacific", "Latin America"]  # Mock geographic analysis
        }

    def _assess_ip_risks(self, target: str, patent_data: Dict) -> Dict:
        """Assess IP-related risks and mitigation strategies."""
        fto_score = patent_data.get("freedom_to_operate", 0.5)

        risk_level = "HIGH" if fto_score < 0.4 else "MEDIUM" if fto_score < 0.7 else "LOW"

        return {
            "overall_risk_level": risk_level,
            "freedom_to_operate_score": fto_score,
            "key_risks": {
                "blocking_patents": fto_score < 0.5,
                "patent_thicket": patent_data.get("total_patents", 0) > 300,
                "recent_enforcement": random.choice([True, False]),
                "evergreening_risk": fto_score < 0.6
            },
            "mitigation_strategies": [
                                         "Design around existing patents",
                                         "Seek licensing opportunities",
                                         "File continuation applications",
                                         "Monitor patent expirations"
                                     ][:random.randint(2, 4)]
        }

    def _generate_strategic_recommendations(self, patent_data: Dict) -> List[str]:
        """Generate strategic recommendations based on patent analysis."""
        recommendations = []

        fto_score = patent_data.get("freedom_to_operate", 0.5)
        competitive_intensity = patent_data.get("competitive_intensity", 0.5)

        if fto_score < 0.4:
            recommendations.append("HIGH PRIORITY: Conduct detailed FTO analysis before program initiation")

        if competitive_intensity > 0.7:
            recommendations.append("Consider alternative mechanisms or novel modalities to avoid crowded space")

        if patent_data.get("recent_filings", 0) > 20:
            recommendations.append("Monitor recent patent filings for emerging competitive threats")

        recommendations.extend([
            "Explore white space opportunities in emerging modalities",
            "Consider strategic partnerships to access key IP",
            "File defensive patents around novel findings"
        ])

        return recommendations[:4]  # Return top 4 recommendations

    def _generate_mock_patent_data(self, target: str) -> Dict:
        """Generate mock patent data for unknown targets."""
        return {
            "total_patents": random.randint(50, 400),
            "active_patents": random.randint(20, 200),
            "recent_filings": random.randint(5, 30),
            "major_assignees": ["Company A", "Company B", "Company C", "Academic Inst"],
            "key_areas": ["Small molecules", "Biologics", "Novel modalities"],
            "freedom_to_operate": random.uniform(0.3, 0.8),
            "competitive_intensity": random.uniform(0.3, 0.9),
            "white_space_score": random.uniform(0.2, 0.7)
        }

    async def get_competitive_intelligence(self, targets: List[str]) -> Dict:
        """
        Cross-target competitive intelligence analysis.
        Shows portfolio-level strategic thinking.
        """
        portfolio_analysis = {}

        for target in targets:
            patent_data = await self.analyze_patent_landscape(target)
            portfolio_analysis[target] = {
                "freedom_to_operate": patent_data["patent_overview"]["freedom_to_operate"],
                "competitive_intensity": patent_data["patent_overview"]["competitive_intensity"],
                "white_space_score": patent_data["patent_overview"]["white_space_score"],
                "risk_level": patent_data["ip_risk_assessment"]["overall_risk_level"]
            }

        # Portfolio-level insights
        avg_fto = sum(data["freedom_to_operate"] for data in portfolio_analysis.values()) / len(targets)
        high_risk_targets = [target for target, data in portfolio_analysis.items()
                             if data["risk_level"] == "HIGH"]

        return {
            "portfolio_overview": portfolio_analysis,
            "portfolio_metrics": {
                "average_fto_score": round(avg_fto, 3),
                "high_risk_targets": high_risk_targets,
                "diversification_score": len(set(data["risk_level"] for data in portfolio_analysis.values())) / 3,
                "strategic_value": "HIGH" if avg_fto > 0.6 else "MEDIUM" if avg_fto > 0.4 else "LOW"
            },
            "strategic_insights": [
                f"Portfolio FTO score: {avg_fto:.2f}",
                f"High-risk targets: {len(high_risk_targets)}/{len(targets)}",
                "Consider IP strategy diversification" if len(high_risk_targets) > len(
                    targets) // 2 else "Strong IP position"
            ]
        }


# Global patent analyzer
patent_analyzer = PatentLandscapeAnalyzer()