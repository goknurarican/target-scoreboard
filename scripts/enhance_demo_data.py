#!/usr/bin/env python3
"""
Enhanced Demo Data Generator for VantAI Target Scoreboard
Creates realistic, rich datasets that demonstrate data integration capabilities
"""

import json
import pandas as pd
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class DemoDataEnhancer:
    def __init__(self, output_dir="data_demo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_comprehensive_target_data(self):
        """Generate comprehensive target dataset with multiple data types."""

        # Extended target list with real oncology targets
        targets = {
            "EGFR": {
                "full_name": "Epidermal Growth Factor Receptor",
                "protein_class": "Receptor tyrosine kinase",
                "chromosome": "7p12",
                "diseases": ["NSCLC", "Glioblastoma", "Colorectal cancer"],
                "druggability_tier": "Tier 1",
                "known_inhibitors": ["Gefitinib", "Erlotinib", "Osimertinib"],
                "market_size_m": 4500,
                "patent_landscape": "Crowded"
            },
            "ERBB2": {
                "full_name": "Erb-B2 Receptor Tyrosine Kinase 2",
                "protein_class": "Receptor tyrosine kinase",
                "chromosome": "17q12",
                "diseases": ["Breast cancer", "Gastric cancer"],
                "druggability_tier": "Tier 1",
                "known_inhibitors": ["Trastuzumab", "Pertuzumab", "T-DM1"],
                "market_size_m": 3200,
                "patent_landscape": "Moderately crowded"
            },
            "KRAS": {
                "full_name": "KRAS Proto-Oncogene GTPase",
                "protein_class": "Small GTPase",
                "chromosome": "12p12.1",
                "diseases": ["Pancreatic cancer", "Colorectal cancer", "NSCLC"],
                "druggability_tier": "Tier 2",
                "known_inhibitors": ["Sotorasib", "Adagrasib"],
                "market_size_m": 8900,
                "patent_landscape": "Emerging competition"
            },
            "TP53": {
                "full_name": "Tumor Protein P53",
                "protein_class": "Transcription factor",
                "chromosome": "17p13.1",
                "diseases": ["Pan-cancer", "Li-Fraumeni syndrome"],
                "druggability_tier": "Tier 3",
                "known_inhibitors": ["APR-246", "PRIMA-1"],
                "market_size_m": 15000,
                "patent_landscape": "Limited competition"
            },
            "MET": {
                "full_name": "MET Proto-Oncogene Receptor Tyrosine Kinase",
                "protein_class": "Receptor tyrosine kinase",
                "chromosome": "7q31.2",
                "diseases": ["NSCLC", "Glioblastoma", "Renal cell carcinoma"],
                "druggability_tier": "Tier 1",
                "known_inhibitors": ["Crizotinib", "Cabozantinib"],
                "market_size_m": 2100,
                "patent_landscape": "Moderate competition"
            }
        }

        # Save enhanced target metadata
        with open(self.output_dir / "target_metadata.json", 'w') as f:
            json.dump(targets, f, indent=2)

        return targets

    def generate_bioactivity_data(self, targets):
        """Generate realistic bioactivity data mimicking ChEMBL."""

        bioactivity_data = {}

        for target, metadata in targets.items():
            # Generate compound bioactivities
            num_compounds = random.randint(20, 200)
            compounds = []

            for i in range(num_compounds):
                # Realistic bioactivity values
                if metadata["druggability_tier"] == "Tier 1":
                    pic50_base = random.uniform(6.0, 9.0)
                elif metadata["druggability_tier"] == "Tier 2":
                    pic50_base = random.uniform(5.0, 7.5)
                else:
                    pic50_base = random.uniform(4.0, 6.5)

                compound = {
                    "compound_id": f"CHEMBL{random.randint(100000, 999999)}",
                    "assay_type": random.choice(["IC50", "EC50", "Ki", "Kd"]),
                    "pic50_value": round(pic50_base + random.normalvariate(0, 0.5), 2),
                    "assay_organism": "Homo sapiens",
                    "confidence_score": random.randint(7, 9),
                    "publication_year": random.randint(2010, 2024)
                }
                compounds.append(compound)

            bioactivity_data[target] = {
                "total_compounds": num_compounds,
                "compounds": compounds,
                "avg_pic50": round(np.mean([c["pic50_value"] for c in compounds]), 2),
                "potent_compounds": len([c for c in compounds if c["pic50_value"] >= 6.0]),
                "data_quality": "high" if num_compounds > 50 else "medium"
            }

        # Save bioactivity data
        with open(self.output_dir / "bioactivity_data.json", 'w') as f:
            json.dump(bioactivity_data, f, indent=2)

        return bioactivity_data

    def generate_expression_atlas(self, targets):
        """Generate tissue expression data mimicking GTEx."""

        tissues = [
            "Brain", "Heart", "Liver", "Lung", "Kidney", "Muscle", "Skin",
            "Breast", "Colon", "Stomach", "Pancreas", "Prostate", "Ovary", "Testis"
        ]

        expression_data = {}

        for target in targets.keys():
            tissue_expression = {}
            for tissue in tissues:
                # Simulate tissue-specific expression
                base_expression = random.uniform(0.1, 10.0)

                # Some targets are more tissue-specific
                if target == "EGFR" and tissue in ["Lung", "Brain", "Skin"]:
                    base_expression *= random.uniform(2.0, 5.0)
                elif target == "ERBB2" and tissue in ["Breast", "Heart"]:
                    base_expression *= random.uniform(3.0, 6.0)

                tissue_expression[tissue] = {
                    "median_tpm": round(base_expression, 2),
                    "q75_tpm": round(base_expression * 1.5, 2),
                    "sample_count": random.randint(50, 300)
                }

            expression_data[target] = tissue_expression

        # Save expression data
        with open(self.output_dir / "expression_atlas.json", 'w') as f:
            json.dump(expression_data, f, indent=2)

        return expression_data

    def generate_clinical_trial_data(self, targets):
        """Generate clinical trial landscape data."""

        trial_phases = ["Phase I", "Phase II", "Phase III", "Approved"]
        indications = ["NSCLC", "Breast cancer", "Colorectal cancer", "Glioblastoma", "Pancreatic cancer"]

        clinical_data = {}

        for target, metadata in targets.items():
            trials = []
            num_trials = random.randint(5, 30)

            for i in range(num_trials):
                trial = {
                    "nct_id": f"NCT{random.randint(10000000, 99999999)}",
                    "title": f"Study of {random.choice(['Novel', 'Investigational', 'Targeted'])} {target} inhibitor",
                    "phase": random.choice(trial_phases),
                    "indication": random.choice(metadata["diseases"]),
                    "sponsor": random.choice(["Roche", "Pfizer", "Novartis", "Merck", "BMS", "AbbVie"]),
                    "start_date": (datetime.now() - timedelta(days=random.randint(30, 2000))).strftime("%Y-%m-%d"),
                    "enrollment": random.randint(20, 500),
                    "status": random.choice(["Active", "Completed", "Recruiting", "Terminated"])
                }
                trials.append(trial)

            # Calculate success metrics
            completed_trials = [t for t in trials if t["status"] == "Completed"]
            success_rate = len([t for t in completed_trials if t["phase"] in ["Phase III", "Approved"]]) / max(1,
                                                                                                               len(completed_trials))

            clinical_data[target] = {
                "total_trials": num_trials,
                "trials": trials,
                "active_trials": len([t for t in trials if t["status"] in ["Active", "Recruiting"]]),
                "success_rate": round(success_rate, 2),
                "pipeline_strength": "Strong" if success_rate > 0.3 else "Moderate" if success_rate > 0.1 else "Weak"
            }

        # Save clinical data
        with open(self.output_dir / "clinical_trials.json", 'w') as f:
            json.dump(clinical_data, f, indent=2)

        return clinical_data

    def generate_literature_analysis(self, targets):
        """Generate literature co-occurrence and trend data."""

        literature_data = {}

        for target in targets.keys():
            # Simulate publication trends
            years = list(range(2014, 2025))
            publications = []

            base_count = random.randint(100, 1000)
            for year in years:
                # Simulate growth trend
                growth_factor = 1 + (year - 2014) * 0.1
                year_pubs = int(base_count * growth_factor * random.uniform(0.8, 1.2))
                publications.append({"year": year, "count": year_pubs})

            # Co-occurrence analysis
            related_terms = {
                "drug_discovery": random.randint(50, 300),
                "biomarker": random.randint(20, 150),
                "resistance": random.randint(30, 200),
                "combination_therapy": random.randint(15, 100),
                "personalized_medicine": random.randint(10, 80)
            }

            literature_data[target] = {
                "total_publications": sum(p["count"] for p in publications),
                "publication_trend": publications,
                "related_concepts": related_terms,
                "h_index": random.randint(20, 150),
                "research_momentum": "High" if publications[-1]["count"] > publications[-3]["count"] else "Stable"
            }

        # Save literature data
        with open(self.output_dir / "literature_analysis.json", 'w') as f:
            json.dump(literature_data, f, indent=2)

        return literature_data

    def generate_all_enhanced_data(self):
        """Generate all enhanced datasets."""
        print("🔬 Generating enhanced demo data for VantAI Target Scoreboard...")

        # Generate all datasets
        targets = self.generate_comprehensive_target_data()
        print(f"✅ Generated metadata for {len(targets)} targets")

        bioactivity = self.generate_bioactivity_data(targets)
        print(
            f"✅ Generated bioactivity data with {sum(data['total_compounds'] for data in bioactivity.values())} compounds")

        expression = self.generate_expression_atlas(targets)
        print(f"✅ Generated expression atlas across {len(list(expression.values())[0])} tissues")

        clinical = self.generate_clinical_trial_data(targets)
        print(f"✅ Generated clinical trial data with {sum(data['total_trials'] for data in clinical.values())} trials")

        literature = self.generate_literature_analysis(targets)
        print(
            f"✅ Generated literature analysis with {sum(data['total_publications'] for data in literature.values())} publications")

        # Generate summary report
        summary = {
            "generation_date": datetime.now().isoformat(),
            "targets_count": len(targets),
            "data_types": ["metadata", "bioactivity", "expression", "clinical_trials", "literature"],
            "quality_metrics": {
                "avg_compounds_per_target": round(
                    sum(data['total_compounds'] for data in bioactivity.values()) / len(targets), 1),
                "avg_trials_per_target": round(sum(data['total_trials'] for data in clinical.values()) / len(targets),
                                               1),
                "high_quality_targets": len([t for t, data in bioactivity.items() if data['data_quality'] == 'high'])
            }
        }

        with open(self.output_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"🎯 Enhanced demo data generation complete!")
        print(f"📊 Data summary saved to {self.output_dir}/data_summary.json")

        return {
            "targets": targets,
            "bioactivity": bioactivity,
            "expression": expression,
            "clinical": clinical,
            "literature": literature,
            "summary": summary
        }


if __name__ == "__main__":
    enhancer = DemoDataEnhancer()
    enhanced_data = enhancer.generate_all_enhanced_data()