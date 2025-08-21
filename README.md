# Target Scoreboard for VantAI

A comprehensive computational platform for modality-aware target prioritization in drug discovery, integrating multi-omics data sources to support evidence-based decision making.
<img width="943" height="271" alt="Screenshot 2025-08-21 at 11 14 20" src="https://github.com/user-attachments/assets/f0f8ac1d-1af7-454d-9fc8-6e56e4625900" />
## Overview
This Target Scoreboard provides systematic evaluation of therapeutic targets by combining genetic associations, protein interaction networks, pathway analysis, safety assessments, and modality-specific druggability scores. The platform offers transparent, explainable scoring with robust sensitivity analysis capabilities.
<img width="917" height="565" alt="Screenshot 2025-08-21 at 11 50 54" src="https://github.com/user-attachments/assets/d6fc7ea5-c739-4f50-9ccd-4773eafc903c" />
Analytics Overview showing target scoring distribution and key metrics

## Key Features
Multi-Modal Scoring System

* Genetics Channel (35%): Disease association strength from GWAS and genetic studies via Open Targets
* PPI Network Channel (25%): Protein interaction connectivity and centrality from STRING database
* Pathway Channel (20%): Biological pathway enrichment analysis using Reactome data
* Safety Channel (10%): Tissue specificity and off-target risk assessment
* Modality Fit Channel (10%): Druggability evaluation for different therapeutic modalities

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Start API: `uvicorn app.main:app --reload`
3. Test API: `./examples/curl_score.sh`
4. Start dashboard: `streamlit run dashboard/app.py`

## Features

- Open Targets integration
- Multi-modal scoring (genetics, PPI, pathways, modality-fit)
- Explainable results with evidence references
- Interactive Streamlit dashboard

## License

All rights reserved. Shared for evaluation only under LICENSE-EVALUATION.md.
