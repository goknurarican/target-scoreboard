# Target Scoreboard for VantAI

A comprehensive computational platform for modality-aware target prioritization in drug discovery, integrating multi-omics data sources to support evidence-based decision making.
<img width="914" height="381" alt="Screenshot 2025-08-21 at 15 51 50" src="https://github.com/user-attachments/assets/958dcf62-c795-49d4-988b-13df067fb470" />

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
  
## Interactive Network Analysis
<img width="874" height="699" alt="Screenshot 2025-08-21 at 11 53 36" src="https://github.com/user-attachments/assets/c89fe72c-6bb1-4490-a73b-6241971d4221" />

PPI network visualization showing first-shell neighbors and interaction strengths

The platform provides detailed protein-protein interaction analysis with:

* First-shell neighbor identification with confidence scores
* Network centrality metrics for hub protein detection
* Interactive visualization of target connectivity
* Mechanism of action hypothesis generation


## Modality-Specific Assessment
<img width="900" height="659" alt="Screenshot 2025-08-21 at 11 55 24" src="https://github.com/user-attachments/assets/cab5e179-94ed-49b4-a75b-c45ba35a85d2" />

Modality fit analysis showing E3 co-expression, ternary feasibility, and PPI hotspot scores
Advanced druggability assessment includes:

* E3 Co-expression: Likelihood of successful PROTAC development
* Ternary Feasibility: Ternary complex formation potential
* PPI Hotspot: Protein-protein interaction druggability
* Small molecule and degrader suitability scoring

## Validation and Benchmarking 
<img width="783" height="664" alt="Screenshot 2025-08-21 at 11 56 04" src="https://github.com/user-attachments/assets/eadbaf44-0fe2-48c7-b7b7-2243417a35ea" />

Performance validation against known therapeutic targets with precision metrics
Comprehensive validation framework featuring:

* Ground truth comparison against FDA-approved targets
* Precision@k metrics (k=1,3,5) for ranking quality assessment
* AUC-PR calculation for overall performance evaluation
* Historical validation against clinical outcomes

## Architecture
Backend (FastAPI)

* RESTful API with async processing capabilities
* Multi-source data integration and caching
* Configurable scoring algorithms with weight optimization
* Export functionality (JSON/CSV) with metadata preservation

Frontend (Streamlit)

* Tabbed interface for organized workflow
* Real-time analysis with shareable URLs
* Interactive visualizations and progress tracking
* Responsive design with professional theming

Data Sources

* Open Targets 2024.06: 15M+ target-disease associations
* STRING v12.0: 24M protein interactions with confidence scores
* Reactome 2024: Curated biological pathway database
* VantAI Proprietary: Modality-specific scoring algorithms


## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Start API: `uvicorn app.main:app --reload`
3. Test API: `./examples/curl_score.sh`
4. Start dashboard: `streamlit run dashboard/app.py`

## Setup
Clone repository
git clone https://github.com/vantai/target-scoreboard.git
cd target-scoreboard

Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

## Install dependencies
`pip install -r requirements.txt`

## Set environment variables
`export API_PORT=8001`
`export STREAMLIT_PORT=8501`

## Start backend API
`python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload`

## Start dashboard (in separate terminal)
`streamlit run dashboard/ui_app.py --server.port 8501`

## License

All rights reserved. Shared for evaluation only under LICENSE-EVALUATION.md.
