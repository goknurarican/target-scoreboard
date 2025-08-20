# VantAI Target Scoreboard

Modality-aware target scoring system built on Open Targets data.

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