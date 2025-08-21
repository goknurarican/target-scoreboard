# VantAI Target Scoreboard

Modality-aware target scoring system built on Open Targets data.
<img width="943" height="271" alt="Screenshot 2025-08-21 at 11 14 20" src="https://github.com/user-attachments/assets/f0f8ac1d-1af7-454d-9fc8-6e56e4625900" />

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
