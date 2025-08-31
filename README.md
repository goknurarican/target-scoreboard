# VantAI Target Scoreboard (Demo)

A lightweight, **modality‑aware target prioritization** demo that integrates real public datasets and transparent scoring to help with early shortlisting. Built end‑to‑end (API + UI) to showcase explainability, sensitivity, and shareable runs.

> **Note (scope):** This is a two‑week, self‑initiated **demo** for evaluation purposes. Genetics & PPI are backed by real data; **safety, pathway, and modality‑fit are heuristic/proxy layers** and called out as such below.

---

## 🎥 Demo Video

90‑second walkthrough → **[https://youtu.be/L896NhYdsZA](https://youtu.be/L896NhYdsZA)**

[![Target Scoreboard Demo](https://img.youtube.com/vi/L896NhYdsZA/0.jpg)](https://youtu.be/L896NhYdsZA)

---

## What it does (at a glance)

* **Explainable multi‑channel score** per target: **Genetics**, **PPI proximity**, **Pathway**, **Safety**, **Modality‑fit** → weighted and normalized (default weights shown in UI).
* **Why‑this‑rank? panel** with channel contributions, short interpretations, and **clickable evidence** (e.g., Open Targets links).
* **Sensitivity tools:** weight impact, **channel ablation**, and rank **stability** (Dirichlet jitter) to see how robust a shortlist is.
* **Benchmark (lite):** demo ground truths (NSCLC/Breast) with **Precision\@k** and **AUC‑PR** for sanity checks.
* **Shareable runs:** full state (disease, targets, weights) is encoded in the URL for easy review.

---

## What’s real vs. proxy

| Channel          | Data source / method                             | Status           |
| ---------------- | ------------------------------------------------ | ---------------- |
| **Genetics**     | Open Targets disease–gene associations           | **Real**         |
| **PPI**          | STRING‑based proximity / neighbors               | **Real**         |
| **Pathway**      | ORA on a limited set (to be expanded)            | **Semi / Proxy** |
| **Safety**       | Off‑tissue/tissue specificity heuristic          | **Proxy**        |
| **Modality‑fit** | E3 co‑expr, ternary hint, PPI hotspot heuristics | **Proxy**        |

> The demo is intended for **early triage/shortlisting**, not final biological decisions.

---

## Screens & examples

**Overview:** score distribution & run metrics

<img width="917" height="565" alt="Screenshot 2025-08-21 at 11 50 54" src="https://github.com/user-attachments/assets/d6fc7ea5-c739-4f50-9ccd-4773eafc903c" />

**PPI neighbors / network:**

<img width="874" height="699" alt="Screenshot 2025-08-21 at 11 53 36" src="https://github.com/user-attachments/assets/c89fe72c-6bb1-4490-a73b-6241971d4221" />

**Modality‑fit (heuristic):**

<img width="900" height="659" alt="Screenshot 2025-08-21 at 11 55 24" src="https://github.com/user-attachments/assets/cab5e179-94ed-49b4-a75b-c45ba35a85d2" />

**Benchmark (lite):**

<img width="783" height="664" alt="Screenshot 2025-08-21 at 11 56 04" src="https://github.com/user-attachments/assets/eadbaf44-0fe2-48c7-b7b7-2243417a35ea" />

---

## Data sources (current)

* **Open Targets (GraphQL v4)** – disease–gene associations
* **STRING v12** – protein–protein interactions
* **Reactome (curated)** – pathway references (limited set in demo)

Versions and timestamps are shown in the app footer / API responses where applicable.

---

## Quick start

### 1) Clone & env

```bash
git clone https://github.com/vantai/target-scoreboard.git
cd target-scoreboard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Run API (FastAPI)

```bash
export API_PORT=8001
python -m uvicorn app.main:app --host 0.0.0.0 --port $API_PORT --reload
```

**Sanity check (Genetics should populate):**

```bash
curl -sS -X POST http://localhost:8001/score \
 -H 'Content-Type: application/json' \
 -d '{
   "disease": "EFO_0003060",                  
   "targets": ["EGFR","ERBB2","ALK","KRAS","MET"],
   "weights": {"genetics":0.35, "ppi":0.25, "pathway":0.20, "safety":0.10, "modality_fit":0.10}
 }' | jq '.targets[0] | {target, genetics: .channels.genetics.score, evidence: .channels.genetics.evidence[0].url}'
```

### 3) Run UI (Streamlit)

```bash
export STREAMLIT_PORT=8501
streamlit run dashboard/ui_app.py --server.port $STREAMLIT_PORT
```

The UI will detect local vs. hosted API automatically. You can share the analysis via the **Share** section (URL includes disease, targets, and weights).

---

## Architecture

**Backend (FastAPI)**

* REST endpoints; multi‑source integration and light caching
* Scoring pipeline with per‑channel components
* Sensitivity: **/sensitivity/ablation**, stability
* JSON/CSV export with metadata (where applicable)

**Frontend (Streamlit)**

* Tabs: **Overview · Rankings · Explain · Evidence · Sensitivity · Benchmark**
* Explain panel: channel contributions + short interpretations + evidence links
* URL‑state (encode/decode/update) for reproducible runs

---

## Troubleshooting

* **Genetics = 0?** Ensure the disease EFO is valid (e.g., NSCLC = `EFO_0003060`). If OT returns `null`, fix the ID and retry.
* **Weights sum ≠ 1?** The UI warns; you can normalize or proceed.
* **No pathway hits?** The demo uses a limited set; see roadmap below to expand.

---

## Limitations & roadmap

1. **Safety 1.0:** integrate GTEx/HPA tissue‑specificity + LOEUF/pLI + DepMap essentiality → combined risk score.
2. **Pathway 1.0:** expand to Reactome/KEGG/MSigDB with proper ORA/GSEA, p‑value & FDR, background controls.
3. **Modality‑fit 1.0:** structure/localization signals (cell compartment, pocket druggability, E3 atlas).
4. **Qualitative → score:** small LLM‑assisted “claim strength / consensus / recency” metric from PMIDs.
5. **PPI provider adapter:** pluggable backend (STRING or proprietary) via a simple interface.

---

## Recent fixes (Aug–Sep 2025)

* **Correct EFO mapping** (e.g., NSCLC → `EFO_0003060`); Genetics now returns real OT scores with evidence links.
* **UI ↔ backend alignment** for `channels.genetics.score` across Rankings & Explain.
* **Sensitivity/Ablation** endpoints wired in the UI; stability uses sample‑size controls to avoid timeouts.
* **Shareable URL state** (targets + weights), clearer API errors, and basic health checks.

---

## License

All rights reserved. Shared for evaluation only under `LICENSE-EVALUATION.md`.
