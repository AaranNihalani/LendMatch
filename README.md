# LendMatch

LendMatch is a responsible lending triage web app built on the local LendingClub accepted and rejected application CSVs. It gives charities a presentable workflow for screening loan applicants, estimating risk, and matching suitable finance partners before a formal lender referral.

## What Is Included

- FastAPI backend serving the API and web app.
- Rebuilt scikit-learn training pipeline in `src/lendmatch_model.py`.
- Three trained models:
  - approval likelihood from accepted/rejected applications
  - default probability from closed accepted loans
  - estimated APR from funded LendingClub loans
- Professional static UI in `docs/` with model health, risk metrics, lender matches, and caseworker guidance.
- Model card at `/model-card` with data source and validation metrics.

## Data

The LendingClub files are already present in this workspace:

- `data/accepted_2007_to_2018Q4.csv`
- `data/rejected_2007_to_2018Q4.csv`

You do not need to download anything for the current build. If you move the project, keep those two files in `data/` or `data/raw/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If dependencies are already installed on your machine, you can skip the install step.

## Train Models

```bash
python -m src.lendmatch_model
```

This creates:

- `models/lendmatch_artifacts.joblib`
- `models/model_card.json`

To train faster while experimenting:

```bash
LENDMATCH_ACCEPTED_SAMPLE=50000 LENDMATCH_REJECTED_SAMPLE=50000 python -m src.lendmatch_model
```

Current model card metrics from the local data sample:

- Approval ROC-AUC: `0.9861`
- Default ROC-AUC: `0.7220`
- Interest-rate MAE: `2.015` APR points
- Interest-rate R2: `0.6135`

## Run The Web App

```bash
uvicorn api.index:app --reload --port 8000
```

Open `http://localhost:8000`.

Useful endpoints:

- `GET /health`
- `GET /model-card`
- `POST /predict`

## Test

```bash
python tests/test_prediction.py
```

## Adoption Notes For Charities

This is a polished decision-support prototype, not a regulated automated credit decision system. Before a charity adopts it for real applicants, add human review, applicant consent, audit logging, fairness testing, local compliance review, and a process for adverse-action explanations.
