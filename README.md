# Data-Driven Loan Matching and Credit Decision Platform

## Overview
This project is a research-grade applied data science system that predicts loan outcomes (default risk, interest rate, approval probability) and recommends optimal lenders. It features a full ML pipeline, a FastAPI backend, and a lightweight static frontend.

## Data Source
The project supports:
- **Real LendingClub-style Accepted/Rejected CSVs** (preferred if available)
- **Synthetic LendingClub-like data** (auto-generated if raw CSVs are missing)

If both real and synthetic CSVs exist in `data/raw/`, the pipeline prefers the real LendingClub 2007–2018Q4 files.

### Using Real Data
Place both files in `data/raw/`:
- `accepted_*.csv` (accepted / funded loans)
- `rejected_*.csv` (rejected applications)

Then run:
```bash
python src/data_pipeline.py
```

### Using Synthetic Data
If `data/raw/` does not contain accepted/rejected CSVs, running:
```bash
python src/data_pipeline.py
```
will generate:
- `data/raw/accepted_synthetic.csv`
- `data/raw/rejected_synthetic.csv`

## Project Structure
```
loan-matching-platform/
├── api/
│   └── index.py               # FastAPI service (also serves docs/)
├── data/
│   ├── raw/                   # Raw data (synthetic or real)
│   └── processed/             # Cleaned and engineered data
├── docs/                      # Static frontend assets
│   ├── index.html             # Main entry point
│   ├── styles.css             # Styles
│   └── main.js                # Frontend logic
├── models/
│   ├── artifacts/             # Scalers and encoders
│   └── ...                    # Trained models (pkl)
├── reports/
│   ├── figures/               # EDA and SHAP plots
│   └── project_report.md      # Final project report
├── src/
│   ├── generate_data.py       # Synthetic data generation
│   ├── data_pipeline.py       # Data cleaning (hybrid mode)
│   ├── feature_engineering.py # Feature engineering
│   ├── eda.py                 # Exploratory Data Analysis
│   ├── model_training.py      # Model training & evaluation
│   ├── lender_matching.py     # Matching algorithm
│   └── prediction_service.py  # Inference service
├── tests/
│   └── test_prediction.py     # Integration test
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Pipeline (End-to-End)**
   To generate data, process it, and train models:
   ```bash
   # Generate/Load data
   python src/data_pipeline.py
   
   # Engineer features
   python src/feature_engineering.py
   
   # Perform EDA
   python src/eda.py
   
   # Train models
   python src/model_training.py
   ```

### Running the Application

1. **Start the Web Application**
   The backend now serves the frontend directly.
   Open a terminal and run:
   ```bash
   uvicorn api.index:app --reload --port 8000
   ```
   
2. **Access the Platform**
   Open your browser and go to `http://localhost:8000`.

## Features
- **Prediction**: Real-time prediction of Default Risk, Interest Rate, and Approval Probability.
- **Lender Matching**: Recommendations based on borrower profile and lender risk appetite.
- **Interpretability**: SHAP values explaining why a specific prediction was made.
- **Interactive UI**: Clean, modern HTML/JS interface.

## Reports
- Full project report: `reports/project_report.md`
- Figures: `reports/figures/`
