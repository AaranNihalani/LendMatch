# Data-Driven Loan Matching and Credit Decision Platform

## Overview
This project is a research-grade applied data science system that predicts loan outcomes (default risk, interest rate, approval probability) and recommends optimal lenders. It features a full ML pipeline, a FastAPI backend, and a Streamlit frontend.

## Data Source
The project is configured to work with either **Synthetic Data** (default) or **Real LendingClub Data**.

### Using Real Data
1.  Download the LendingClub dataset (e.g., `loan.csv`) from Kaggle or other sources.
2.  Place the file at `data/raw/loan.csv`.
3.  Run the pipeline: `python src/data_pipeline.py`.
    - The system will automatically detect the real data file and use it instead of generating synthetic data.

### Using Synthetic Data
If no `loan.csv` is found, the system generates 10,000 synthetic records mimicking the LendingClub schema to ensure reproducibility and immediate usability.

## Project Structure
```
loan-matching-platform/
├── backend/
│   └── api.py                 # FastAPI service
├── data/
│   ├── raw/                   # Raw data (synthetic or real)
│   └── processed/             # Cleaned and engineered data
├── docs/                  # Frontend assets (served by GitHub Pages)
│   ├── index.html         # Main entry point
│   ├── styles.css         # Styles
│   └── main.js            # Frontend logic
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
   python3 -m venv venv
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
   uvicorn backend.api:app --reload --port 8000
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
