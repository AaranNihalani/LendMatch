# Data-Driven Loan Matching and Credit Decision Platform

## 1. Motivation
The goal of this project is to build a research-grade applied data science system that predicts loan outcomes (default risk, interest rate, approval probability) and recommends optimal lenders. By leveraging machine learning on historical lending data, we aim to automate credit decisions, reduce risk for lenders, and find the best financial products for borrowers.

## 2. Dataset
We utilized synthetic data modeled after the **LendingClub Loan Data (2007–2018)**.
- **Size**: 10,000 samples (synthetic generation).
- **Key Variables**: `loan_amnt`, `term`, `int_rate`, `emp_length`, `annual_inc`, `home_ownership`, `purpose`, `dti`, `fico_range_low`, `fico_range_high`, `revol_bal`, `revol_util`, `loan_status`.
- **Target Variables**:
  - `default_flag`: Derived from `loan_status` (Charged Off vs Fully Paid).
  - `int_rate`: Continuous variable for interest rate prediction.
  - `approval_probability`: Approximated using loan status outcomes.

## 3. Feature Engineering
We engineered several financial ratios to enhance model performance:
1. **Loan-to-Income Ratio**: `loan_amnt / annual_inc` - Measures affordability.
2. **Credit Score Midpoint**: Average of FICO low and high ranges.
3. **Revolving Credit Ratio**: `revol_bal / loan_amnt` - Indicates existing debt burden relative to new loan.
4. **Log Income**: `log(annual_inc)` - Normalizes the skewed income distribution.
5. **Debt Burden Index**: `dti * loan_to_income_ratio` - Composite risk metric.

Categorical variables (`term`, `emp_length`, `home_ownership`, `purpose`) were One-Hot Encoded. Continuous variables were standardized using `StandardScaler`.

## 4. Model Design
We trained three distinct models to serve the platform's objectives:

### A. Default Prediction Model (Classification)
- **Goal**: Predict probability of loan default.
- **Algorithms Tested**: Logistic Regression, Random Forest, XGBoost.
- **Selection Metric**: ROC-AUC.
- **Best Model**: Logistic Regression (AUC ~0.98). *Note: High performance due to synthetic data design.*

### B. Interest Rate Model (Regression)
- **Goal**: Predict the fair interest rate for a loan.
- **Algorithms Tested**: Linear Regression, Gradient Boosting, XGBoost.
- **Selection Metric**: RMSE.
- **Best Model**: Linear Regression (RMSE ~1.01).

### C. Approval Model (Classification)
- **Goal**: Estimate likelihood of loan approval.
- **Approach**: Modeled as the inverse of default risk (Probability of 'Fully Paid'), trained using Logistic Regression.

## 5. Evaluation
- **Default Model**: Achieved high discrimination (AUC > 0.95), effectively separating good and bad loans.
- **Interest Rate Model**: Predictions are within ~1% of the actual rate on average (MAE ~0.8), providing reliable pricing guidance.
- **Evaluation Plots**: ROC curves and confusion matrices are generated in `reports/figures/`.

## 6. Interpretability
We used **SHAP (SHapley Additive exPlanations)** to interpret the models.
- **Global Importance**: `fico_range_low`, `dti`, and `loan_amnt` were consistently the top drivers of default risk.
- **Local Explanations**: The system can explain individual predictions (e.g., "High DTI increased default risk by 15%").
- **Plots**: SHAP summary plots available in `reports/figures/`.

## 7. Lender Matching Algorithm
We implemented a matching engine that ranks lenders based on a composite score:
`score = (0.5 * approval_prob) - (0.3 * interest_rate) - (0.2 * default_risk)`

We simulated three lender profiles:
1. **Conservative Bank**: Low risk tolerance, lower rates, strict approval.
2. **Balanced FinTech**: Medium risk, market rates.
3. **High Yield Capital**: High risk tolerance, higher rates, lenient approval.

The algorithm adjusts the base predictions (e.g., interest rate spread) for each lender and returns the best fit for the borrower.

## 8. Economic Implications
- **Risk Reduction**: Automated default prediction allows lenders to minimize bad debt.
- **Fair Pricing**: Data-driven interest rate modeling ensuring rates reflect actual risk.
- **Market Efficiency**: The matching platform reduces search friction, connecting borrowers with the most suitable lenders instantly.
