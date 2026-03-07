import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a style
    plt.style.use('ggplot')
    
    # 1. Loan Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['loan_amnt'], kde=True)
    plt.title('Distribution of Loan Amounts')
    plt.xlabel('Loan Amount')
    plt.savefig(os.path.join(output_dir, 'loan_distribution.png'))
    plt.close()
    
    # 2. Interest Rate vs Credit Score
    # We need to reconstruct credit score midpoint if it's scaled or use the raw data
    # Ideally EDA is done on raw/cleaned data, not the scaled one.
    # Let's check if we should load the raw/cleaned data instead.
    # The input_path is 'data/processed/cleaned_data.csv' (before scaling)
    
    plt.figure(figsize=(10, 6))
    # Calculate midpoint if not present (it was added in FE, but let's assume we use cleaned_data)
    if 'credit_score_midpoint' not in df.columns:
        df['credit_score_midpoint'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        
    sns.scatterplot(x='credit_score_midpoint', y='int_rate', data=df, alpha=0.5)
    plt.title('Interest Rate vs. FICO Score')
    plt.savefig(os.path.join(output_dir, 'int_rate_vs_fico.png'))
    plt.close()
    
    # 3. Default Rate by Income
    # Bin income
    df['income_bin'] = pd.qcut(df['annual_inc'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    default_rate_income = df.groupby('income_bin', observed=True)['default_flag'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='income_bin', y='default_flag', data=default_rate_income)
    plt.title('Default Rate by Income Group')
    plt.ylabel('Default Rate')
    plt.savefig(os.path.join(output_dir, 'default_rate_by_income.png'))
    plt.close()
    
    # 4. Default Rate by Loan Purpose
    default_rate_purpose = df.groupby('purpose', observed=True)['default_flag'].mean().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='default_flag', y='purpose', data=default_rate_purpose)
    plt.title('Default Rate by Loan Purpose')
    plt.xlabel('Default Rate')
    plt.savefig(os.path.join(output_dir, 'default_rate_by_purpose.png'))
    plt.close()
    
    # 5. Correlation Matrix
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    # Use cleaned_data.csv because it has the original values (before scaling/encoding)
    perform_eda("data/processed/cleaned_data.csv", "reports/figures")
