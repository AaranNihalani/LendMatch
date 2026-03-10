import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def _load_sampled_csv(input_path, max_rows=250000, chunksize=200000, seed=42):
    df_parts = []
    remaining = int(max_rows)
    for chunk in pd.read_csv(input_path, low_memory=False, chunksize=chunksize):
        if remaining <= 0:
            break
        take = min(len(chunk), remaining)
        df_parts.append(chunk.sample(n=take, random_state=seed))
        remaining -= take
    if not df_parts:
        return pd.DataFrame()
    return pd.concat(df_parts, axis=0, ignore_index=True)

def perform_eda(input_path, output_dir, max_rows=250000):
    print(f"Loading data from {input_path}...")
    df = _load_sampled_csv(input_path, max_rows=max_rows)
    
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["figure.dpi"] = 120
    
    # 1. Loan Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["loan_amnt"].dropna(), kde=True, bins=40, color="#2b6cb0")
    plt.title("Distribution of Loan Amounts")
    plt.xlabel("Loan Amount ($)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loan_distribution.png"), bbox_inches="tight")
    plt.close()
    
    # 2. Interest Rate vs Credit Score
    plt.figure(figsize=(10, 6))
    if "credit_score_midpoint" not in df.columns and "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["credit_score_midpoint"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        
    plot_df = df.dropna(subset=["credit_score_midpoint", "int_rate"]).copy()
    sns.scatterplot(x="credit_score_midpoint", y="int_rate", data=plot_df, alpha=0.35, s=18, color="#2f855a")
    sns.regplot(x="credit_score_midpoint", y="int_rate", data=plot_df, scatter=False, color="#22543d", line_kws={"linewidth": 2})
    plt.title("Interest Rate vs. FICO Score")
    plt.xlabel("FICO Score (midpoint)")
    plt.ylabel("Interest Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "int_rate_vs_fico.png"), bbox_inches="tight")
    plt.close()
    
    # 3. Default Rate by Income
    if "default_flag" not in df.columns and "default_target" in df.columns:
        df["default_flag"] = df["default_target"]

    income_df = df.dropna(subset=["annual_inc", "default_flag"]).copy()
    income_df["income_bin"] = pd.qcut(income_df["annual_inc"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    default_rate_income = income_df.groupby("income_bin", observed=True)["default_flag"].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="income_bin", y="default_flag", data=default_rate_income, color="#805ad5")
    plt.title("Default Rate by Income Group")
    plt.xlabel("Income Quintile")
    plt.ylabel("Default Rate")
    plt.ylim(0, min(0.6, float(default_rate_income["default_flag"].max() * 1.35) if len(default_rate_income) else 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "default_rate_by_income.png"), bbox_inches="tight")
    plt.close()
    
    # 4. Default Rate by Loan Purpose
    purpose_df = df.dropna(subset=["purpose", "default_flag"]).copy()
    default_rate_purpose = (
        purpose_df.groupby("purpose", observed=True)["default_flag"].mean().sort_values(ascending=False).reset_index()
    )
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="default_flag", y="purpose", data=default_rate_purpose, color="#dd6b20")
    plt.title("Default Rate by Loan Purpose")
    plt.xlabel("Default Rate")
    plt.ylabel("Purpose")
    plt.xlim(0, min(0.6, float(default_rate_purpose["default_flag"].max() * 1.35) if len(default_rate_purpose) else 0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "default_rate_by_purpose.png"), bbox_inches="tight")
    plt.close()
    
    # 5. Correlation Matrix
    # Select numeric columns
    numeric_df = df.select_dtypes(include=["float64", "int64", "float32", "int32"])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=False, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), bbox_inches="tight")
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    perform_eda("data/processed/accepted_clean.csv", "reports/figures")
