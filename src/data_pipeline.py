import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import re

class DataPipeline:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def run(self):
        print("Starting Data Pipeline...")
        
        # 1. Identify Files
        accepted_files = [f for f in os.listdir(self.raw_dir) if 'accepted' in f.lower() and f.endswith('.csv')]
        rejected_files = [f for f in os.listdir(self.raw_dir) if 'rejected' in f.lower() and f.endswith('.csv')]
        
        if not accepted_files or not rejected_files:
            print("Error: Could not find both 'accepted_*.csv' and 'rejected_*.csv' in data/raw/")
            return
            
        accepted_path = os.path.join(self.raw_dir, accepted_files[0])
        rejected_path = os.path.join(self.raw_dir, rejected_files[0])
        
        print(f"Using Accepted Data: {accepted_path}")
        print(f"Using Rejected Data: {rejected_path}")
        
        # 2. Process Accepted Data (For Default & Interest Rate Models)
        self.process_accepted_data(accepted_path)
        
        # 3. Process Approval Data (Combined Accepted + Rejected)
        self.process_approval_data(accepted_path, rejected_path)
        
        print("Data Pipeline Completed Successfully.")

    def process_accepted_data(self, path):
        print("\n--- Processing Accepted Data (Rich Features) ---")
        # Read with low_memory=False to avoid DtypeWarning, or specify dtypes
        df = pd.read_csv(path, low_memory=False)
        print(f"Original Accepted Shape: {df.shape}")
        
        # Filter for relevant loan statuses (Closed loans for training outcome models)
        # We want to predict Default vs Fully Paid
        # Current, In Grace Period, etc. are not final outcomes for binary classification of default risk
        # BUT for Interest Rate model, we can use all funded loans.
        # Let's keep all for now, and filter in Feature Engineering or Training.
        # Actually, let's filter rows where loan_status is empty (if any)
        df = df.dropna(subset=['loan_status'])
        
        # Create Target Variables
        # 1. Default Flag: 1 if Bad (Charged Off, Default, Late), 0 if Good (Fully Paid)
        # We ignore 'Current' for the Default Model training usually, or treat as 0? 
        # Best practice: Train on closed loans (Fully Paid vs Charged Off).
        # We will create a column 'is_closed' to easily filter later.
        
        def classify_status(status):
            if pd.isna(status): return np.nan
            status = status.lower()
            if 'fully paid' in status:
                return 0 # Good
            elif 'charged off' in status or 'default' in status:
                return 1 # Bad
            else:
                return np.nan # Current, Late, Grace Period -> Not a final binary outcome yet

        df['default_target'] = df['loan_status'].apply(classify_status)
        
        # 2. Interest Rate is already numeric 'int_rate'
        
        # Basic Cleaning
        # Convert term to numeric
        if 'term' in df.columns:
            df['term_months'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
        
        # Emp Length to numeric
        if 'emp_length' in df.columns:
            df['emp_length_num'] = df['emp_length'].apply(self._parse_emp_length)
            
        # Drop columns with >80% missing
        threshold = len(df) * 0.2
        df = df.dropna(thresh=threshold, axis=1)
        
        # Save "Rich" dataset for Default/Interest models
        output_path = os.path.join(self.processed_dir, "accepted_clean.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved Accepted Clean Data to {output_path} (Shape: {df.shape})")

    def process_approval_data(self, acc_path, rej_path):
        print("\n--- Processing Approval Data (Combined) ---")
        
        # Load Rejected
        df_rej = pd.read_csv(rej_path, low_memory=False)
        print(f"Original Rejected Shape: {df_rej.shape}")
        
        # Load Accepted (Just the columns we need to map)
        # Accepted Columns: loan_amnt, dti, addr_state, emp_length, policy_code, fico_range_low, fico_range_high
        # We also need a date if we want to split by time, but random split is ok for now.
        acc_cols = ['loan_amnt', 'dti', 'addr_state', 'emp_length', 'policy_code', 'fico_range_low', 'fico_range_high']
        df_acc = pd.read_csv(acc_path, usecols=lambda c: c in acc_cols, low_memory=False)
        
        # Standardize Accepted
        df_acc['risk_score'] = (df_acc['fico_range_low'] + df_acc['fico_range_high']) / 2
        df_acc = df_acc.rename(columns={
            'loan_amnt': 'amount',
            'addr_state': 'state'
        })
        df_acc['accepted'] = 1
        # Select common columns
        # Exclude policy_code as it directly leaks acceptance status (1=Accepted, 0/2=Rejected)
        common_cols = ['amount', 'risk_score', 'dti', 'state', 'emp_length', 'accepted']
        df_acc = df_acc[common_cols]
        
        # Standardize Rejected
        # Rejected Columns: Amount Requested, Risk_Score, Debt-To-Income Ratio, State, Employment Length, Policy Code
        df_rej = df_rej.rename(columns={
            'Amount Requested': 'amount',
            'Risk_Score': 'risk_score',
            'Debt-To-Income Ratio': 'dti',
            'State': 'state',
            'Employment Length': 'emp_length'
            # 'Policy Code': 'policy_code' # Dropped
        })
        df_rej['accepted'] = 0
        
        # Clean Rejected DTI (remove %)
        df_rej['dti'] = df_rej['dti'].astype(str).str.rstrip('%')
        df_rej['dti'] = pd.to_numeric(df_rej['dti'], errors='coerce')
        
        # Filter to common columns
        df_rej = df_rej[common_cols]
        
        # Combine
        print(f"Combining {len(df_acc)} Accepted and {len(df_rej)} Rejected records...")
        df_combined = pd.concat([df_acc, df_rej], axis=0, ignore_index=True)
        
        # Clean Combined
        # Drop rows with missing Amount or Risk Score (crucial for approval)
        df_combined.dropna(subset=['amount', 'risk_score'], inplace=True)
        
        # Emp Length cleaning
        df_combined['emp_length_num'] = df_combined['emp_length'].apply(self._parse_emp_length)
        df_combined.drop(columns=['emp_length'], inplace=True) # Use numeric version
        
        # Fill missing DTI with median
        df_combined['dti'].fillna(df_combined['dti'].median(), inplace=True)
        
        output_path = os.path.join(self.processed_dir, "approval_data.csv")
        df_combined.to_csv(output_path, index=False)
        print(f"Saved Approval Data to {output_path} (Shape: {df_combined.shape})")
        
    def _parse_emp_length(self, x):
        if pd.isna(x) or x == 'Unknown': return 0
        x = str(x)
        if '<' in x: return 0
        if '+' in x: return 10
        # Extract first number found
        nums = re.findall(r'\d+', x)
        if nums:
            return int(nums[0])
        return 0

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
