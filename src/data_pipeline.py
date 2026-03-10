import pandas as pd
import numpy as np
import os
import re
try:
    from src.generate_data import generate_synthetic_lendingclub_like_data
except ModuleNotFoundError:
    from generate_data import generate_synthetic_lendingclub_like_data

class DataPipeline:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def _select_raw_csv(self, filenames, prefer_regexes=None, avoid_substrings=None):
        if not filenames:
            return None
        prefer_regexes = prefer_regexes or []
        avoid_substrings = [s.lower() for s in (avoid_substrings or [])]

        scored = []
        for name in filenames:
            lower = name.lower()
            avoid = any(s in lower for s in avoid_substrings)
            prefer = any(re.search(rx, name, flags=re.IGNORECASE) for rx in prefer_regexes)
            path = os.path.join(self.raw_dir, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            scored.append(((prefer, not avoid, size), name))

        scored.sort(reverse=True)
        return scored[0][1]
        
    def run(self):
        print("Starting Data Pipeline...")
        
        # 1. Identify Files
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir, exist_ok=True)

        accepted_files = [f for f in os.listdir(self.raw_dir) if "accepted" in f.lower() and f.endswith(".csv")]
        rejected_files = [f for f in os.listdir(self.raw_dir) if "rejected" in f.lower() and f.endswith(".csv")]
        
        if not accepted_files or not rejected_files:
            print("No accepted/rejected raw CSVs found. Generating synthetic LendingClub-like data...")
            generated = generate_synthetic_lendingclub_like_data(output_dir=self.raw_dir)
            accepted_files = [os.path.basename(generated["accepted"])]
            rejected_files = [os.path.basename(generated["rejected"])]

        accepted_file = self._select_raw_csv(
            accepted_files,
            prefer_regexes=[r"2007.*2018", r"accepted_2007_to_2018q4"],
            avoid_substrings=["synthetic"],
        )
        rejected_file = self._select_raw_csv(
            rejected_files,
            prefer_regexes=[r"2007.*2018", r"rejected_2007_to_2018q4"],
            avoid_substrings=["synthetic"],
        )
        accepted_path = os.path.join(self.raw_dir, accepted_file)
        rejected_path = os.path.join(self.raw_dir, rejected_file)
        
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
        header_cols = pd.read_csv(path, nrows=0).columns.tolist()
        desired_cols = [
            "loan_status",
            "loan_amnt",
            "term",
            "int_rate",
            "installment",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "fico_range_low",
            "fico_range_high",
            "inq_last_6mths",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "total_acc",
            "collections_12_mths_ex_med",
            "acc_now_delinq",
            "tot_coll_amt",
            "tot_cur_bal",
            "total_rev_hi_lim",
            "mort_acc",
            "pub_rec_bankruptcies",
            "tax_liens",
            "home_ownership",
            "verification_status",
            "purpose",
            "addr_state",
            "initial_list_status",
            "application_type",
            "emp_length",
            "earliest_cr_line",
            "grade",
            "sub_grade",
        ]
        usecols = [c for c in desired_cols if c in header_cols]
        print(f"Reading Accepted Columns: {len(usecols)}")
        
        # Filter for relevant loan statuses (Closed loans for training outcome models)
        # We want to predict Default vs Fully Paid
        # Current, In Grace Period, etc. are not final outcomes for binary classification of default risk
        # BUT for Interest Rate model, we can use all funded loans.
        # Let's keep all for now, and filter in Feature Engineering or Training.
        # Actually, let's filter rows where loan_status is empty (if any)
        
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
        percent_cols = [c for c in ["int_rate", "revol_util"] if c in usecols]
        numeric_cols = [
            "loan_amnt",
            "installment",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "fico_range_low",
            "fico_range_high",
            "inq_last_6mths",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "total_acc",
            "collections_12_mths_ex_med",
            "acc_now_delinq",
            "tot_coll_amt",
            "tot_cur_bal",
            "total_rev_hi_lim",
            "mort_acc",
            "pub_rec_bankruptcies",
            "tax_liens",
        ]
        numeric_cols = [c for c in numeric_cols if c in usecols]

        output_path = os.path.join(self.processed_dir, "accepted_clean.csv")
        wrote_header = False
        total_rows = 0

        for chunk in pd.read_csv(path, low_memory=False, usecols=usecols, chunksize=200000):
            if "loan_status" in chunk.columns:
                chunk = chunk.dropna(subset=["loan_status"])
                chunk["default_target"] = chunk["loan_status"].astype(str).str.lower().apply(classify_status)

            for c in percent_cols:
                chunk[c] = chunk[c].astype(str).str.rstrip("%")
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            for c in numeric_cols:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            if "term" in chunk.columns:
                chunk["term_months"] = pd.to_numeric(chunk["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

            if "emp_length" in chunk.columns:
                chunk["emp_length_num"] = chunk["emp_length"].apply(self._parse_emp_length)

            chunk.to_csv(output_path, index=False, mode="w" if not wrote_header else "a", header=not wrote_header)
            wrote_header = True
            total_rows += len(chunk)

        print(f"Saved Accepted Clean Data to {output_path} (Rows: {total_rows})")

    def process_approval_data(self, acc_path, rej_path):
        print("\n--- Processing Approval Data (Combined) ---")
        
        # Load Rejected
        rej_usecols = [
            "Amount Requested",
            "Risk_Score",
            "Debt-To-Income Ratio",
            "State",
            "Employment Length",
        ]
        rej_header = pd.read_csv(rej_path, nrows=0).columns.tolist()
        rej_usecols = [c for c in rej_usecols if c in rej_header]
        df_rej = pd.read_csv(rej_path, low_memory=False, usecols=rej_usecols)
        print(f"Original Rejected Shape: {df_rej.shape}")
        
        # Load Accepted (Just the columns we need to map)
        # Accepted Columns: loan_amnt, dti, addr_state, emp_length, policy_code, fico_range_low, fico_range_high
        # We also need a date if we want to split by time, but random split is ok for now.
        acc_cols = ['loan_amnt', 'dti', 'addr_state', 'emp_length', 'policy_code', 'fico_range_low', 'fico_range_high']
        df_acc = pd.read_csv(acc_path, usecols=lambda c: c in acc_cols, low_memory=False)
        
        # Standardize Accepted
        for c in ["fico_range_low", "fico_range_high", "loan_amnt", "dti"]:
            if c in df_acc.columns:
                df_acc[c] = pd.to_numeric(df_acc[c], errors="coerce")
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
        if "amount" in df_rej.columns:
            df_rej["amount"] = pd.to_numeric(df_rej["amount"], errors="coerce")
        if "risk_score" in df_rej.columns:
            df_rej["risk_score"] = pd.to_numeric(df_rej["risk_score"], errors="coerce")
        
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
        df_combined["dti"] = df_combined["dti"].fillna(df_combined["dti"].median())
        
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
