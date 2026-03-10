import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import re

class FeatureEngineering:
    def __init__(self, processed_dir="data/processed", models_dir="models"):
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.artifacts_dir = os.path.join(models_dir, "artifacts")
        # Ensure directories exist
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.artifacts_dir, exist_ok=True)
            print(f"Directories checked/created: {self.models_dir}, {self.artifacts_dir}")
        except Exception as e:
            print(f"Error creating directories: {e}")

    def _is_stale(self, output_path, input_paths):
        if not os.path.exists(output_path):
            return True
        try:
            output_mtime = os.path.getmtime(output_path)
        except OSError:
            return True
        for p in input_paths:
            if not os.path.exists(p):
                continue
            try:
                if os.path.getmtime(p) > output_mtime:
                    return True
            except OSError:
                return True
        return False

    def _load_sampled_csv(self, input_path, max_rows=200000, chunksize=200000, seed=42):
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

    def _load_stratified_binary_sample_csv(
        self,
        input_path,
        label_col,
        n_per_class=100000,
        positive_value=1,
        negative_value=0,
        chunksize=200000,
        seed=42,
    ):
        rng = np.random.default_rng(seed)
        pos_parts = []
        neg_parts = []
        pos_needed = int(n_per_class)
        neg_needed = int(n_per_class)

        for chunk in pd.read_csv(input_path, low_memory=False, chunksize=chunksize):
            if label_col not in chunk.columns:
                continue

            labels = pd.to_numeric(chunk[label_col], errors="coerce")
            pos = chunk[labels == positive_value]
            neg = chunk[labels == negative_value]

            if pos_needed > 0 and len(pos) > 0:
                take = min(pos_needed, len(pos))
                pos_parts.append(pos.sample(n=take, random_state=int(rng.integers(0, 1_000_000_000))))
                pos_needed -= take

            if neg_needed > 0 and len(neg) > 0:
                take = min(neg_needed, len(neg))
                neg_parts.append(neg.sample(n=take, random_state=int(rng.integers(0, 1_000_000_000))))
                neg_needed -= take

            if pos_needed <= 0 and neg_needed <= 0:
                break

        df = pd.concat(pos_parts + neg_parts, axis=0, ignore_index=True) if (pos_parts or neg_parts) else pd.DataFrame()
        if df.empty:
            return df
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
    def run(self):
        print("Starting Feature Engineering...")
        code_path = os.path.abspath(__file__)
        
        # 1. Process Accepted Data (Rich Features for Default/Interest)
        accepted_path = os.path.join(self.processed_dir, "accepted_clean.csv")
        full_train_path = os.path.join(self.processed_dir, "train_data_full.csv")
        full_preprocessor_path = os.path.join(self.artifacts_dir, "full_preprocessor.pkl")
        
        if os.path.exists(accepted_path) and (
            self._is_stale(full_train_path, [accepted_path, code_path])
            or self._is_stale(full_preprocessor_path, [accepted_path, code_path])
        ):
            self.process_accepted_features(accepted_path)
        elif os.path.exists(full_train_path) and os.path.exists(full_preprocessor_path):
            print(f"Skipping Accepted Data Processing: {full_train_path} is up-to-date.")
        else:
            print(f"Warning: {accepted_path} not found.")
            
        # 2. Process Approval Data (Combined)
        approval_path = os.path.join(self.processed_dir, "approval_data.csv")
        approval_train_path = os.path.join(self.processed_dir, "train_data_approval.csv")
        approval_preprocessor_path = os.path.join(self.artifacts_dir, "approval_preprocessor.pkl")
        
        if os.path.exists(approval_path) and (
            self._is_stale(approval_train_path, [approval_path, code_path])
            or self._is_stale(approval_preprocessor_path, [approval_path, code_path])
        ):
            self.process_approval_features(approval_path)
        elif os.path.exists(approval_train_path) and os.path.exists(approval_preprocessor_path):
            print(f"Skipping Approval Data Processing: {approval_train_path} is up-to-date.")
        else:
            print(f"Warning: {approval_path} not found.")
            
    def process_accepted_features(self, path):
        print("\n--- Engineering Features for Default/Interest Models ---")
        print("Loading Accepted Data (Sampling 200k rows)...")
        df = self._load_sampled_csv(path, max_rows=200000)
        
        # Preserve targets that might be transformed
        if 'int_rate' in df.columns:
            df['int_rate_target'] = df['int_rate']
        
        # --- Feature Creation ---
        # Dates
        # earliest_cr_line: Mon-Year (e.g., Aug-2003)
        if 'earliest_cr_line' in df.columns:
            # Parse date
            df['earliest_cr_line_dt'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
            # Calculate months since
            reference_date = pd.to_datetime('2019-01-01') # Post-dataset reference
            df['months_since_earliest_cr_line'] = (reference_date - df['earliest_cr_line_dt']).dt.days / 30.0
            df['months_since_earliest_cr_line'] = df['months_since_earliest_cr_line'].fillna(df['months_since_earliest_cr_line'].median())

        # FICO Average
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        
        # Ratios
        if 'annual_inc' in df.columns:
            df['annual_inc'] = df['annual_inc'].replace(0, 0.01) # Avoid div/0
            if 'loan_amnt' in df.columns:
                df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
            if 'installment' in df.columns:
                df['installment_to_income'] = (df['installment'] * 12) / df['annual_inc']
        
        if 'revol_bal' in df.columns and 'annual_inc' in df.columns:
            df['revol_bal_to_income'] = df['revol_bal'] / df['annual_inc']

        # Log Transforms for skewed data
        for col in ['annual_inc', 'revol_bal', 'total_acc']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])

        # --- Feature Selection ---
        # Numeric Features
        numeric_features = [
            'loan_amnt', 'term_months', 'int_rate', 'installment', 'annual_inc', 'dti', 
            'delinq_2yrs', 'fico_avg', 'inq_last_6mths', 'open_acc', 'pub_rec', 
            'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 
            'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
            'emp_length_num', 'months_since_earliest_cr_line',
            'loan_to_income', 'installment_to_income', 'revol_bal_to_income',
            'annual_inc_log', 'revol_bal_log', 'total_acc_log',
            'mort_acc', 'pub_rec_bankruptcies', 'tax_liens'
        ]
        
        # Categorical Features
        categorical_features = [
            'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'application_type'
        ]
        
        # Filter available columns
        numeric_features = [c for c in numeric_features if c in df.columns]
        categorical_features = [c for c in categorical_features if c in df.columns]
        
        print(f"Numeric Features: {len(numeric_features)}")
        print(f"Categorical Features: {len(categorical_features)}")
        
        # --- Targets ---
        targets = ['default_target', 'int_rate', 'grade', 'sub_grade']
        
        # Define Preprocessing Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep targets and other cols
        )
        
        # Fit and Transform
        X = df[numeric_features + categorical_features]
        # Keep targets separate
        y_cols = [c for c in df.columns if c not in X.columns]
        y = df[y_cols]
        
        preprocessor.set_output(transform="pandas")
        X_processed = preprocessor.fit_transform(X)
        
        # Save Preprocessor
        joblib.dump(preprocessor, os.path.join(self.artifacts_dir, 'full_preprocessor.pkl'))
        
        # Combine X and y
        # X_processed preserves index from X (if set_output="pandas"), so we must match y's index
        df_final = pd.concat([X_processed, y], axis=1)
        
        # Save
        output_path = os.path.join(self.processed_dir, "train_data_full.csv")
        df_final.to_csv(output_path, index=False)
        print(f"Saved Full Training Data to {output_path} (Shape: {df_final.shape})")
        
        # Save feature names list for later use
        feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
        joblib.dump(feature_names, os.path.join(self.artifacts_dir, 'feature_names.pkl'))


    def process_approval_features(self, path):
        print("\n--- Engineering Features for Approval Model ---")
        print("Loading Approval Data...")
        df = self._load_stratified_binary_sample_csv(path, label_col="accepted", n_per_class=100000)
        if df.empty:
            print("Approval data is empty after sampling.")
            return
        print(f"Loaded {len(df)} sampled rows.")
        
        # Features: amount, risk_score, dti, emp_length_num, state, policy_code
        # Target: accepted
        
        numeric_features = ['amount', 'risk_score', 'dti', 'emp_length_num']
        categorical_features = ['state'] # 'title' often too dirty in rejected data, skipping for now
        
        # Ensure columns exist
        for col in numeric_features:
            if col not in df.columns:
                df[col] = 0
        
        X = df[numeric_features + categorical_features]
        y = df['accepted']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        preprocessor.set_output(transform="pandas")
        X_processed = preprocessor.fit_transform(X)
        
        # Save Preprocessor
        joblib.dump(preprocessor, os.path.join(self.artifacts_dir, 'approval_preprocessor.pkl'))
        
        # Combine
        # X_processed preserves index (if set_output="pandas"), so we must match y's index
        df_final = pd.concat([X_processed, y], axis=1)
        
        # Save
        output_path = os.path.join(self.processed_dir, "train_data_approval.csv")
        df_final.to_csv(output_path, index=False)
        print(f"Saved Approval Training Data to {output_path} (Shape: {df_final.shape})")

if __name__ == "__main__":
    fe = FeatureEngineering()
    fe.run()
