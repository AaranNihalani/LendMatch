import pandas as pd
import numpy as np
import joblib
import os
try:
    from src.lender_matching import LenderMatcher
except ModuleNotFoundError:
    from lender_matching import LenderMatcher

class PredictionService:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.artifacts_dir = os.path.join(models_dir, "artifacts")
        self.lender_matcher = LenderMatcher()
        self._last_approval_prep_error = None
        self._last_risk_prep_error = None
        
        # Load Models and Artifacts
        print("Loading models and artifacts...")
        try:
            self.approval_model = joblib.load(os.path.join(models_dir, "approval_model.pkl"))
            self.default_model = joblib.load(os.path.join(models_dir, "default_model.pkl"))
            self.interest_model = joblib.load(os.path.join(models_dir, "interest_model.pkl"))
            
            # Check for specific preprocessors, fallback to generic if not found
            if os.path.exists(os.path.join(self.artifacts_dir, "approval_preprocessor.pkl")):
                self.approval_preprocessor = joblib.load(os.path.join(self.artifacts_dir, "approval_preprocessor.pkl"))
            else:
                self.approval_preprocessor = None
                print("Warning: approval_preprocessor.pkl not found.")

            if os.path.exists(os.path.join(self.artifacts_dir, "full_preprocessor.pkl")):
                self.full_preprocessor = joblib.load(os.path.join(self.artifacts_dir, "full_preprocessor.pkl"))
            else:
                self.full_preprocessor = None
                print("Warning: full_preprocessor.pkl not found.")
                
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.approval_model = None
            self.default_model = None
            self.interest_model = None

    def _prepare_approval_features(self, input_data):
        """
        Prepare features for the Approval Model.
        Expected input keys: amount, risk_score, dti, emp_length, state
        """
        df = pd.DataFrame([{}])
        
        # Mapping input keys to model feature names
        # Model expects: amount, risk_score, dti, emp_length_num, state
        
        # Ensure emp_length is numeric
        emp_len = input_data.get('emp_length', 0)
        if isinstance(emp_len, str):
            # Parse "< 1 year", "10+ years", etc.
            import re
            nums = re.findall(r'\d+', emp_len)
            if nums:
                df['emp_length_num'] = int(nums[0])
            else:
                df['emp_length_num'] = 0
        else:
             df['emp_length_num'] = emp_len
             
        # Rename/Ensure columns
        df['amount'] = input_data.get('loan_amount', 0)
        df['risk_score'] = input_data.get('fico_score', 0)
        df['dti'] = input_data.get('dti', 0)
        # policy_code removed as it was a leaked feature
        df['state'] = input_data.get('state', 'Unknown')
        
        # Transform
        if self.approval_preprocessor:
            # The preprocessor expects specific columns. 
            # We must ensure only those are passed if the preprocessor is strict, 
            # or rely on ColumnTransformer ignoring others if configured so.
            # But typically ColumnTransformer needs the columns specified in its transformers to be present.
            try:
                if hasattr(self.approval_preprocessor, "feature_names_in_"):
                    required = list(self.approval_preprocessor.feature_names_in_)
                    df_in = df.reindex(columns=required)
                else:
                    df_in = df
                self._last_approval_prep_error = None
                return self.approval_preprocessor.transform(df_in)
            except Exception as e:
                print(f"Approval preprocessing error: {e}")
                self._last_approval_prep_error = str(e)
                return None
        return df

    def _prepare_risk_features(self, input_data):
        """
        Prepare features for Default and Interest Rate Models.
        Replicates logic from FeatureEngineering.process_accepted_features
        """
        df = pd.DataFrame([{}])
        
        # 1. Map basic inputs to dataframe columns expected by logic
        df['loan_amnt'] = float(input_data.get('loan_amount', 0))
        df['annual_inc'] = float(input_data.get('annual_inc', 0))
        df['fico_range_low'] = float(input_data.get('fico_score', 0))
        df['fico_range_high'] = float(input_data.get('fico_score', 0)) # Assuming exact score
        df['dti'] = float(input_data.get('dti', 0))
        df['term_months'] = int(input_data.get('term', 36))
        df['home_ownership'] = input_data.get('home_ownership', 'RENT')
        df['purpose'] = input_data.get('purpose', 'debt_consolidation')
        df['addr_state'] = input_data.get('state', 'CA')
        df['revol_bal'] = float(input_data.get('revol_bal', 0))
        df['total_acc'] = float(input_data.get('total_acc', 10))
        df['emp_length'] = input_data.get('emp_length', '1 year')
        
        # 2. Derived Features
        
        # Emp Length Num
        if isinstance(df['emp_length'].iloc[0], str):
             import re
             nums = re.findall(r'\d+', df['emp_length'].iloc[0])
             df['emp_length_num'] = int(nums[0]) if nums else 0
        else:
             df['emp_length_num'] = df['emp_length']

        # FICO Avg
        df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        
        # Months Since Earliest Credit Line
        # Input might be a date string or year. 
        # For simplicity in demo, we might ask for "years of credit history" directly or parse date.
        # If input has 'earliest_cr_line' as date:
        earliest_line = input_data.get('earliest_cr_line', 'Jan-2000')
        try:
            dt = pd.to_datetime(earliest_line, format='%b-%Y')
        except:
            dt = pd.to_datetime('2000-01-01') # Default fallback
            
        reference_date = pd.to_datetime('2019-01-01')
        df['months_since_earliest_cr_line'] = (reference_date - dt).days / 30.0
        
        # Ratios
        df['annual_inc'] = df['annual_inc'].replace(0, 0.01)
        df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
        # Installment (approximate if not provided)
        # Simple amortization for estimation
        r = 0.10 / 12 # Guess 10% rate
        n = df['term_months'].iloc[0]
        est_installment = df['loan_amnt'] * (r * (1 + r)**n) / ((1 + r)**n - 1)
        df['installment'] = input_data.get('installment', est_installment)
        df['installment_to_income'] = (df['installment'] * 12) / df['annual_inc']
        df['revol_bal_to_income'] = df['revol_bal'] / df['annual_inc']
        
        # Logs
        for col in ['annual_inc', 'revol_bal', 'total_acc']:
            df[f'{col}_log'] = np.log1p(df[col])
            
        # Add missing columns with 0/defaults that the model might expect
        # (The pipeline handles imputation, but column must exist)
        expected_cols = [
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 
            'revol_util', 'collections_12_mths_ex_med', 'acc_now_delinq', 
            'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 
            'mort_acc', 'pub_rec_bankruptcies', 'tax_liens',
            'verification_status', 'initial_list_status', 'application_type',
            'int_rate' # Required by preprocessor
        ]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = 0 if c not in ['verification_status', 'initial_list_status', 'application_type'] else 'Unknown'
                # For int_rate, use input_data if provided (e.g. predicted value), else 0
                if c == 'int_rate':
                    df[c] = input_data.get('int_rate', 0.0)

        # Transform
        if self.full_preprocessor:
            try:
                # ColumnTransformer with set_output="pandas" returns a DataFrame with feature names
                # We need to make sure we catch errors if columns are missing
                if hasattr(self.full_preprocessor, "feature_names_in_"):
                    required = list(self.full_preprocessor.feature_names_in_)
                    df_in = df.reindex(columns=required)
                else:
                    df_in = df
                self._last_risk_prep_error = None
                X_processed = self.full_preprocessor.transform(df_in)
                
                feature_names = None
                if hasattr(self.full_preprocessor, "get_feature_names_out"):
                    try:
                        feature_names = list(self.full_preprocessor.get_feature_names_out())
                    except Exception:
                        feature_names = None

                if isinstance(X_processed, pd.DataFrame):
                    X_df = X_processed
                elif feature_names is not None and getattr(X_processed, "shape", None) is not None and len(feature_names) == X_processed.shape[1]:
                    X_df = pd.DataFrame(X_processed, columns=feature_names)
                else:
                    X_df = pd.DataFrame(X_processed)

                cols_to_keep = [c for c in X_df.columns if str(c).startswith('num__') or str(c).startswith('cat__')]
                return X_df[cols_to_keep]
                
            except Exception as e:
                print(f"Risk preprocessing error: {e}")
                self._last_risk_prep_error = str(e)
                return None
        return df

    def predict(self, input_data):
        """
        Main prediction pipeline.
        
        Args:
            input_data (dict): User input.
            
        Returns:
            dict: Prediction results (Approval, Default Prob, Interest Rate, Offers).
        """
        results = {}
        warnings = []
        
        # 1. Approval Prediction
        X_app = self._prepare_approval_features(input_data)
        if self.approval_model and X_app is not None:
            try:
                # Predict Probability of "Accepted" (1)
                # Ensure input shape matches
                prob_approval = self.approval_model.predict_proba(X_app)[:, 1][0]
                results['approval_probability'] = float(prob_approval)
                results['is_approved'] = bool(prob_approval > 0.5)
            except Exception as e:
                print(f"Approval prediction failed: {e}")
                warnings.append(f"approval_prediction_failed: {e}")
                results['approval_probability'] = 0.0
                results['is_approved'] = False
        else:
            if not self.approval_model:
                warnings.append("approval_model_missing")
            if X_app is None:
                if self._last_approval_prep_error:
                    warnings.append(f"approval_features_missing: {self._last_approval_prep_error}")
                else:
                    warnings.append("approval_features_missing")
            results['approval_probability'] = 0.0
            results['is_approved'] = False
            
        # 2. Risk & Pricing (Only if approved or generally for info)
        # We'll generate it regardless for the "Counterfactual" analysis
        
        default_prob = 0.0
        interest_rate = 15.0 # Default fallback
        
        # Step A: Predict Interest Rate first (using dummy int_rate=0 in preprocessing)
        input_for_rate = input_data.copy()
        input_for_rate['int_rate'] = 0.0 # Dummy
        X_risk_rate = self._prepare_risk_features(input_for_rate)
        
        if self.interest_model and X_risk_rate is not None:
            try:
                # Drop num__int_rate (it exists because preprocessor added it)
                # But X_risk_rate might be None if transform failed
                X_for_rate_pred = X_risk_rate.drop(columns=['num__int_rate'], errors='ignore')
                interest_rate = self.interest_model.predict(X_for_rate_pred)[0]
            except Exception as e:
                print(f"Interest prediction failed: {e}")
                warnings.append(f"interest_prediction_failed: {e}")
        else:
            if not self.interest_model:
                warnings.append("interest_model_missing")
            if X_risk_rate is None:
                if self._last_risk_prep_error:
                    warnings.append(f"risk_features_missing_for_interest: {self._last_risk_prep_error}")
                else:
                    warnings.append("risk_features_missing_for_interest")

        results['predicted_interest_rate'] = float(interest_rate)
                
        # Step B: Predict Default Risk using the predicted interest rate
        input_for_default = input_data.copy()
        input_for_default['int_rate'] = float(interest_rate) # Use predicted rate
        X_risk_default = self._prepare_risk_features(input_for_default)
        
        if self.default_model and X_risk_default is not None:
            try:
                default_prob = self.default_model.predict_proba(X_risk_default)[:, 1][0]
            except Exception as e:
                print(f"Default prediction failed: {e}")
                warnings.append(f"default_prediction_failed: {e}")
        else:
            if not self.default_model:
                warnings.append("default_model_missing")
            if X_risk_default is None:
                if self._last_risk_prep_error:
                    warnings.append(f"risk_features_missing_for_default: {self._last_risk_prep_error}")
                else:
                    warnings.append("risk_features_missing_for_default")

        results['default_probability'] = float(default_prob)

        # 3. Lender Matching
        # Only meaningful if approval probability is decent
        if results.get('approval_probability', 0) > 0.2:
            offers = self.lender_matcher.generate_offers(
                input_data, 
                results.get('predicted_interest_rate', 15.0), 
                results.get('default_probability', 0.1)
            )
            results['offers'] = offers
        else:
            results['offers'] = []

        if warnings:
            results["warnings"] = warnings

        return results
