import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, classification_report, roc_curve, accuracy_score
import shap
import matplotlib.pyplot as plt
import joblib
import os
import re

class ModelTrainer:
    def __init__(self, processed_dir="data/processed", models_dir="models", reports_dir="reports/figures"):
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
    def sanitize_columns(self, df):
        # XGBoost doesn't like brackets or < >
        df.columns = [re.sub(r'[\[\]<>]', '', col) for col in df.columns]
        return df
        
    def run(self):
        print("Starting Model Training...")
        
        # 1. Train Approval Model
        self.train_approval_workflow()
        
        # 2. Train Default & Interest Models
        self.train_risk_pricing_workflow()
        
        print("Model Training Completed.")

    def train_approval_workflow(self):
        print("\n=== Approval Model Workflow ===")
        path = os.path.join(self.processed_dir, "train_data_approval.csv")
        if not os.path.exists(path):
            print(f"Skipping Approval Model: {path} not found.")
            return
            
        df = pd.read_csv(path)
        df = self.sanitize_columns(df)
        
        print(f"Loaded Approval Data: {df.shape}")
        
        target = 'accepted'
        if target not in df.columns:
            print("Target 'accepted' not found in approval data.")
            return
            
        feature_cols = [c for c in df.columns if c.startswith('num__') or c.startswith('cat__')]
        X = df[feature_cols]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training XGBoost Classifier for Approval...")
        
        # Calculate scale_pos_weight for imbalance
        # sum(y_train) is count of 1s (Accepted)
        # len(y_train) - sum(y_train) is count of 0s (Rejected)
        # weight = count(0) / count(1)
        num_pos = sum(y_train)
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        print(f"Class Balance - Accepted: {num_pos}, Rejected: {num_neg}, Scale Weight: {scale_pos_weight:.2f}")

        # Use XGBoost for better performance
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"Approval Model - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        # Save
        joblib.dump(model, os.path.join(self.models_dir, 'approval_model.pkl'))
        
        # Feature Importance
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, max_num_features=10)
        plt.title("Approval Model Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, 'approval_importance.png'))
        plt.close()

    def train_risk_pricing_workflow(self):
        print("\n=== Risk & Pricing Models Workflow ===")
        path = os.path.join(self.processed_dir, "train_data_full.csv")
        if not os.path.exists(path):
            print(f"Skipping Risk/Pricing Models: {path} not found.")
            return
            
        df = pd.read_csv(path)
        df = self.sanitize_columns(df)
        
        print(f"Loaded Full Data: {df.shape}")
        
        # Identify Targets
        target_default = 'default_target'
        target_interest = 'int_rate_target' # Use the preserved target
        
        # Use only engineered features (prefixed with num__ or cat__)
        # This avoids including raw object columns (like emp_title, url) that cause XGBoost errors
        feature_cols = [c for c in df.columns if c.startswith('num__') or c.startswith('cat__')]
        
        print(f"Selected {len(feature_cols)} features for training.")
        
        X = df[feature_cols]
        
        # --- Default Model ---
        # Filter rows where default_target is known (0 or 1)
        # Use int_rate as feature
        mask_default = df[target_default].notna()
        X_def = X[mask_default]
        y_def = df.loc[mask_default, target_default]
        
        if len(X_def) > 0:
            print(f"\nTraining Default Model on {len(X_def)} samples...")
            X_train, X_test, y_train, y_test = train_test_split(X_def, y_def, test_size=0.2, random_state=42)
            
            # Use XGBoost
            model_def = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), # Handle imbalance
                random_state=42
            )
            
            model_def.fit(X_train, y_train)
            
            # Evaluate
            y_prob = model_def.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            print(f"Default Model ROC-AUC: {auc:.4f}")
            
            joblib.dump(model_def, os.path.join(self.models_dir, 'default_model.pkl'))
            
            # SHAP
            self.interpret_model(model_def, X_test.iloc[:100], "default_model")
            
        else:
            print("No valid labels for Default Model.")

        # --- Interest Rate Model ---
        # Train on all data where int_rate_target exists
        if target_interest not in df.columns:
             print(f"Skipping Interest Rate Model: {target_interest} not found.")
             return

        mask_int = df[target_interest].notna()
        # Drop num__int_rate from features to avoid leakage
        X_int = X[mask_int].drop(columns=['num__int_rate'], errors='ignore')
        y_int = df.loc[mask_int, target_interest]
        
        if len(X_int) > 0:
            print(f"\nTraining Interest Rate Model on {len(X_int)} samples...")
            X_train, X_test, y_train, y_test = train_test_split(X_int, y_int, test_size=0.2, random_state=42)
            
            # XGBRegressor
            model_int = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
            
            model_int.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model_int.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f"Interest Rate Model - RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            joblib.dump(model_int, os.path.join(self.models_dir, 'interest_model.pkl'))
            
            # SHAP
            self.interpret_model(model_int, X_test.iloc[:100], "interest_model")

    def interpret_model(self, model, X_sample, model_name):
        print(f"Generating SHAP plots for {model_name}...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.savefig(os.path.join(self.reports_dir, f'{model_name}_shap_summary.png'), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"SHAP interpretation failed: {e}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
