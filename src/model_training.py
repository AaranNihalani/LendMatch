import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, roc_curve, accuracy_score, confusion_matrix
import joblib
import os
import re

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import shap
except ImportError:
    shap = None

class ModelTrainer:
    def __init__(self, processed_dir="data/processed", models_dir="models", reports_dir="reports/figures"):
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

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
        
    def sanitize_columns(self, df):
        df.columns = [re.sub(r'[\[\]<>]', '', col) for col in df.columns]
        return df
        
    def run(self):
        print("Starting Model Training...")
        self._code_path = os.path.abspath(__file__)
        
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

        model_path = os.path.join(self.models_dir, "approval_model.pkl")
        if not self._is_stale(model_path, [path, getattr(self, "_code_path", os.path.abspath(__file__))]):
            print(f"Skipping Approval Model: {os.path.basename(model_path)} is up-to-date.")
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
        
        print("Training GradientBoostingClassifier for Approval...")

        num_pos = sum(y_train)
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        print(f"Class Balance - Accepted: {num_pos}, Rejected: {num_neg}, Scale Weight: {scale_pos_weight:.2f}")

        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
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
        joblib.dump(model, model_path)
        
        # Feature Importance
        if plt is not None and hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + .5
            plt.figure(figsize=(12, 8))
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            plt.yticks(pos, np.array(X.columns)[sorted_idx])
            plt.title('Approval Model Feature Importance (MDI)')
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(self.reports_dir, 'approval_importance.png'), bbox_inches='tight', dpi=220)
            plt.close()

    def train_risk_pricing_workflow(self):
        print("\n=== Risk & Pricing Models Workflow ===")
        path = os.path.join(self.processed_dir, "train_data_full.csv")
        if not os.path.exists(path):
            print(f"Skipping Risk/Pricing Models: {path} not found.")
            return

        default_model_path = os.path.join(self.models_dir, "default_model.pkl")
        interest_model_path = os.path.join(self.models_dir, "interest_model.pkl")
        code_path = getattr(self, "_code_path", os.path.abspath(__file__))
            
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
        
        if len(X_def) > 0 and self._is_stale(default_model_path, [path, code_path]):
            print(f"\nTraining Default Model on {len(X_def)} samples...")
            X_train, X_test, y_train, y_test = train_test_split(X_def, y_def, test_size=0.2, random_state=42)
            
            model_def = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
            
            model_def.fit(X_train, y_train)
            
            # Evaluate
            y_prob = model_def.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            print(f"Default Model ROC-AUC: {auc:.4f}")
            
            joblib.dump(model_def, default_model_path)

            if plt is not None:
                self.save_roc_plot(y_test, y_prob, "default_model_roc.png", "Default Model ROC Curve")
                self.save_confusion_matrix(y_test, (y_prob >= 0.5).astype(int), "default_model_confusion.png", "Default Model Confusion Matrix")
            
            # SHAP
            self.interpret_model(model_def, X_test.iloc[:500], "default_model")
            
        elif len(X_def) > 0:
            print(f"Skipping Default Model: {os.path.basename(default_model_path)} is up-to-date.")
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
        
        if len(X_int) > 0 and self._is_stale(interest_model_path, [path, code_path]):
            print(f"\nTraining Interest Rate Model on {len(X_int)} samples...")
            X_train, X_test, y_train, y_test = train_test_split(X_int, y_int, test_size=0.2, random_state=42)
            
            model_int = GradientBoostingRegressor(
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
            
            joblib.dump(model_int, interest_model_path)
            
            # SHAP
            self.interpret_model(model_int, X_test.iloc[:500], "interest_model")

            if plt is not None:
                self.save_regression_scatter(y_test, y_pred, "interest_model_pred_vs_actual.png", "Interest Model: Predicted vs Actual")
        elif len(X_int) > 0:
            print(f"Skipping Interest Rate Model: {os.path.basename(interest_model_path)} is up-to-date.")

    def save_roc_plot(self, y_true, y_prob, filename, title):
        if plt is None:
            return
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(7.5, 6))
        plt.plot(fpr, tpr, color="#2b6cb0", linewidth=2, label="ROC")
        plt.plot([0, 1], [0, 1], color="#718096", linestyle="--", linewidth=1, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, filename), bbox_inches="tight", dpi=220)
        plt.close()

    def save_confusion_matrix(self, y_true, y_pred, filename, title):
        if plt is None:
            return
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6.5, 5.5))
        plt.imshow(cm, cmap="Blues")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", color="#1a202c")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, filename), bbox_inches="tight", dpi=220)
        plt.close()

    def save_regression_scatter(self, y_true, y_pred, filename, title):
        if plt is None:
            return
        plt.figure(figsize=(7.5, 6))
        plt.scatter(y_true, y_pred, alpha=0.35, s=18, color="#2f855a")
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="#718096", linewidth=1)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, filename), bbox_inches="tight", dpi=220)
        plt.close()

    def interpret_model(self, model, X_sample, model_name):
        print(f"Generating SHAP plots for {model_name}...")
        if shap is None or plt is None:
            print("Skipping SHAP plots (missing optional dependencies: shap, matplotlib).")
            return
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.savefig(os.path.join(self.reports_dir, f'{model_name}_shap_summary.png'), bbox_inches='tight', dpi=220)
            plt.close()
        except Exception as e:
            print(f"SHAP interpretation failed: {e}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
