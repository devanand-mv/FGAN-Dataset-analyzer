import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class ModelEvaluator:
    @staticmethod
    def evaluate_quality(real_minority_df, synthetic_minority_df, numeric_cols):
        """
        Calculates data validity and diversity metrics between real and synthetic minority samples.
        """
        metrics = {}
        
        # 1. Duplicate check (Exact matches)
        total_synthetic = len(synthetic_minority_df)
        merged = pd.merge(real_minority_df, synthetic_minority_df, how='inner')
        exact_duplicates = len(merged)
        duplicate_rate = (exact_duplicates / max(total_synthetic, 1)) * 100
        
        # Synthetic self-duplicate rate
        synthetic_unique = len(synthetic_minority_df.drop_duplicates())
        self_duplicate_rate = ((total_synthetic - synthetic_unique) / max(total_synthetic, 1)) * 100
        
        metrics["total_synthetic"] = total_synthetic
        metrics["exact_overlap_with_real"] = exact_duplicates
        metrics["overlap_percentage"] = round(duplicate_rate, 2)
        metrics["synthetic_diversity_unique_pct"] = round((synthetic_unique / max(total_synthetic, 1)) * 100, 2)
        metrics["null_count"] = int(synthetic_minority_df.isnull().sum().sum())
        
        # 2. Statistical Moments Comparison (Mean and Std comparison across numeric features)
        feature_stats = []
        for col in numeric_cols:
            if col in real_minority_df.columns and col in synthetic_minority_df.columns:
                real_mean = float(real_minority_df[col].mean())
                real_std = float(real_minority_df[col].std())
                syn_mean = float(synthetic_minority_df[col].mean())
                syn_std = float(synthetic_minority_df[col].std())
                
                # Percentage difference in mean
                mean_diff_pct = abs(real_mean - syn_mean) / (abs(real_mean) + 1e-6) * 100
                feature_stats.append({
                    "Feature": col,
                    "Real Mean": round(real_mean, 2),
                    "Syn Mean": round(syn_mean, 2),
                    "Real Std": round(real_std, 2),
                    "Syn Std": round(syn_std, 2),
                    "Mean Δ%": round(mean_diff_pct, 1)
                })
        metrics["feature_stats"] = pd.DataFrame(feature_stats)
        return metrics

    @staticmethod
    def train_and_compare_classifiers(X_train_orig, y_train_orig, X_train_bal, y_train_bal, X_test, y_test, model_type="RandomForest", minority_class=1):
        """
        Trains the exact same classifier on:
        1. Original Imbalanced Data
        2. GAN-Augmented Balanced Data
        Evaluates both strictly on the UNTOUCHED real test set.
        """
        if model_type == "RandomForest":
            clf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_bal = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf_orig = LogisticRegression(max_iter=1000, random_state=42)
            clf_bal = LogisticRegression(max_iter=1000, random_state=42)
            
        # Fit both
        clf_orig.fit(X_train_orig, y_train_orig)
        clf_bal.fit(X_train_bal, y_train_bal)
        
        # Predict on untouched test set
        pred_orig = clf_orig.predict(X_test)
        pred_bal = clf_bal.predict(X_test)
        
        # Probabilities for AUC if binary
        try:
            prob_orig = clf_orig.predict_proba(X_test)[:, 1]
            prob_bal = clf_bal.predict_proba(X_test)[:, 1]
            auc_orig = roc_auc_score(y_test, prob_orig)
            auc_bal = roc_auc_score(y_test, prob_bal)
        except Exception:
            auc_orig = 0.0
            auc_bal = 0.0

        pos_label = minority_class if minority_class in np.unique(y_test) else 1

        res_orig = {
            "Accuracy": round(accuracy_score(y_test, pred_orig) * 100, 2),
            "Minority Precision": round(precision_score(y_test, pred_orig, pos_label=pos_label, zero_division=0) * 100, 2),
            "Minority Recall": round(recall_score(y_test, pred_orig, pos_label=pos_label, zero_division=0) * 100, 2),
            "Minority F1-Score": round(f1_score(y_test, pred_orig, pos_label=pos_label, zero_division=0) * 100, 2),
            "ROC-AUC": round(auc_orig, 4),
            "Confusion Matrix": confusion_matrix(y_test, pred_orig).tolist()
        }
        
        res_bal = {
            "Accuracy": round(accuracy_score(y_test, pred_bal) * 100, 2),
            "Minority Precision": round(precision_score(y_test, pred_bal, pos_label=pos_label, zero_division=0) * 100, 2),
            "Minority Recall": round(recall_score(y_test, pred_bal, pos_label=pos_label, zero_division=0) * 100, 2),
            "Minority F1-Score": round(f1_score(y_test, pred_bal, pos_label=pos_label, zero_division=0) * 100, 2),
            "ROC-AUC": round(auc_bal, 4),
            "Confusion Matrix": confusion_matrix(y_test, pred_bal).tolist()
        }
        
        comparison_df = pd.DataFrame([
            {"Metric": "Minority Recall (%)", "Original Imbalanced": res_orig["Minority Recall"], "GAN Balanced": res_bal["Minority Recall"], "Change": round(res_bal["Minority Recall"] - res_orig["Minority Recall"], 2)},
            {"Metric": "Minority F1-Score (%)", "Original Imbalanced": res_orig["Minority F1-Score"], "GAN Balanced": res_bal["Minority F1-Score"], "Change": round(res_bal["Minority F1-Score"] - res_orig["Minority F1-Score"], 2)},
            {"Metric": "Minority Precision (%)", "Original Imbalanced": res_orig["Minority Precision"], "GAN Balanced": res_bal["Minority Precision"], "Change": round(res_bal["Minority Precision"] - res_orig["Minority Precision"], 2)},
            {"Metric": "Overall Accuracy (%)", "Original Imbalanced": res_orig["Accuracy"], "GAN Balanced": res_bal["Accuracy"], "Change": round(res_bal["Accuracy"] - res_orig["Accuracy"], 2)},
            {"Metric": "ROC-AUC Score", "Original Imbalanced": res_orig["ROC-AUC"], "GAN Balanced": res_bal["ROC-AUC"], "Change": round(res_bal["ROC-AUC"] - res_orig["ROC-AUC"], 4)},
        ])
        
        return {
            "original": res_orig,
            "balanced": res_bal,
            "comparison_df": comparison_df
        }
