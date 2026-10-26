import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class TabularDataProcessor:
    def __init__(self, target_col, scaling="minmax"):
        self.target_col = target_col
        self.scaling = scaling
        self.scaler = MinMaxScaler(feature_range=(-1, 1)) if scaling == "minmax" else StandardScaler()
        self.numerical_cols = []
        self.categorical_cols = []
        self.feature_cols = []
        self.category_mappings = {}
        self.numeric_bounds = {}
        self.imbalance_info = {}

    def analyze_distribution(self, df):
        """Analyzes class distribution, calculates imbalance ratio and status."""
        target_counts = df[self.target_col].value_counts().sort_values(ascending=False)
        total_samples = len(df)
        classes = target_counts.index.tolist()
        counts = target_counts.values.tolist()
        percentages = (target_counts / total_samples * 100).round(2).tolist()

        maj_class = classes[0]
        min_class = classes[-1]
        maj_count = counts[0]
        min_count = counts[-1]

        imbalance_ratio = round(maj_count / max(min_count, 1), 2)
        min_pct = percentages[-1]

        # Classification of balance state
        # IR <= 1.25 is balanced (e.g. 55%-45% or 50%-50%)
        # 1.25 < IR <= 2.5 is moderate imbalance
        # IR > 2.5 is severe imbalance
        if imbalance_ratio <= 1.25:
            balance_status = "BALANCED"
            status_desc = f"The dataset is well balanced (Imbalance Ratio: {imbalance_ratio}:1, Minority: {min_pct}%)."
        elif imbalance_ratio <= 2.5:
            balance_status = "MODERATELY_IMBALANCED"
            status_desc = f"Moderate class imbalance detected (Imbalance Ratio: {imbalance_ratio}:1, Minority: {min_pct}%)."
        else:
            balance_status = "HIGHLY_IMBALANCED"
            status_desc = f"Severe class imbalance detected (Imbalance Ratio: {imbalance_ratio}:1, Minority: {min_pct}%)."

        self.imbalance_info = {
            "classes": classes,
            "counts": counts,
            "percentages": percentages,
            "maj_class": maj_class,
            "min_class": min_class,
            "maj_count": maj_count,
            "min_count": min_count,
            "imbalance_ratio": imbalance_ratio,
            "status": balance_status,
            "status_desc": status_desc,
            "total_samples": total_samples,
            "num_needed_for_50_50": max(0, maj_count - min_count)
        }
        return self.imbalance_info

    def fit_transform(self, df, is_training=True):
        """Prepares and normalizes features for GAN training."""
        features = df.drop(columns=[self.target_col]).copy()
        self.feature_cols = features.columns.tolist()

        self.numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [c for c in self.feature_cols if c not in self.numerical_cols]

        processed_features = pd.DataFrame(index=features.index)

        # Numerical handling
        for col in self.numerical_cols:
            col_data = features[col].fillna(features[col].median())
            self.numeric_bounds[col] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "is_int": np.issubdtype(features[col].dtype, np.integer)
            }
            processed_features[col] = col_data

        if self.numerical_cols:
            if is_training:
                scaled_num = self.scaler.fit_transform(processed_features[self.numerical_cols])
            else:
                scaled_num = self.scaler.transform(processed_features[self.numerical_cols])
            for idx, col in enumerate(self.numerical_cols):
                processed_features[col] = scaled_num[:, idx]

        # Categorical handling (simple label encoding for tabular GAN)
        for col in self.categorical_cols:
            if is_training:
                unique_cats = features[col].astype(str).unique().tolist()
                cat_to_id = {cat: i for i, cat in enumerate(unique_cats)}
                self.category_mappings[col] = {
                    "cat_to_id": cat_to_id,
                    "id_to_cat": {i: cat for cat, i in cat_to_id.items()},
                    "n_cats": len(unique_cats)
                }
            cat_map = self.category_mappings[col]["cat_to_id"]
            # Map known categories, default to 0 for unknown
            encoded = features[col].astype(str).map(lambda x: cat_map.get(x, 0)).astype(float)
            # scale categorical codes to [-1, 1]
            n_cats = max(1, self.category_mappings[col]["n_cats"] - 1)
            processed_features[col] = (encoded / n_cats) * 2.0 - 1.0

        return processed_features.values.astype(np.float32)

    def inverse_transform(self, synthetic_matrix):
        """Converts synthetic continuous GAN outputs back to valid dataframe matching original column types."""
        df_recon = pd.DataFrame(synthetic_matrix, columns=self.feature_cols)

        # Inverse transform numerical
        if self.numerical_cols:
            unscaled_num = self.scaler.inverse_transform(df_recon[self.numerical_cols])
            for idx, col in enumerate(self.numerical_cols):
                bounds = self.numeric_bounds[col]
                # clip to observed boundaries to ensure domain validity
                clipped = np.clip(unscaled_num[:, idx], bounds["min"], bounds["max"])
                if bounds["is_int"]:
                    clipped = np.round(clipped).astype(int)
                df_recon[col] = clipped

        # Inverse transform categorical
        for col in self.categorical_cols:
            cat_info = self.category_mappings[col]
            n_cats = max(1, cat_info["n_cats"] - 1)
            # denormalize from [-1, 1] to category index
            norm_vals = np.clip((df_recon[col].values + 1.0) / 2.0 * n_cats, 0, n_cats)
            cat_ids = np.round(norm_vals).astype(int)
            id_to_cat = cat_info["id_to_cat"]
            df_recon[col] = [id_to_cat.get(cid, id_to_cat[0]) for cid in cat_ids]

        return df_recon
