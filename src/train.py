import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def train_model(
    processed_data_path: str = "data/processed/olist_processed.csv",
    model_output_dir: str = "models",
):
    print("1. Loading processed feature matrix...")
    df = pd.read_csv(processed_data_path)

    X = df.drop(columns=["is_delayed"])
    y = df["is_delayed"]

    # 2. Stratified Train/Test Split (Preserves delayed class ratio in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(
        f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples"
    )

    # 3. Model Initialization (HistGradientBoostingClassifier handling class imbalance)
    print("\n2. Training Gradient Boosting Classifier...")

    # Calculate class weight ratio for imbalance
    neg_count, pos_count = np.bincount(y_train)
    class_weight_ratio = neg_count / pos_count

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced",  # Adjusts loss function for imbalanced delayed class
        random_state=42,
    )

    model.fit(X_train, y_train)

    # 4. Model Evaluation on Unseen Test Data
    print("\n3. Evaluating Model Performance...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print("-" * 50)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score:  {pr_auc:.4f}")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))

    # 5. Serialize Artifact for API Deployment
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "delay_model.joblib")

    # Save model object along with feature column metadata
    artifact = {
        "model": model,
        "feature_names": list(X.columns),
    }

    joblib.dump(artifact, model_path)
    print(f"Model artifact successfully saved to '{model_path}'!")


if __name__ == "__main__":
    train_model()