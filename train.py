"""
train.py - House Price Prediction Model Training Script

This script loads the California Housing dataset, performs feature engineering
(geographical proximity and structural ratios), trains a HistGradientBoosting
model using a scikit-learn pipeline (with StandardScaler), evaluates it,
and saves the trained pipeline to disk.

Usage:
    python train.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Coordinates for California's two main economic centers
SF_COORD = (37.7749, -122.4194)
LA_COORD = (34.0522, -118.2437)


def engineer_features(df):
    """
    Engineer advanced features from raw California Housing inputs.
    Supports both full DataFrames (training) and single-row DataFrames (inference).
    """
    new_df = df.copy()

    # 1. Proximity to major urban/economic hubs (approx. Euclidean distance)
    new_df["dist_to_SF"] = np.sqrt(
        (new_df["Latitude"] - SF_COORD[0]) ** 2
        + (new_df["Longitude"] - SF_COORD[1]) ** 2
    )
    new_df["dist_to_LA"] = np.sqrt(
        (new_df["Latitude"] - LA_COORD[0]) ** 2
        + (new_df["Longitude"] - LA_COORD[1]) ** 2
    )

    # 2. Structural/demographic ratios
    new_df["bedrms_per_room"] = new_df["AveBedrms"] / new_df["AveRooms"]
    new_df["rooms_per_occup"] = new_df["AveRooms"] / new_df["AveOccup"]

    return new_df


def load_data():
    """Load California Housing dataset, apply feature engineering, and return X and y."""
    housing = fetch_california_housing(as_frame=True)
    X_raw = housing.data
    y = housing.target

    # Apply feature engineering
    X = engineer_features(X_raw)

    print(f"[OK] Raw dataset loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
    print(f"[OK] Feature engineering complete: {X.shape[1]} features total")
    print(f"   Features: {list(X.columns)}")
    return X, y


def train_model(X_train, y_train):
    """
    Build and train a scikit-learn pipeline consisting of:
      1. StandardScaler  – normalises engineered features
      2. HistGradientBoostingRegressor – trains a gradient boosted decision tree model
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", HistGradientBoostingRegressor(random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    print("[OK] Model trained successfully")
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Compute and print key regression metrics on the test set."""
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Model Evaluation Metrics (Test Set) ---")
    print("=" * 42)
    print(f"   R2 Score            : {r2:.4f}")
    print(f"   Mean Absolute Error : {mae:.4f}  (in $100k units)")
    print(f"                       ~ ${mae * 100_000:,.0f}")
    print("=" * 42)
    return r2, mae


def save_model(pipeline, output_dir="model", filename="model.pkl"):
    """Persist the trained pipeline to disk using joblib."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    joblib.dump(pipeline, filepath)
    print(f"\n[SAVE] Model saved to: {filepath}")


def main():
    print("House Price Prediction -- Model Training")
    print("-" * 46)

    # 1. Load data & engineer features
    X, y = load_data()

    # 2. Train / test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

    # 3. Train the model
    pipeline = train_model(X_train, y_train)

    # 4. Evaluate the model
    evaluate_model(pipeline, X_test, y_test)

    # 5. Save the model
    save_model(pipeline)

    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()

