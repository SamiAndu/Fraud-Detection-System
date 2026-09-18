import pandas as pd
import numpy as np


def create_features(df):
    """
    Create machine-learning features from transaction data.
    """

    features = df.copy()

    # -----------------------------
    # Transaction amount features
    # -----------------------------

    # Log transformation reduces the effect of very large amounts.
    features["log_amount"] = np.log1p(features["amount"])

    # Flag unusually large transactions.
    features["is_large_amount"] = (
            features["amount"] > 250
    ).astype(int)

    # -----------------------------
    # Time-based features
    # -----------------------------

    # Flag transactions between 1 AM and 5 AM.
    features["is_night"] = (
            (features["transaction_hour"] >= 1)
            & (features["transaction_hour"] <= 5)
    ).astype(int)

    # Convert hour into cyclical features.
    # This helps ML models understand that hour 23 and hour 0
    # are close to each other.
    features["hour_sin"] = np.sin(
        2 * np.pi * features["transaction_hour"] / 24
    )

    features["hour_cos"] = np.cos(
        2 * np.pi * features["transaction_hour"] / 24
    )

    # -----------------------------
    # Transaction velocity features
    # -----------------------------

    features["is_high_velocity"] = (
            features["transactions_last_24h"] > 10
    ).astype(int)

    features["is_rapid_transaction"] = (
            features["previous_transaction_minutes"] < 5
    ).astype(int)

    # -----------------------------
    # Country feature
    # -----------------------------

    features["is_unusual_country"] = (
        features["country"].isin(["NG", "AE"])
    ).astype(int)

    # -----------------------------
    # Risk interaction feature
    # -----------------------------

    features["behavior_risk_count"] = (
            features["is_large_amount"]
            + features["is_night"]
            + features["is_high_velocity"]
            + features["is_rapid_transaction"]
            + features["is_unusual_country"]
    )

    return features


def prepare_ml_data(df):
    """
    Convert transaction data into ML-ready numerical features.
    """

    features = create_features(df)

    # Target variable
    target = features["is_fraud"]

    # Remove identifiers and target from ML features.
    columns_to_drop = [
        "transaction_id",
        "customer_id",
        "is_fraud",
    ]

    X = features.drop(
        columns=columns_to_drop
    )

    # Convert categorical variables into numerical columns.
    X = pd.get_dummies(
        X,
        columns=[
            "merchant_category",
            "country",
            "device_type",
        ],
        dtype=int
    )

    return X, target, features


if __name__ == "__main__":

    input_file = "data/raw/transactions.csv"
    transactions = pd.read_csv(input_file)

    X, y, feature_data = prepare_ml_data(
        transactions
    )

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print(f"Original rows: {len(transactions):,}")
    print(f"ML feature count: {X.shape[1]}")
    print(f"Target count: {len(y):,}")

    print("\nFeature columns:")
    for column in X.columns:
        print(f" - {column}")

    print("\nFeature preview:")
    print(X.head())

    print("\nTarget distribution:")
    print(y.value_counts())

    # Save ML-ready features
    output_file = "data/processed/ml_features.csv"

    ml_data = X.copy()
    ml_data["is_fraud"] = y

    ml_data.to_csv(
        output_file,
        index=False
    )

    print("\nML-ready dataset saved to:")
    print(output_file)