import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def load_data():
    """Load the ML-ready transaction dataset."""

    file_path = "data/processed/ml_features.csv"

    df = pd.read_csv(file_path)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    return X, y


def train_model(X_train, y_train):
    """Train the Random Forest fraud classifier."""

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraud"],
            digits=4,
        )
    )

    print("Confusion Matrix:")

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print(matrix)

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"\nROC-AUC: {roc_auc:.4f}")

    return predictions, probabilities


def main():

    print("=" * 60)
    print("FRAUD DETECTION - MACHINE LEARNING MODEL")
    print("=" * 60)

    # Load data
    X, y = load_data()

    print(f"\nTotal transactions: {len(X):,}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Fraudulent transactions: {y.sum():,}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining transactions: {len(X_train):,}")
    print(f"Testing transactions: {len(X_test):,}")

    # Train model
    print("\nTraining Random Forest...")

    model = train_model(
        X_train,
        y_train
    )

    print("Training complete.")
    # Save trained model
    model_path = "models/fraud_model.joblib"

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved to: {model_path}")

    # Evaluate
    evaluate_model(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main()