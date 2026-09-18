import joblib
import pandas as pd

from feature import prepare_ml_data
from rules import apply_rules


MODEL_PATH = "models/fraud_model.joblib"
INPUT_PATH = "data/raw/transactions.csv"
OUTPUT_PATH = "data/processed/final_risk_results.csv"


def load_model():
    """Load the trained machine-learning model."""

    return joblib.load(MODEL_PATH)


def calculate_hybrid_score(rule_score, ml_probability):
    """
    Combine rule-based risk and ML probability.

    Rule score is normalized to a 0-10 scale.
    ML probability is converted to a 0-10 scale.

    The hybrid score gives:
        40% weight to rules
        60% weight to machine learning
    """

    rule_component = min(rule_score, 10)

    ml_component = ml_probability * 10

    hybrid_score = (
            0.40 * rule_component
            + 0.60 * ml_component
    )

    return round(hybrid_score, 2)


def assign_risk_level(score):
    """Convert hybrid score into a risk level."""

    if score >= 6:
        return "HIGH"

    if score >= 3:
        return "MEDIUM"

    return "LOW"


def main():

    print("=" * 60)
    print("HYBRID FRAUD DETECTION ENGINE")
    print("=" * 60)

    # ----------------------------------------
    # Load raw transactions
    # ----------------------------------------

    transactions = pd.read_csv(INPUT_PATH)

    print(
        f"\nTransactions loaded: "
        f"{len(transactions):,}"
    )

    # ----------------------------------------
    # Apply rule engine
    # ----------------------------------------

    rule_results = apply_rules(
        transactions
    )

    # ----------------------------------------
    # Prepare ML features
    # ----------------------------------------

    X, _, _ = prepare_ml_data(
        transactions
    )

    # ----------------------------------------
    # Load trained ML model
    # ----------------------------------------

    model = load_model()

    # Make sure feature order matches training
    if hasattr(model, "feature_names_in_"):

        X = X.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    # ----------------------------------------
    # Generate ML probabilities
    # ----------------------------------------

    ml_probabilities = model.predict_proba(X)[:, 1]

    # ----------------------------------------
    # Build final results
    # ----------------------------------------

    results = rule_results[
        [
            "transaction_id",
            "customer_id",
            "amount",
            "merchant_category",
            "country",
            "transaction_hour",
            "rule_risk_score",
            "rule_risk_level",
            "rule_reasons",
        ]
    ].copy()

    results["ml_fraud_probability"] = (
        ml_probabilities
    )

    results["ml_fraud_probability_pct"] = (
            results["ml_fraud_probability"] * 100
    ).round(2)

    # Calculate hybrid scores
    results["hybrid_risk_score"] = [
        calculate_hybrid_score(
            rule_score,
            ml_probability
        )
        for rule_score, ml_probability
        in zip(
            results["rule_risk_score"],
            results["ml_fraud_probability"]
        )
    ]

    results["final_risk_level"] = (
        results["hybrid_risk_score"]
        .apply(assign_risk_level)
    )

    # ----------------------------------------
    # Save results
    # ----------------------------------------

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ----------------------------------------
    # Display summary
    # ----------------------------------------

    print("\nFinal risk distribution:")

    print(
        results[
            "final_risk_level"
        ].value_counts()
    )

    print("\nAverage ML fraud probability:")

    print(
        f"{results['ml_fraud_probability'].mean() * 100:.2f}%"
    )

    print("\nHigh-risk transactions:")

    high_risk = results[
        results["final_risk_level"] == "HIGH"
        ]

    print(
        high_risk[
            [
                "transaction_id",
                "amount",
                "country",
                "rule_risk_score",
                "ml_fraud_probability_pct",
                "hybrid_risk_score",
                "final_risk_level",
                "rule_reasons",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nResults saved to:")

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()