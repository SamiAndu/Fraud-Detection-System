from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
FEATURE_DATA = PROJECT_ROOT / "data" / "processed" / "ml_features.csv"
RISK_RESULTS = PROJECT_ROOT / "data" / "processed" / "final_risk_results.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "fraud_model.joblib"


def test_raw_dataset_exists():
    """Verify that the synthetic transaction dataset exists."""
    assert RAW_DATA.exists()


def test_raw_dataset_has_expected_size():
    """Verify that the generated dataset contains 10,000 transactions."""
    df = pd.read_csv(RAW_DATA)

    assert len(df) == 10_000
    assert "transaction_id" in df.columns
    assert "amount" in df.columns
    assert "country" in df.columns


def test_feature_dataset_exists():
    """Verify that the ML feature dataset was generated."""
    assert FEATURE_DATA.exists()

    df = pd.read_csv(FEATURE_DATA)

    assert len(df) == 10_000
    assert "is_fraud" in df.columns


def test_risk_results_exist():
    """Verify that the final risk-scoring pipeline produced results."""
    assert RISK_RESULTS.exists()

    df = pd.read_csv(RISK_RESULTS)

    assert len(df) == 10_000

    required_columns = [
        "transaction_id",
        "rule_risk_score",
        "ml_fraud_probability",
        "hybrid_risk_score",
        "final_risk_level",
        "rule_reasons",
    ]

    for column in required_columns:
        assert column in df.columns


def test_risk_levels_are_valid():
    """Verify that only expected risk levels are produced."""
    df = pd.read_csv(RISK_RESULTS)

    valid_levels = {"LOW", "MEDIUM", "HIGH"}

    assert set(df["final_risk_level"].unique()).issubset(valid_levels)


def test_model_exists():
    """Verify that the trained machine-learning model was saved."""
    assert MODEL_FILE.exists()