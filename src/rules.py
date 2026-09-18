import pandas as pd


def check_high_amount(transaction):
    """Flag unusually large transactions."""
    if transaction["amount"] > 250:
        return 2, "High transaction amount"
    return 0, None


def check_high_velocity(transaction):
    """Flag customers making many transactions in 24 hours."""
    if transaction["transactions_last_24h"] > 10:
        return 3, "High transaction velocity"
    return 0, None


def check_unusual_hour(transaction):
    """Flag transactions occurring during unusual hours."""
    if 1 <= transaction["transaction_hour"] <= 5:
        return 1, "Unusual transaction hour"
    return 0, None


def check_rapid_transaction(transaction):
    """Flag transactions occurring very shortly after another transaction."""
    if transaction["previous_transaction_minutes"] < 5:
        return 2, "Rapid transaction"
    return 0, None


def check_unusual_country(transaction):
    """Flag transactions from selected higher-risk countries."""
    if transaction["country"] in ["NG", "AE"]:
        return 2, "Unusual country"
    return 0, None


def evaluate_transaction(transaction):
    """
    Apply all fraud detection rules to a transaction.

    Returns:
        risk_score: Total number of risk points.
        risk_level: LOW, MEDIUM, or HIGH.
        reasons: List explaining why the transaction was flagged.
    """

    rules = [
        check_high_amount,
        check_high_velocity,
        check_unusual_hour,
        check_rapid_transaction,
        check_unusual_country,
    ]

    risk_score = 0
    reasons = []

    for rule in rules:
        points, reason = rule(transaction)

        risk_score += points

        if reason:
            reasons.append(reason)

    if risk_score >= 6:
        risk_level = "HIGH"
    elif risk_score >= 3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_score, risk_level, reasons


def apply_rules(df):
    """Apply the rule engine to every transaction."""

    results = []

    for _, transaction in df.iterrows():
        score, level, reasons = evaluate_transaction(transaction)

        results.append({
            "rule_risk_score": score,
            "rule_risk_level": level,
            "rule_reasons": "; ".join(reasons)
        })

    results_df = pd.DataFrame(results, index=df.index)

    return pd.concat([df, results_df], axis=1)


if __name__ == "__main__":


    # Load the synthetic transaction dataset
    input_file = "data/raw/transactions.csv"

    transactions = pd.read_csv(input_file)

    # Apply fraud rules
    transactions_with_rules = apply_rules(transactions)

    # Save results
    output_file = "data/processed/rule_results.csv"

    transactions_with_rules.to_csv(
        output_file,
        index=False
    )

    print("=" * 60)
    print("RULE-BASED FRAUD DETECTION")
    print("=" * 60)

    print(
        f"Total transactions: "
        f"{len(transactions_with_rules):,}"
    )

    print("\nRisk-level distribution:")

    print(
        transactions_with_rules[
            "rule_risk_level"
        ].value_counts()
    )

    print("\nExample flagged transactions:")

    flagged = transactions_with_rules[
        transactions_with_rules["rule_risk_score"] > 0
        ]

    print(
        flagged[
            [
                "transaction_id",
                "amount",
                "country",
                "transaction_hour",
                "transactions_last_24h",
                "rule_risk_score",
                "rule_risk_level",
                "rule_reasons",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
     main()
