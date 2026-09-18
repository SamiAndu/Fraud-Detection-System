import os
import numpy as np
import pandas as pd


# Reproducibility
np.random.seed(42)

# Number of transactions
N_TRANSACTIONS = 10_000

# Possible values
merchant_categories = [
    "grocery",
    "restaurant",
    "electronics",
    "travel",
    "online",
    "fuel",
    "entertainment",
    "retail",
]

countries = [
    "US",
    "CA",
    "GB",
    "DE",
    "FR",
    "ET",
    "AE",
    "NG",
    "IN",
]

device_types = [
    "mobile",
    "desktop",
    "tablet",
]


def generate_transactions(n):
    """Generate a synthetic transaction dataset."""

    transaction_ids = [
        f"TXN{i:06d}" for i in range(1, n + 1)
    ]

    customer_ids = [
        f"CUST{np.random.randint(1, 1501):05d}"
        for _ in range(n)
    ]

    amounts = np.round(
        np.random.lognormal(mean=3.5, sigma=1.0, size=n),
        2
    )

    merchant_category = np.random.choice(
        merchant_categories,
        size=n,
        p=[0.18, 0.15, 0.10, 0.08, 0.15, 0.12, 0.07, 0.15]
    )

    country = np.random.choice(
        countries,
        size=n,
        p=[0.55, 0.08, 0.06, 0.05, 0.04, 0.04, 0.05, 0.05, 0.08]
    )

    transaction_hour = np.random.randint(0, 24, size=n)

    device_type = np.random.choice(
        device_types,
        size=n,
        p=[0.55, 0.35, 0.10]
    )

    previous_transaction_minutes = np.round(
        np.random.exponential(scale=180, size=n),
        2
    )

    transactions_last_24h = np.random.poisson(
        lam=3,
        size=n
    )

    customer_age = np.random.randint(
        18,
        76,
        size=n
    )

    # Start with mostly legitimate transactions
    is_fraud = np.zeros(n, dtype=int)

    # ---------------------------------------
    # Synthetic fraud patterns
    # ---------------------------------------

    # Pattern 1: unusually large transaction
    large_transaction = amounts > 250

    # Pattern 2: many transactions within 24 hours
    high_velocity = transactions_last_24h > 10

    # Pattern 3: unusual late-night transaction
    unusual_hour = (
            (transaction_hour >= 1)
            & (transaction_hour <= 5)
    )

    # Pattern 4: selected higher-risk countries
    unusual_country = np.isin(
        country,
        ["NG", "AE"]
    )

    # Pattern 5: very short time between transactions
    rapid_transaction = (
            previous_transaction_minutes < 5
    )

    # Combine patterns
    fraud_score = (
            large_transaction.astype(int)
            + high_velocity.astype(int)
            + unusual_hour.astype(int)
            + unusual_country.astype(int)
            + rapid_transaction.astype(int)
    )

    # Transactions with multiple suspicious characteristics
    is_fraud[fraud_score >= 2] = 1

    # Add a small amount of random fraud to make the dataset less deterministic
    random_fraud = np.random.random(n) < 0.01
    is_fraud[random_fraud] = 1

    # Create DataFrame
    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "customer_id": customer_ids,
        "amount": amounts,
        "merchant_category": merchant_category,
        "country": country,
        "transaction_hour": transaction_hour,
        "device_type": device_type,
        "previous_transaction_minutes": previous_transaction_minutes,
        "transactions_last_24h": transactions_last_24h,
        "customer_age": customer_age,
        "is_fraud": is_fraud,
    })

    return df


def main():
    """Generate and save the dataset."""

    df = generate_transactions(N_TRANSACTIONS)

    # Create output directory if needed
    output_directory = "data/raw"
    os.makedirs(output_directory, exist_ok=True)

    output_file = os.path.join(
        output_directory,
        "transactions.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("=" * 50)
    print("Synthetic Fraud Detection Dataset")
    print("=" * 50)

    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum():,}")
    print(
        f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}%"
    )

    print("\nDataset preview:")
    print(df.head())

    print("\nSaved to:")
    print(output_file)


if __name__ == "__main__":
    main()