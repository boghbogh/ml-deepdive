"""
Synthetic Credit Card Transaction Data Generator

Generates realistic transaction data with fraud patterns for the
Banking ML Demo (Credit Card Fraud Detection).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail",
    "electronics", "travel", "entertainment", "healthcare"
]

CARD_TYPES = ["visa", "mastercard", "amex"]

COUNTRIES = ["US", "US", "US", "US", "US", "US", "US", "US",
             "UK", "CA", "MX", "BR", "DE", "FR", "NG", "CN"]

# Amount ranges by merchant category (mean, std)
CATEGORY_AMOUNTS = {
    "grocery": (75, 40),
    "restaurant": (45, 25),
    "gas_station": (55, 20),
    "online_retail": (120, 80),
    "electronics": (350, 200),
    "travel": (500, 300),
    "entertainment": (60, 35),
    "healthcare": (200, 150),
}

# US geographic bounds (lat, lon)
US_REGIONS = [
    (40.7, -74.0),   # Northeast
    (33.7, -84.4),   # Southeast
    (41.9, -87.6),   # Midwest
    (29.8, -95.4),   # South
    (34.0, -118.2),  # West
    (47.6, -122.3),  # Northwest
]


def generate_customers(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate customer profiles."""
    home_regions = rng.choice(len(US_REGIONS), size=n_customers)
    return pd.DataFrame({
        "CUSTOMER_ID": [f"CUST_{i:05d}" for i in range(n_customers)],
        "CUSTOMER_AGE": rng.integers(22, 75, size=n_customers),
        "CUSTOMER_INCOME": rng.integers(30000, 200000, size=n_customers),
        "HOME_LAT": [US_REGIONS[r][0] + rng.normal(0, 1) for r in home_regions],
        "HOME_LON": [US_REGIONS[r][1] + rng.normal(0, 1) for r in home_regions],
        "CARD_TYPE": rng.choice(CARD_TYPES, size=n_customers),
    })


def generate_merchants(n_merchants: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate merchant profiles."""
    return pd.DataFrame({
        "MERCHANT_ID": [f"MERCH_{i:04d}" for i in range(n_merchants)],
        "MERCHANT_CATEGORY": rng.choice(MERCHANT_CATEGORIES, size=n_merchants),
    })


def generate_normal_transactions(
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n_transactions: int,
    start_date: datetime,
    end_date: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate legitimate transactions."""
    date_range_seconds = int((end_date - start_date).total_seconds())

    cust_idx = rng.choice(len(customers), size=n_transactions)
    merch_idx = rng.choice(len(merchants), size=n_transactions)

    cust_rows = customers.iloc[cust_idx].reset_index(drop=True)
    merch_rows = merchants.iloc[merch_idx].reset_index(drop=True)

    # Generate amounts based on merchant category
    amounts = []
    for cat in merch_rows["MERCHANT_CATEGORY"]:
        mean, std = CATEGORY_AMOUNTS[cat]
        amt = max(1.0, rng.normal(mean, std))
        amounts.append(round(amt, 2))

    # Generate timestamps weighted toward business hours
    hours = rng.choice(24, size=n_transactions, p=_hour_distribution())
    random_seconds = rng.integers(0, date_range_seconds, size=n_transactions)
    timestamps = [
        start_date + timedelta(seconds=int(s))
        for s in random_seconds
    ]
    # Adjust hour
    timestamps = [
        t.replace(hour=int(h), minute=rng.integers(0, 60))
        for t, h in zip(timestamps, hours)
    ]

    return pd.DataFrame({
        "CUSTOMER_ID": cust_rows["CUSTOMER_ID"].values,
        "MERCHANT_ID": merch_rows["MERCHANT_ID"].values,
        "AMOUNT": amounts,
        "TIMESTAMP": timestamps,
        "MERCHANT_CATEGORY": merch_rows["MERCHANT_CATEGORY"].values,
        "LOCATION_LAT": cust_rows["HOME_LAT"].values + rng.normal(0, 0.3, n_transactions),
        "LOCATION_LONG": cust_rows["HOME_LON"].values + rng.normal(0, 0.3, n_transactions),
        "CARD_TYPE": cust_rows["CARD_TYPE"].values,
        "IS_ONLINE": rng.choice([0, 1], size=n_transactions, p=[0.6, 0.4]),
        "COUNTRY": ["US"] * n_transactions,
        "CUSTOMER_AGE": cust_rows["CUSTOMER_AGE"].values,
        "CUSTOMER_INCOME": cust_rows["CUSTOMER_INCOME"].values,
        "IS_FRAUD": 0,
    })


def generate_fraudulent_transactions(
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n_fraud: int,
    start_date: datetime,
    end_date: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate fraudulent transactions with realistic fraud patterns."""
    date_range_seconds = int((end_date - start_date).total_seconds())

    cust_idx = rng.choice(len(customers), size=n_fraud)
    merch_idx = rng.choice(len(merchants), size=n_fraud)

    cust_rows = customers.iloc[cust_idx].reset_index(drop=True)
    merch_rows = merchants.iloc[merch_idx].reset_index(drop=True)

    # Fraud amounts: higher than normal, skewed distribution
    amounts = []
    for cat in merch_rows["MERCHANT_CATEGORY"]:
        mean, std = CATEGORY_AMOUNTS[cat]
        fraud_multiplier = rng.uniform(2.5, 8.0)
        amt = max(50.0, rng.normal(mean * fraud_multiplier, std * 2))
        amounts.append(round(amt, 2))

    # Fraud timestamps: weighted toward late night / early morning
    hours = rng.choice(24, size=n_fraud, p=_fraud_hour_distribution())
    random_seconds = rng.integers(0, date_range_seconds, size=n_fraud)
    timestamps = [
        start_date + timedelta(seconds=int(s))
        for s in random_seconds
    ]
    timestamps = [
        t.replace(hour=int(h), minute=rng.integers(0, 60))
        for t, h in zip(timestamps, hours)
    ]

    # Geographic anomalies: transactions far from home
    location_lats = cust_rows["HOME_LAT"].values + rng.normal(0, 5, n_fraud)
    location_lons = cust_rows["HOME_LON"].values + rng.normal(0, 5, n_fraud)

    # Some international transactions
    countries = rng.choice(COUNTRIES, size=n_fraud)

    # Higher online rate for fraud
    is_online = rng.choice([0, 1], size=n_fraud, p=[0.3, 0.7])

    return pd.DataFrame({
        "CUSTOMER_ID": cust_rows["CUSTOMER_ID"].values,
        "MERCHANT_ID": merch_rows["MERCHANT_ID"].values,
        "AMOUNT": amounts,
        "TIMESTAMP": timestamps,
        "MERCHANT_CATEGORY": merch_rows["MERCHANT_CATEGORY"].values,
        "LOCATION_LAT": location_lats,
        "LOCATION_LONG": location_lons,
        "CARD_TYPE": cust_rows["CARD_TYPE"].values,
        "IS_ONLINE": is_online,
        "COUNTRY": countries,
        "CUSTOMER_AGE": cust_rows["CUSTOMER_AGE"].values,
        "CUSTOMER_INCOME": cust_rows["CUSTOMER_INCOME"].values,
        "IS_FRAUD": 1,
    })


def _hour_distribution() -> list:
    """Normal transaction hour distribution: peaked during business hours."""
    weights = [
        1, 1, 1, 1, 1, 2,       # 0-5: low activity
        4, 6, 8, 9, 10, 10,     # 6-11: morning ramp
        10, 9, 8, 8, 9, 10,     # 12-17: afternoon
        9, 8, 6, 4, 3, 2,       # 18-23: evening decline
    ]
    total = sum(weights)
    return [w / total for w in weights]


def _fraud_hour_distribution() -> list:
    """Fraud transaction hour distribution: peaked late night."""
    weights = [
        8, 9, 10, 10, 9, 7,     # 0-5: high fraud
        4, 3, 3, 3, 4, 4,       # 6-11: some fraud
        4, 4, 3, 3, 3, 4,       # 12-17: some fraud
        5, 5, 6, 7, 8, 8,       # 18-23: increasing
    ]
    total = sum(weights)
    return [w / total for w in weights]


def main(
    n_transactions: int = 100_000,
    n_customers: int = 5_000,
    n_merchants: int = 500,
    fraud_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic credit card transaction dataset.

    Args:
        n_transactions: Total number of transactions.
        n_customers: Number of unique customers.
        n_merchants: Number of unique merchants.
        fraud_rate: Proportion of fraudulent transactions.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic transaction data.
    """
    rng = np.random.default_rng(seed)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)

    customers = generate_customers(n_customers, rng)
    merchants = generate_merchants(n_merchants, rng)

    n_fraud = int(n_transactions * fraud_rate)
    n_normal = n_transactions - n_fraud

    normal_txns = generate_normal_transactions(
        customers, merchants, n_normal, start_date, end_date, rng
    )
    fraud_txns = generate_fraudulent_transactions(
        customers, merchants, n_fraud, start_date, end_date, rng
    )

    df = pd.concat([normal_txns, fraud_txns], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["TRANSACTION_ID"] = [f"TXN_{i:06d}" for i in range(len(df))]

    # Reorder columns
    cols = [
        "TRANSACTION_ID", "CUSTOMER_ID", "MERCHANT_ID", "AMOUNT",
        "TIMESTAMP", "MERCHANT_CATEGORY", "LOCATION_LAT", "LOCATION_LONG",
        "CARD_TYPE", "IS_ONLINE", "COUNTRY", "CUSTOMER_AGE",
        "CUSTOMER_INCOME", "IS_FRAUD",
    ]
    df = df[cols].sort_values("TIMESTAMP").reset_index(drop=True)

    print(f"Generated {len(df):,} transactions")
    print(f"  Fraud rate: {df['IS_FRAUD'].mean():.2%}")
    print(f"  Customers: {df['CUSTOMER_ID'].nunique():,}")
    print(f"  Merchants: {df['MERCHANT_ID'].nunique():,}")
    print(f"  Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")

    return df


if __name__ == "__main__":
    df = main()
    output_path = Path(__file__).parent / "synthetic_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
