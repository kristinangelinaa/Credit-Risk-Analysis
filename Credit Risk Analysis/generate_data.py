"""
Credit Card Transaction Data Generator
Generates realistic transaction data with fraud cases for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate transaction data
num_transactions = 100000
num_customers = 5000

# Date range: 6 months of transactions
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 6, 30)

transactions = []

# Merchant categories
merchant_categories = [
    'Grocery', 'Gas Station', 'Restaurant', 'Online Retail',
    'Electronics', 'Travel', 'Entertainment', 'Healthcare',
    'Utilities', 'Clothing', 'Home Improvement', 'Other'
]

# Transaction types
transaction_types = ['Purchase', 'Online', 'Contactless', 'Chip', 'Swipe', 'ATM']

# Generate customer profiles
customer_profiles = {}
for i in range(1, num_customers + 1):
    customer_profiles[i] = {
        'avg_transaction': np.random.uniform(30, 200),
        'std_transaction': np.random.uniform(10, 100),
        'preferred_categories': random.sample(merchant_categories, k=random.randint(2, 5)),
        'avg_daily_transactions': np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05]),
        'home_location': random.choice(['USA', 'Canada', 'UK']),
    }

transaction_id = 1

for _ in range(num_transactions):
    # Random customer
    customer_id = random.randint(1, num_customers)
    profile = customer_profiles[customer_id]

    # Random timestamp
    random_days = random.randint(0, (end_date - start_date).days)
    transaction_date = start_date + timedelta(days=random_days)
    transaction_time = transaction_date + timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )

    # Determine if this is fraud (2% fraud rate)
    is_fraud = np.random.random() < 0.02

    if is_fraud:
        # Fraudulent transaction characteristics
        amount = np.random.choice([
            np.random.uniform(500, 2000),  # Large unusual amount
            np.random.uniform(1, 10),      # Small test transaction
        ], p=[0.7, 0.3])

        # More likely foreign or online
        location = random.choice(['Foreign', 'USA', 'Online'], p=[0.4, 0.2, 0.4])

        # Random category (not necessarily preferred)
        merchant_category = random.choice(merchant_categories)

        # More likely online or swipe
        transaction_type = random.choice(['Online', 'Swipe', 'ATM'], p=[0.5, 0.3, 0.2])

        # Unusual time (late night)
        if random.random() < 0.5:
            transaction_time = transaction_time.replace(hour=random.randint(0, 5))

        # Card present less likely
        card_present = 0 if random.random() < 0.7 else 1

        # Failed PIN attempts possible
        failed_pin_attempts = random.choice([0, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])

    else:
        # Normal transaction
        amount = max(1, np.random.normal(profile['avg_transaction'], profile['std_transaction']))

        # Mostly domestic
        location = profile['home_location'] if random.random() < 0.95 else random.choice(['Foreign', 'Online'])

        # Preferred category
        merchant_category = random.choice(profile['preferred_categories']) if random.random() < 0.8 else random.choice(merchant_categories)

        # Normal transaction types
        transaction_type = random.choice(transaction_types, p=[0.2, 0.25, 0.25, 0.2, 0.08, 0.02])

        # Normal hours (6 AM - 11 PM)
        if random.random() < 0.9:
            transaction_time = transaction_time.replace(hour=random.randint(6, 23))

        # Card usually present for in-person
        card_present = 1 if transaction_type in ['Chip', 'Contactless', 'Swipe'] else 0

        # Rarely failed PIN
        failed_pin_attempts = 0 if random.random() < 0.98 else 1

    # Additional features
    distance_from_home = 0 if location == profile['home_location'] else np.random.uniform(100, 3000)

    # Time since last transaction (in minutes)
    time_since_last = np.random.exponential(500) if not is_fraud else np.random.exponential(20)

    # Chip technology usage
    chip_used = 1 if transaction_type in ['Chip', 'Contactless'] else 0

    # Online transaction flag
    is_online = 1 if transaction_type == 'Online' or location == 'Online' else 0

    # Weekend flag
    is_weekend = 1 if transaction_time.weekday() >= 5 else 0

    # Hour of day
    hour_of_day = transaction_time.hour

    # Amount deviation from customer's average
    amount_deviation = abs(amount - profile['avg_transaction']) / (profile['std_transaction'] + 1)

    transactions.append({
        'transaction_id': f'TXN_{transaction_id:08d}',
        'customer_id': f'CUST_{customer_id:05d}',
        'transaction_datetime': transaction_time,
        'amount': round(amount, 2),
        'merchant_category': merchant_category,
        'transaction_type': transaction_type,
        'location': location,
        'card_present': card_present,
        'chip_used': chip_used,
        'is_online': is_online,
        'failed_pin_attempts': failed_pin_attempts,
        'distance_from_home': round(distance_from_home, 2),
        'time_since_last_txn': round(time_since_last, 2),
        'hour_of_day': hour_of_day,
        'is_weekend': is_weekend,
        'amount_deviation': round(amount_deviation, 2),
        'is_fraud': is_fraud
    })

    transaction_id += 1

# Create DataFrame
df = pd.DataFrame(transactions)

# Sort by datetime
df = df.sort_values('transaction_datetime').reset_index(drop=True)

# Save to CSV
df.to_csv('data/credit_card_transactions.csv', index=False)

print("=" * 70)
print("CREDIT CARD TRANSACTION DATASET GENERATED")
print("=" * 70)
print(f"\nTotal Transactions: {len(df):,}")
print(f"Fraudulent Transactions: {df['is_fraud'].sum():,}")
print(f"Fraud Rate: {df['is_fraud'].mean()*100:.2f}%")
print(f"Legitimate Transactions: {(~df['is_fraud']).sum():,}")
print(f"\nDate Range: {df['transaction_datetime'].min()} to {df['transaction_datetime'].max()}")
print(f"Unique Customers: {df['customer_id'].nunique():,}")

print("\n" + "-" * 70)
print("FRAUD STATISTICS")
print("-" * 70)

print("\nFraud by Location:")
print(df.groupby('location')['is_fraud'].agg(['sum', 'mean']))

print("\nFraud by Transaction Type:")
print(df.groupby('transaction_type')['is_fraud'].agg(['sum', 'mean']))

print("\nFraud by Merchant Category:")
fraud_by_category = df.groupby('merchant_category')['is_fraud'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
print(fraud_by_category.head())

print("\n" + "=" * 70)
print("Dataset saved to 'data/credit_card_transactions.csv'")
print("=" * 70)
