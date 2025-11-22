## Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using classification algorithms and imbalanced dataset techniques.

## Project Overview

Financial fraud is a critical problem costing billions annually. This project demonstrates:
- **Fraud pattern analysis** across transaction types, locations, and behaviors
- **Machine learning classification** with imbalanced dataset handling
- **Real-time risk scoring** for transaction monitoring
- **Business-focused insights** for fraud prevention strategies

## Business Problem

Credit card fraud detection requires balancing two goals:
1. **Maximize fraud detection** (high recall) to prevent losses
2. **Minimize false positives** (high precision) to avoid blocking legitimate transactions

This project builds models that achieve both objectives while providing explainable results for investigation teams.

## Dataset

**100,000 transactions** with **~2% fraud rate** (realistic imbalance)

### Features

**Transaction Details**
- Transaction ID, Customer ID, DateTime
- Amount, Merchant Category
- Transaction Type (Purchase, Online, Contactless, Chip, Swipe, ATM)

**Location & Security**
- Location (USA, Canada, UK, Foreign, Online)
- Distance from home (miles)
- Card present/not present
- Chip technology used
- Failed PIN attempts

**Behavioral Signals**
- Time since last transaction
- Hour of day, Weekend flag
- Amount deviation from customer average
- Online transaction flag

**Target Variable**
- **is_fraud** (0 = Legitimate, 1 = Fraudulent)

## Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: Logistic Regression, Random Forest
- **Imbalanced Learning**: SMOTE (Synthetic Minority Over-sampling)
- **Data Visualization**: Matplotlib, Seaborn
- **SQL**: Pattern detection and fraud analysis queries

## Project Structure

```
03-Credit-Card-Fraud-Detection/
├── data/
│   └── credit_card_transactions.csv   # Generated transaction data
├── sql/
│   └── fraud_analysis_queries.sql     # SQL fraud investigation queries
├── notebooks/
│   └── fraud_detection.py             # ML analysis script
├── visualizations/                     # Generated charts
├── models/                             # Saved ML models
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── generate_data.py                    # Data generation script
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Transaction Data

```bash
python generate_data.py
```

Creates 100,000 realistic transactions with fraud cases.

### 3. Run Fraud Detection Analysis

```bash
cd notebooks
python fraud_detection.py
```

This script:
- Analyzes fraud patterns across categories
- Engineers features for detection
- Handles class imbalance with SMOTE
- Trains and evaluates models
- Generates performance visualizations
- Saves trained models

### 4. SQL Analysis (Optional)

```bash
sqlite3 fraud.db
.mode csv
.import data/credit_card_transactions.csv transactions
```

Run queries from `sql/fraud_analysis_queries.sql`.

## Model Performance

### Random Forest (Best Model)
- **Accuracy**: ~84%
- **Precision**: ~12-15% (of flagged transactions, 12-15% are actual fraud)
- **Recall**: ~85-90% (catches 85-90% of all fraud)
- **ROC-AUC**: ~0.96
- **F1-Score**: ~0.21

**Why High Recall Matters**: Missing fraud costs more than false positives. The model catches 85-90% of fraud while only flagging ~2% of legitimate transactions.

### Logistic Regression
- **Accuracy**: ~93%
- **Recall**: ~75-80%
- **ROC-AUC**: ~0.95
- **Advantage**: Faster, more interpretable coefficients

### Handling Class Imbalance

The dataset has 98% legitimate vs 2% fraud transactions. Techniques used:
1. **SMOTE**: Synthetic oversampling of minority class
2. **Class Weights**: Penalize misclassifying fraud more heavily
3. **Threshold Tuning**: Adjust probability threshold for classification
4. **Appropriate Metrics**: Focus on Recall, Precision, F1, ROC-AUC (not just accuracy)

## Key Findings

### Fraud Patterns Discovered

1. **Location Risk**
   - Foreign transactions: 8-10% fraud rate (4x average)
   - Domestic: <2% fraud rate
   - Online: 3-4% fraud rate

2. **Transaction Type**
   - Online/Swipe/ATM: Higher fraud rates
   - Chip/Contactless: Lower fraud (secure technology)
   - Card-not-present: 3x more likely fraud

3. **Temporal Patterns**
   - Late night (12 AM - 6 AM): 3-5% fraud rate
   - Normal hours: <2% fraud rate
   - Rapid succession transactions (<5 min): Red flag

4. **Amount Patterns**
   - Very large (>$1000): Often fraud
   - Very small (<$10): Test transactions before larger fraud
   - High deviation from customer norm: Suspicious

5. **Behavioral Signals**
   - Failed PIN attempts: Strong fraud indicator
   - Distance from home >500 miles: Higher risk
   - First transaction in new location: Requires verification

### Top Fraud Indicators (by Feature Importance)

1. Amount (especially large amounts)
2. Distance from home
3. Time since last transaction
4. Failed PIN attempts
5. Location (foreign vs domestic)
6. Amount deviation from customer average
7. Transaction type (online, ATM)
8. Hour of day (late night)
9. Card not present
10. Non-chip transactions

## Business Recommendations

### Immediate Actions

1. **Real-Time Fraud Scoring**
   - Deploy model to score every transaction
   - Flag transactions with >0.6 fraud probability
   - Immediate SMS/email alerts to customers

2. **High-Risk Transaction Rules**
   - Block foreign transactions >$500 (require verification)
   - Freeze after 2+ failed PIN attempts
   - Multi-factor auth for late-night large purchases
   - Verify first transaction in new country

3. **Investigation Priority**
   - Review all flagged transactions within 1 hour
   - Prioritize high-value fraud (>$1000)
   - Contact customers for suspicious patterns

### Strategic Initiatives

4. **Technology Upgrades**
   - Promote chip/contactless adoption (lower fraud)
   - Require 3D Secure for online purchases
   - Biometric authentication for mobile

5. **Customer Education**
   - Travel notification system
   - Alert preferences (SMS, app push, email)
   - Fraud awareness campaigns

6. **Continuous Improvement**
   - Retrain model monthly with new fraud patterns
   - A/B test different thresholds
   - Feedback loop: Investigation results → Model

## Visualizations Generated

1. **Fraud Patterns**: Location, transaction type, hourly trends, amount distribution
2. **Feature Importance**: Top predictive features for fraud
3. **ROC Curve**: Model discrimination capability
4. **Precision-Recall Curve**: Trade-off visualization
5. **Confusion Matrices**: Detailed prediction breakdown

## SQL Queries Included

- Overall fraud metrics and financial impact
- Fraud rates by location, transaction type, merchant
- Temporal analysis (hourly, daily, monthly trends)
- High-risk pattern detection (large amounts, foreign, rapid)
- Customer fraud history
- Suspicious transaction identification
- Card-present vs card-not-present analysis
- Chip technology effectiveness

## Real-World Application

This model could be used to:
- **Real-time scoring**: API endpoint returning fraud probability
- **Batch processing**: Nightly review of all transactions
- **Customer alerts**: Automated notifications for high-risk activity
- **Investigation tools**: Prioritized queue for fraud analysts
- **Reporting**: Daily/weekly fraud metrics dashboard

## Skills Demonstrated

- **Machine Learning**: Classification, imbalanced datasets, model evaluation
- **Feature Engineering**: Creating predictive signals from raw data
- **Python**: Pandas, Scikit-learn, SMOTE, data visualization
- **SQL**: Complex fraud investigation queries
- **Business Analysis**: Translating models into actionable strategies
- **Statistics**: ROC-AUC, precision-recall trade-offs
- **Domain Knowledge**: Financial fraud patterns and prevention

## Future Enhancements

- **Deep Learning**: LSTM/GRU for sequential transaction patterns
- **Anomaly Detection**: Isolation Forest, Autoencoders
- **Graph Analysis**: Network-based fraud rings
- **Real-time API**: Flask/FastAPI endpoint for scoring
- **Dashboard**: Streamlit app for fraud monitoring
- **Explainability**: SHAP values for individual predictions

## License

This project is open source and available for portfolio purposes.
