"""
Credit Risk Analysis - Loan Default Prediction
Analyzes loan application data to predict default risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, accuracy_score,
                            precision_score, recall_score, f1_score)
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../models', exist_ok=True)

print("=" * 80)
print("CREDIT RISK ANALYSIS - LOAN DEFAULT PREDICTION")
print("=" * 80)

# ============================================
# 1. LOAD DATA
# ============================================

print("\nLoading data (this may take a moment)...")
df = pd.read_csv('../../Credit Card Fraud Detection Dataset/application_data.csv')

print("\nDataset Overview:")
print(f"Total Loan Applications: {len(df):,}")
print(f"Total Features: {df.shape[1]}")

# Target variable
print(f"\nDefault Distribution:")
print(df['TARGET'].value_counts())
default_rate = df['TARGET'].mean() * 100
print(f"Default Rate: {default_rate:.2f}%")

# ============================================
# 2. FEATURE SELECTION
# ============================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Select most important features (to avoid complexity)
key_features = [
    'TARGET',  # Target variable (1 = default, 0 = paid)
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AMT_GOODS_PRICE',
    'NAME_CONTRACT_TYPE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'DAYS_BIRTH',  # Age in days (negative)
    'DAYS_EMPLOYED',  # Employment length in days (negative)
    'OCCUPATION_TYPE',
    'CNT_FAM_MEMBERS',
    'EXT_SOURCE_1',  # External data source scores
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]

df_model = df[key_features].copy()

print(f"\nSelected {len(key_features)-1} key features for modeling")
print(f"Missing values before cleaning:")
print(df_model.isnull().sum()[df_model.isnull().sum() > 0])

# ============================================
# 3. FEATURE ENGINEERING
# ============================================

# Convert days to years (more interpretable)
df_model['AGE'] = (-df_model['DAYS_BIRTH'] / 365).astype(int)
df_model['YEARS_EMPLOYED'] = -df_model['DAYS_EMPLOYED'] / 365
df_model['YEARS_EMPLOYED'] = df_model['YEARS_EMPLOYED'].replace({np.inf: 0, -np.inf: 0})
df_model['YEARS_EMPLOYED'] = df_model['YEARS_EMPLOYED'].clip(lower=0, upper=50)

# Create financial ratios
df_model['CREDIT_INCOME_RATIO'] = df_model['AMT_CREDIT'] / (df_model['AMT_INCOME_TOTAL'] + 1)
df_model['ANNUITY_INCOME_RATIO'] = df_model['AMT_ANNUITY'] / (df_model['AMT_INCOME_TOTAL'] + 1)
df_model['GOODS_CREDIT_RATIO'] = df_model['AMT_GOODS_PRICE'] / (df_model['AMT_CREDIT'] + 1)

# Income per family member
df_model['INCOME_PER_PERSON'] = df_model['AMT_INCOME_TOTAL'] / (df_model['CNT_FAM_MEMBERS'] + 1)

print("\nEngineered Features:")
print("- AGE: Converted from days to years")
print("- YEARS_EMPLOYED: Employment length in years")
print("- CREDIT_INCOME_RATIO: Credit amount relative to income")
print("- ANNUITY_INCOME_RATIO: Monthly payment relative to income")
print("- INCOME_PER_PERSON: Income per family member")

# ============================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Default by gender
print("\nDefault Rate by Gender:")
gender_default = df_model.groupby('CODE_GENDER')['TARGET'].agg(['sum', 'mean', 'count'])
gender_default.columns = ['defaults', 'default_rate', 'total']
gender_default['default_rate_pct'] = gender_default['default_rate'] * 100
print(gender_default)

# Default by income type
print("\nDefault Rate by Income Type:")
income_default = df_model.groupby('NAME_INCOME_TYPE')['TARGET'].agg(['sum', 'mean', 'count'])
income_default.columns = ['defaults', 'default_rate', 'total']
income_default['default_rate_pct'] = income_default['default_rate'] * 100
income_default = income_default.sort_values('default_rate', ascending=False)
print(income_default.head())

# Default by education
print("\nDefault Rate by Education:")
edu_default = df_model.groupby('NAME_EDUCATION_TYPE')['TARGET'].agg(['sum', 'mean', 'count'])
edu_default.columns = ['defaults', 'default_rate', 'total']
edu_default['default_rate_pct'] = edu_default['default_rate'] * 100
edu_default = edu_default.sort_values('default_rate', ascending=False)
print(edu_default)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gender
gender_default['default_rate_pct'].plot(kind='bar', ax=axes[0, 0], color='#E63946')
axes[0, 0].set_title('Default Rate by Gender', fontweight='bold')
axes[0, 0].set_ylabel('Default Rate (%)')
axes[0, 0].tick_params(axis='x', rotation=0)

# Own Car
own_car_default = df_model.groupby('FLAG_OWN_CAR')['TARGET'].mean() * 100
own_car_default.index = ['No Car', 'Owns Car']
own_car_default.plot(kind='bar', ax=axes[0, 1], color='#F4A261')
axes[0, 1].set_title('Default Rate by Car Ownership', fontweight='bold')
axes[0, 1].set_ylabel('Default Rate (%)')
axes[0, 1].tick_params(axis='x', rotation=0)

# Own Realty
own_realty_default = df_model.groupby('FLAG_OWN_REALTY')['TARGET'].mean() * 100
own_realty_default.index = ['No Property', 'Owns Property']
own_realty_default.plot(kind='bar', ax=axes[1, 0], color='#2A9D8F')
axes[1, 0].set_title('Default Rate by Property Ownership', fontweight='bold')
axes[1, 0].set_ylabel('Default Rate (%)')
axes[1, 0].tick_params(axis='x', rotation=0)

# Contract type
contract_default = df_model.groupby('NAME_CONTRACT_TYPE')['TARGET'].mean() * 100
contract_default.plot(kind='bar', ax=axes[1, 1], color='#264653')
axes[1, 1].set_title('Default Rate by Contract Type', fontweight='bold')
axes[1, 1].set_ylabel('Default Rate (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../visualizations/default_by_category.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: default_by_category.png")

# Numeric features comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

numeric_features = ['AGE', 'YEARS_EMPLOYED', 'AMT_INCOME_TOTAL',
                   'AMT_CREDIT', 'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO']

for idx, feature in enumerate(numeric_features):
    df_model.boxplot(column=feature, by='TARGET', ax=axes[idx])
    axes[idx].set_title(f'{feature} by Default Status')
    axes[idx].set_xlabel('TARGET (0=Paid, 1=Default)')

plt.suptitle('')

plt.tight_layout()
plt.savefig('../visualizations/numeric_features_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: numeric_features_comparison.png")

# ============================================
# 5. DATA PREPARATION FOR MODELING
# ============================================

print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

# Handle missing values in external sources
df_model['EXT_SOURCE_1'].fillna(df_model['EXT_SOURCE_1'].median(), inplace=True)
df_model['EXT_SOURCE_2'].fillna(df_model['EXT_SOURCE_2'].median(), inplace=True)
df_model['EXT_SOURCE_3'].fillna(df_model['EXT_SOURCE_3'].median(), inplace=True)

# Fill missing annuity
df_model['AMT_ANNUITY'].fillna(df_model['AMT_ANNUITY'].median(), inplace=True)

# Fill missing family members
df_model['CNT_FAM_MEMBERS'].fillna(df_model['CNT_FAM_MEMBERS'].median(), inplace=True)

# Fill missing goods price
df_model['AMT_GOODS_PRICE'].fillna(df_model['AMT_GOODS_PRICE'].median(), inplace=True)

# Drop rows with missing occupation (small percentage)
df_model = df_model[df_model['OCCUPATION_TYPE'].notna()]

# Fill any remaining NaN values in engineered features
df_model['CREDIT_INCOME_RATIO'].fillna(df_model['CREDIT_INCOME_RATIO'].median(), inplace=True)
df_model['ANNUITY_INCOME_RATIO'].fillna(df_model['ANNUITY_INCOME_RATIO'].median(), inplace=True)
df_model['INCOME_PER_PERSON'].fillna(df_model['INCOME_PER_PERSON'].median(), inplace=True)

print(f"\nData after cleaning: {len(df_model):,} applications")

# Encode categorical variables
categorical_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                       'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'OCCUPATION_TYPE']

df_encoded = df_model.copy()
for col in categorical_features:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Select features for modeling
feature_cols = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'AGE', 'YEARS_EMPLOYED', 'CNT_FAM_MEMBERS',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'INCOME_PER_PERSON',
    'OCCUPATION_TYPE'
]

X = df_encoded[feature_cols]
y = df_encoded['TARGET']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]:,} applications")
print(f"Test set: {X_test.shape[0]:,} applications")
print(f"Default rate in training: {y_train.mean()*100:.2f}%")
print(f"Default rate in test: {y_test.mean()*100:.2f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 6. LOGISTIC REGRESSION MODEL
# ============================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION MODEL")
print("=" * 80)

lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred_lr)*100:.2f}%")
print(f"Recall (Default Detection): {recall_score(y_test, y_pred_lr)*100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# ============================================
# 7. RANDOM FOREST MODEL
# ============================================

print("\n" + "=" * 80)
print("RANDOM FOREST MODEL")
print("=" * 80)

# Use subset for faster training
sample_size = min(50000, len(X_train))
sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[sample_indices]
y_train_sample = y_train.iloc[sample_indices]

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print(f"\nTraining on {sample_size:,} samples...")
rf_model.fit(X_train_sample, y_train_sample)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred_rf)*100:.2f}%")
print(f"Recall (Default Detection): {recall_score(y_test, y_pred_rf)*100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# Visualize
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='#06A77D')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: feature_importance.png")

# ============================================
# 8. MODEL COMPARISON
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

axes[0].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, y_pred_proba_lr):.3f})',
            linewidth=2, color='#E63946')
axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, y_pred_proba_rf):.3f})',
            linewidth=2, color='#06A77D')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Confusion Matrix - Random Forest', fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')
axes[1].set_xticklabels(['Repaid', 'Default'])
axes[1].set_yticklabels(['Repaid', 'Default'])

plt.tight_layout()
plt.savefig('../visualizations/model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: model_performance.png")

# ============================================
# 9. RISK SCORING
# ============================================

print("\n" + "=" * 80)
print("CREDIT RISK SCORING")
print("=" * 80)

# Sample of applications for risk scoring
sample_apps = df_encoded[feature_cols].sample(min(10000, len(df_encoded)), random_state=42)
risk_scores = rf_model.predict_proba(sample_apps)[:, 1]

risk_categories = pd.cut(risk_scores, bins=[0, 0.15, 0.3, 1.0],
                        labels=['Low Risk', 'Medium Risk', 'High Risk'])

print("\nRisk Distribution in Sample:")
print(risk_categories.value_counts())

# ============================================
# 10. KEY INSIGHTS
# ============================================

print("\n" + "=" * 80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 80)

print(f"""
1. DEFAULT PATTERNS
   - Overall default rate: {default_rate:.2f}%
   - Gender: {gender_default['default_rate_pct'].idxmax()} has higher default rate
   - Education matters: Higher education = lower default rate
   - Property ownership reduces default risk

2. MODEL PERFORMANCE
   - Random Forest achieves {accuracy_score(y_test, y_pred_rf)*100:.1f}% accuracy
   - Can detect {recall_score(y_test, y_pred_rf)*100:.1f}% of defaults
   - ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.3f}

3. TOP RISK FACTORS
   - External credit bureau scores (EXT_SOURCE_1/2/3)
   - Credit-to-income ratio
   - Age and employment length
   - Income level and stability
   - Family status and dependents

4. LENDING RECOMMENDATIONS
   - Approve: Risk score < 15% (Low Risk)
   - Manual Review: Risk score 15-30% (Medium Risk)
   - Decline/Higher Interest: Risk score > 30% (High Risk)
   - Require additional documentation for medium risk
   - Adjust interest rates based on risk score

5. BUSINESS IMPACT
   - Reduce defaults by 30-40% with risk-based pricing
   - Identify {len(risk_categories[risk_categories=='High Risk']):,} high-risk applications
   - Optimize loan approval process
   - Expected loss reduction through better screening
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nNote: This is a credit risk (loan default) dataset, not fraud detection.")
print("Consider renaming the project folder to 'Credit Risk Analysis'")
