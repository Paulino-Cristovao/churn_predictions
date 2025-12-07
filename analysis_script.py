import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


import warnings

warnings.filterwarnings('ignore')

# --- 1. Pydantic Models ---
class Subscription(BaseModel):
    subscription_id: str
    subscription_created_at: datetime
    vendor: Optional[str]
    v2_segment: Optional[str]
    v2_modules: Optional[str]
    naf_code: Optional[str]
    naf_section: Optional[str]
    revenue_range: Optional[str]
    employee_count: Optional[str]
    regional_pole: Optional[str]
    market: Optional[str]
    legal_structure: Optional[str]
    company_age_in_years: Optional[float]
    company_age_group: Optional[str]
    subscription_status: str
    trial_starts_at: datetime
    trial_ends_at: datetime
    canceled_at: Optional[datetime]
    first_paid_invoice_paid_at: Optional[datetime]

# --- 2. Data Loading ---
print("Loading data...")
daily_usage_df = pd.read_csv('Data/daily_usage.csv')
subscriptions_df = pd.read_csv('Data/subscriptions.csv')

print(f"Daily Usage shape: {daily_usage_df.shape}")
print(f"Subscriptions shape: {subscriptions_df.shape}")

# --- 3. Data Preprocessing & Cleaning ---
print("Preprocessing data...")

# Convert date columns to datetime
date_cols = ['subscription_created_at', 'trial_starts_at', 'trial_ends_at', 'canceled_at', 'first_paid_invoice_paid_at']
for col in date_cols:
    subscriptions_df[col] = pd.to_datetime(subscriptions_df[col], errors='coerce')

# Filter for 15-day trials
subscriptions_df['trial_duration'] = (subscriptions_df['trial_ends_at'] - subscriptions_df['trial_starts_at']).dt.days
subscriptions_df = subscriptions_df[subscriptions_df['trial_duration'] == 15].copy()
print(f"Subscriptions after filtering for 15-day trials: {subscriptions_df.shape}")

# Define Target Variable
# Converted = 1 if first_paid_invoice_paid_at is present, else 0
subscriptions_df['converted'] = subscriptions_df['first_paid_invoice_paid_at'].notna().astype(int)
print(f"Target distribution:\n{subscriptions_df['converted'].value_counts(normalize=True)}")

# Aggregate Daily Usage
# We sum up all usage stats per subscription
usage_cols = [col for col in daily_usage_df.columns if col not in ['subscription_id', 'day_date']]
usage_agg = daily_usage_df.groupby('subscription_id')[usage_cols].sum().reset_index()

# Merge Subscriptions with Usage
df = pd.merge(subscriptions_df, usage_agg, on='subscription_id', how='left')

# Fill missing usage with 0 (assuming no record means no usage)
df[usage_cols] = df[usage_cols].fillna(0)

# --- 4. Feature Engineering ---
print("Feature Engineering...")

# Select Features
# We exclude ID, dates, and target-leaky columns (status, canceled_at, first_paid_invoice_paid_at)
categorical_features = ['vendor', 'v2_segment', 'naf_section', 'revenue_range', 'employee_count', 'regional_pole', 'legal_structure', 'company_age_group']
numerical_features = ['company_age_in_years'] + usage_cols

# Clean Categorical Features
for col in categorical_features:
    df[col] = df[col].fillna('Unknown')

# Clean Numerical Features
df['company_age_in_years'] = pd.to_numeric(df['company_age_in_years'], errors='coerce')
df['company_age_in_years'] = df['company_age_in_years'].fillna(df['company_age_in_years'].median())

# Prepare X and y
X = df[categorical_features + numerical_features]
y = df['converted']

# Split Data
# Using a temporal split based on trial_started_at would be ideal, but for simplicity and size we'll use random split
# However, let's respect the prompt's mention of "Temporal train/test (80/20 by trial start date)"
df = df.sort_values('trial_starts_at')
train_size = int(0.8 * len(df))
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 5. Modeling ---
print("Training models...")

results = {}

# Logistic Regression
lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=1000))])
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'AUC': roc_auc_score(y_test, y_prob_lr),
    'y_prob': y_prob_lr
}

# XGBoost
print("Attempting to import XGBoost...")
try:
    import xgboost as xgb
    xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'Accuracy': accuracy_score(y_test, y_pred_xgb),
        'AUC': roc_auc_score(y_test, y_prob_xgb),
        'y_prob': y_prob_xgb
    }
except Exception as e:
    print(f"XGBoost import failed: {e}. Using GradientBoostingClassifier instead.")
    from sklearn.ensemble import GradientBoostingClassifier
    xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', GradientBoostingClassifier(random_state=42))])
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    results['Gradient Boosting'] = {
        'Accuracy': accuracy_score(y_test, y_pred_xgb),
        'AUC': roc_auc_score(y_test, y_prob_xgb),
        'y_prob': y_prob_xgb
    }

# LightGBM
print("Attempting to import LightGBM...")
try:
    import lightgbm as lgb
    lgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))])
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
    results['LightGBM'] = {
        'Accuracy': accuracy_score(y_test, y_pred_lgb),
        'AUC': roc_auc_score(y_test, y_prob_lgb),
        'y_prob': y_prob_lgb
    }
except Exception as e:
    print(f"LightGBM import failed: {e}. Using HistGradientBoostingClassifier instead.")
    from sklearn.ensemble import HistGradientBoostingClassifier
    lgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', HistGradientBoostingClassifier(random_state=42))])
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
    results['HistGradientBoosting'] = {
        'Accuracy': accuracy_score(y_test, y_pred_lgb),
        'AUC': roc_auc_score(y_test, y_prob_lgb),
        'y_prob': y_prob_lgb
    }

# --- 6. Evaluation & Plotting ---
print("\nResults:")
for name, metrics in results.items():
    print(f"{name}: Accuracy={metrics['Accuracy']:.4f}, AUC={metrics['AUC']:.4f}")

# Plot ROC Curves
plt.figure(figsize=(10, 8))
for name, metrics in results.items():
    fpr, tpr, _ = roc_curve(y_test, metrics['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['AUC']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curves.png')
print("Saved ROC curves to roc_curves.png")

# Feature Importance
# We try to get feature importance from the best available model
print("Generating Feature Importance plot...")
model_for_importance = None
model_name = ""

if hasattr(lgb_model.named_steps['classifier'], 'feature_importances_'):
    model_for_importance = lgb_model
    model_name = "LightGBM"
elif hasattr(xgb_model.named_steps['classifier'], 'feature_importances_'):
    model_for_importance = xgb_model
    model_name = "XGBoost/GradientBoosting"
else:
    print("No tree-based model with feature_importances_ available.")

if model_for_importance:
    # Extract feature names after one-hot encoding
    feature_names = numerical_features + \
        list(model_for_importance.named_steps['preprocessor'].named_transformers_['cat']
             .named_steps['onehot'].get_feature_names_out(categorical_features))

    clf = model_for_importance.named_steps['classifier']
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot Top 20 Features
    plt.figure(figsize=(12, 10))
    plt.title(f"Feature Importances ({model_name})")
    plt.barh(range(min(20, len(indices))), importances[indices[:20]], align="center")
    plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved feature importance to feature_importance.png")

print("Analysis complete.")
