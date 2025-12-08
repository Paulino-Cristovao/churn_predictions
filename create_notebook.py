import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kolecto Churn Prediction Analysis\n",
    "\n",
    "This notebook analyzes user behavior during the 15-day trial period to predict conversion to paid subscriptions.\n",
    "We use Logistic Regression, XGBoost, and LightGBM models."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 0. Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pydantic import BaseModel, Field, ValidationError\n",
    "from typing import Optional, List\n",
    "from datetime import datetime\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Pydantic Models for Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Subscription(BaseModel):\n",
    "    subscription_id: str\n",
    "    subscription_created_at: datetime\n",
    "    vendor: Optional[str]\n",
    "    v2_segment: Optional[str]\n",
    "    v2_modules: Optional[str]\n",
    "    naf_code: Optional[str]\n",
    "    naf_section: Optional[str]\n",
    "    revenue_range: Optional[str]\n",
    "    employee_count: Optional[str]\n",
    "    regional_pole: Optional[str]\n",
    "    market: Optional[str]\n",
    "    legal_structure: Optional[str]\n",
    "    company_age_in_years: Optional[float]\n",
    "    company_age_group: Optional[str]\n",
    "    subscription_status: str\n",
    "    trial_starts_at: datetime\n",
    "    trial_ends_at: datetime\n",
    "    canceled_at: Optional[datetime]\n",
    "    first_paid_invoice_paid_at: Optional[datetime]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Loading data...\")\n",
    "daily_usage_df = pd.read_csv('Data/daily_usage.csv')\n",
    "subscriptions_df = pd.read_csv('Data/subscriptions.csv')\n",
    "\n",
    "print(f\"Daily Usage shape: {daily_usage_df.shape}\")\n",
    "print(f\"Subscriptions shape: {subscriptions_df.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.1 Date Conversion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert date columns to datetime\n",
    "date_cols = ['subscription_created_at', 'trial_starts_at', 'trial_ends_at', 'canceled_at', 'first_paid_invoice_paid_at']\n",
    "for col in date_cols:\n",
    "    subscriptions_df[col] = pd.to_datetime(subscriptions_df[col], errors='coerce')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 Filter Trials"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter for 15-day trials\n",
    "subscriptions_df['trial_duration'] = (subscriptions_df['trial_ends_at'] - subscriptions_df['trial_starts_at']).dt.days\n",
    "subscriptions_df = subscriptions_df[subscriptions_df['trial_duration'] == 15].copy()\n",
    "print(f\"Subscriptions after filtering for 15-day trials: {subscriptions_df.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.3 Target Definition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define Target Variable\n",
    "# Converted = 1 if first_paid_invoice_paid_at is present, else 0\n",
    "subscriptions_df['converted'] = subscriptions_df['first_paid_invoice_paid_at'].notna().astype(int)\n",
    "print(f\"Target distribution:\\n{subscriptions_df['converted'].value_counts(normalize=True)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.4 Usage Aggregation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Aggregate Daily Usage\n",
    "# We sum up all usage stats per subscription\n",
    "usage_cols = [col for col in daily_usage_df.columns if col not in ['subscription_id', 'day_date']]\n",
    "usage_agg = daily_usage_df.groupby('subscription_id')[usage_cols].sum().reset_index()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.5 Merge Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Merge Subscriptions with Usage\n",
    "df = pd.merge(subscriptions_df, usage_agg, on='subscription_id', how='left')\n",
    "\n",
    "# Fill missing usage with 0 (assuming no record means no usage)\n",
    "df[usage_cols] = df[usage_cols].fillna(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature Engineering"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.1 Feature Selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Select Features\n",
    "categorical_features = ['vendor', 'v2_segment', 'naf_section', 'revenue_range', 'employee_count', 'regional_pole', 'legal_structure', 'company_age_group']\n",
    "numerical_features = ['company_age_in_years'] + usage_cols"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Handling Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean Categorical Features\n",
    "for col in categorical_features:\n",
    "    df[col] = df[col].fillna('Unknown')\n",
    "\n",
    "# Clean Numerical Features\n",
    "df['company_age_in_years'] = pd.to_numeric(df['company_age_in_years'], errors='coerce')\n",
    "df['company_age_in_years'] = df['company_age_in_years'].fillna(df['company_age_in_years'].median())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.3 Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare X and y\n",
    "X = df[categorical_features + numerical_features]\n",
    "y = df['converted']\n",
    "\n",
    "# Split Data\n",
    "df = df.sort_values('trial_starts_at')\n",
    "train_size = int(0.8 * len(df))\n",
    "X_train = X.iloc[:train_size]\n",
    "y_train = y.iloc[:train_size]\n",
    "X_test = X.iloc[train_size:]\n",
    "y_test = y.iloc[train_size:]\n",
    "\n",
    "print(f\"Train set: {X_train.shape}, Test set: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.4 Preprocessing Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preprocessing Pipeline\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('scaler', StandardScaler())\n",
    "])\n",
    "\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),\n",
    "    ('onehot', OneHotEncoder(handle_unknown='ignore'))\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numeric_transformer, numerical_features),\n",
    "        ('cat', categorical_transformer, categorical_features)\n",
    "    ])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = {}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.1 Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logistic Regression\n",
    "lr_model = Pipeline(steps=[('preprocessor', preprocessor),\n",
    "                           ('classifier', LogisticRegression(max_iter=1000))])\n",
    "lr_model.fit(X_train, y_train)\n",
    "y_pred_lr = lr_model.predict(X_test)\n",
    "y_prob_lr = lr_model.predict_proba(X_test)[:, 1]\n",
    "results['Logistic Regression'] = {\n",
    "    'Accuracy': accuracy_score(y_test, y_pred_lr),\n",
    "    'AUC': roc_auc_score(y_test, y_prob_lr),\n",
    "    'y_prob': y_prob_lr\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.2 XGBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# XGBoost\n",
    "print(\"Attempting to import XGBoost...\")\n",
    "try:\n",
    "    import xgboost as xgb\n",
    "    xgb_model = Pipeline(steps=[('preprocessor', preprocessor),\n",
    "                                ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])\n",
    "    xgb_model.fit(X_train, y_train)\n",
    "    y_pred_xgb = xgb_model.predict(X_test)\n",
    "    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]\n",
    "    results['XGBoost'] = {\n",
    "        'Accuracy': accuracy_score(y_test, y_pred_xgb),\n",
    "        'AUC': roc_auc_score(y_test, y_prob_xgb),\n",
    "        'y_prob': y_prob_xgb\n",
    "    }\n",
    "except Exception as e:\n",
    "    print(f\"XGBoost import failed: {e}. Using GradientBoostingClassifier instead.\")\n",
    "    from sklearn.ensemble import GradientBoostingClassifier\n",
    "    xgb_model = Pipeline(steps=[('preprocessor', preprocessor),\n",
    "                                ('classifier', GradientBoostingClassifier(random_state=42))])\n",
    "    xgb_model.fit(X_train, y_train)\n",
    "    y_pred_xgb = xgb_model.predict(X_test)\n",
    "    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]\n",
    "    results['Gradient Boosting'] = {\n",
    "        'Accuracy': accuracy_score(y_test, y_pred_xgb),\n",
    "        'AUC': roc_auc_score(y_test, y_prob_xgb),\n",
    "        'y_prob': y_prob_xgb\n",
    "    }"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.3 LightGBM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# LightGBM\n",
    "print(\"Attempting to import LightGBM...\")\n",
    "try:\n",
    "    import lightgbm as lgb\n",
    "    lgb_model = Pipeline(steps=[('preprocessor', preprocessor),\n",
    "                                ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))])\n",
    "    lgb_model.fit(X_train, y_train)\n",
    "    y_pred_lgb = lgb_model.predict(X_test)\n",
    "    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]\n",
    "    results['LightGBM'] = {\n",
    "        'Accuracy': accuracy_score(y_test, y_pred_lgb),\n",
    "        'AUC': roc_auc_score(y_test, y_prob_lgb),\n",
    "        'y_prob': y_prob_lgb\n",
    "    }\n",
    "except Exception as e:\n",
    "    print(f\"LightGBM import failed: {e}. Using HistGradientBoostingClassifier instead.\")\n",
    "    from sklearn.ensemble import HistGradientBoostingClassifier\n",
    "    lgb_model = Pipeline(steps=[('preprocessor', preprocessor),\n",
    "                                ('classifier', HistGradientBoostingClassifier(random_state=42))])\n",
    "    lgb_model.fit(X_train, y_train)\n",
    "    y_pred_lgb = lgb_model.predict(X_test)\n",
    "    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]\n",
    "    results['HistGradientBoosting'] = {\n",
    "        'Accuracy': accuracy_score(y_test, y_pred_lgb),\n",
    "        'AUC': roc_auc_score(y_test, y_prob_lgb),\n",
    "        'y_prob': y_prob_lgb\n",
    "    }"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.1 Performance Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\nResults:\")\n",
    "for name, metrics in results.items():\n",
    "    print(f\"{name}: Accuracy={metrics['Accuracy']:.4f}, AUC={metrics['AUC']:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 ROC Curves"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot ROC Curves\n",
    "plt.figure(figsize=(10, 8))\n",
    "for name, metrics in results.items():\n",
    "    fpr, tpr, _ = roc_curve(y_test, metrics['y_prob'])\n",
    "    plt.plot(fpr, tpr, label=f\"{name} (AUC = {metrics['AUC']:.2f})\")\n",
    "plt.plot([0, 1], [0, 1], 'k--')\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC Curves')\n",
    "plt.legend()\n",
    "plt.savefig('roc_curves.png')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Importance\n",
    "# We try to get feature importance from the best available model\n",
    "print(\"Generating Feature Importance plot...\")\n",
    "model_for_importance = None\n",
    "model_name = \"\"\n",
    "\n",
    "if hasattr(lgb_model.named_steps['classifier'], 'feature_importances_'):\n",
    "    model_for_importance = lgb_model\n",
    "    model_name = \"LightGBM\"\n",
    "elif hasattr(xgb_model.named_steps['classifier'], 'feature_importances_'):\n",
    "    model_for_importance = xgb_model\n",
    "    model_name = \"XGBoost/GradientBoosting\"\n",
    "else:\n",
    "    print(\"No tree-based model with feature_importances_ available.\")\n",
    "\n",
    "if model_for_importance:\n",
    "    # Extract feature names after one-hot encoding\n",
    "    feature_names = numerical_features + \\\n",
    "        list(model_for_importance.named_steps['preprocessor'].named_transformers_['cat']\n",
    "             .named_steps['onehot'].get_feature_names_out(categorical_features))\n",
    "\n",
    "    clf = model_for_importance.named_steps['classifier']\n",
    "    importances = clf.feature_importances_\n",
    "    indices = np.argsort(importances)[::-1]\n",
    "\n",
    "    # Plot Top 20 Features\n",
    "    plt.figure(figsize=(12, 10))\n",
    "    plt.title(f\"Feature Importances ({model_name})\")\n",
    "    plt.barh(range(min(20, len(indices))), importances[indices[:20]], align=\"center\")\n",
    "    plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])\n",
    "    plt.gca().invert_yaxis()\n",
    "    plt.tight_layout()\n",
    "    plt.savefig('feature_importance.png')\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('churn_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
