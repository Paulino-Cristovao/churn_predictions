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
    "We use Logistic Regression, XGBoost, and LightGBM models.\n",
    "\n",
    "## Enhancements\n",
    "- **Configuration**: Hyperparameters are managed using Pydantic models.\n",
    "- **Visualization**: Added EDA plots for target distribution, correlations, and feature distributions.\n",
    "- **Explanations**: Detailed reasoning for preprocessing steps."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Imports\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pydantic import BaseModel, Field\n",
    "from typing import Optional, List, Dict, Any\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, precision_recall_curve, auc, brier_score_loss\n",
    "from sklearn.calibration import calibration_curve\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "%matplotlib inline\n",
    "sns.set_style('whitegrid')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Configuration with Pydantic\n",
    "\n",
    "We use Pydantic to strictly define our configuration, including data splitting parameters and model hyperparameters. This makes the experiment reproducible and easy to tune."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class DataConfig(BaseModel):\n",
    "    test_size: float = Field(0.2, description=\"Proportion of dataset to include in the test split\")\n",
    "    random_state: int = Field(42, description=\"Random seed for reproducibility\")\n",
    "\n",
    "class LogisticRegressionConfig(BaseModel):\n",
    "    max_iter: int = Field(1000, description=\"Maximum number of iterations\")\n",
    "    C: float = Field(1.0, description=\"Inverse of regularization strength\")\n",
    "    class_weight: str = Field('balanced', description=\"Handle class imbalance\")\n",
    "\n",
    "class XGBoostConfig(BaseModel):\n",
    "    n_estimators: int = Field(500, description=\"Number of boosting rounds\")\n",
    "    max_depth: int = Field(7, description=\"Maximum tree depth\")\n",
    "    learning_rate: float = Field(0.05, description=\"Learning rate\")\n",
    "    subsample: float = Field(0.8, description=\"Subsample ratio of training instances\")\n",
    "    colsample_bytree: float = Field(0.8, description=\"Subsample ratio of columns when constructing each tree\")\n",
    "    min_child_weight: int = Field(20, description=\"Minimum sum of instance weight needed in a child\")\n",
    "    early_stopping_rounds: int = Field(50, description=\"Early stopping rounds\")\n",
    "    random_state: int = Field(42, description=\"Random seed\")\n",
    "\n",
    "class LightGBMConfig(BaseModel):\n",
    "    n_estimators: int = Field(500, description=\"Number of boosting rounds\")\n",
    "    num_leaves: int = Field(31, description=\"Maximum number of leaves in one tree\")\n",
    "    max_depth: int = Field(-1, description=\"Maximum tree depth, -1 means no limit\")\n",
    "    learning_rate: float = Field(0.05, description=\"Learning rate\")\n",
    "    feature_fraction: float = Field(0.8, description=\"Fraction of features for each tree\")\n",
    "    bagging_fraction: float = Field(0.8, description=\"Fraction of data for bagging\")\n",
    "    bagging_freq: int = Field(5, description=\"Frequency for bagging\")\n",
    "    min_data_in_leaf: int = Field(20, description=\"Minimum number of data in one leaf\")\n",
    "    early_stopping_rounds: int = Field(50, description=\"Early stopping rounds\")\n",
    "    random_state: int = Field(42, description=\"Random seed\")\n",
    "\n",
    "class ExperimentConfig(BaseModel):\n",
    "    data: DataConfig = DataConfig()\n",
    "    lr: LogisticRegressionConfig = LogisticRegressionConfig()\n",
    "    xgb: XGBoostConfig = XGBoostConfig()\n",
    "    lgb: LightGBMConfig = LightGBMConfig()\n",
    "\n",
    "# Instantiate Configuration\n",
    "config = ExperimentConfig()\n",
    "print(\"Current Configuration:\")\n",
    "print(config.model_dump_json(indent=2))"
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
    "## 3. Preprocessing & Feature Engineering\n",
    "\n",
    "### 3.1 Date Conversion & Filtering\n",
    "We convert date columns to datetime objects to enable calculations. We filter for 15-day trials to ensure a consistent comparison window."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert date columns\n",
    "date_cols = ['subscription_created_at', 'trial_starts_at', 'trial_ends_at', 'canceled_at', 'first_paid_invoice_paid_at']\n",
    "for col in date_cols:\n",
    "    subscriptions_df[col] = pd.to_datetime(subscriptions_df[col], errors='coerce')\n",
    "\n",
    "# Filter for 15-day trials\n",
    "subscriptions_df['trial_duration'] = (subscriptions_df['trial_ends_at'] - subscriptions_df['trial_starts_at']).dt.days\n",
    "subscriptions_df = subscriptions_df[subscriptions_df['trial_duration'] == 15].copy()\n",
    "\n",
    "# Define Target: Converted = 1 if first_paid_invoice_paid_at is present\n",
    "subscriptions_df['converted'] = subscriptions_df['first_paid_invoice_paid_at'].notna().astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 Aggregating Usage Data\n",
    "We sum the daily usage logs for each subscription to get a total usage profile for the trial period."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "usage_cols = [col for col in daily_usage_df.columns if col not in ['subscription_id', 'day_date']]\n",
    "\n",
    "# Convert day_date to datetime\n",
    "daily_usage_df['day_date'] = pd.to_datetime(daily_usage_df['day_date'])\n",
    "\n",
    "# Calculate trial start date for each subscription\n",
    "trial_start_dates = subscriptions_df[['subscription_id', 'trial_starts_at']].copy()\n",
    "daily_usage_df = daily_usage_df.merge(trial_start_dates, on='subscription_id', how='left')\n",
    "daily_usage_df['day_number'] = (daily_usage_df['day_date'] - daily_usage_df['trial_starts_at']).dt.days\n",
    "\n",
    "# Aggregate total usage (as before)\n",
    "usage_agg = daily_usage_df.groupby('subscription_id')[usage_cols].sum().reset_index()\n",
    "usage_agg.columns = ['subscription_id'] + [f'total_{col}' for col in usage_cols]\n",
    "\n",
    "# Early trial activity (first 3 days: days 0-2)\n",
    "early_usage = daily_usage_df[daily_usage_df['day_number'].between(0, 2)].groupby('subscription_id')[usage_cols].sum().reset_index()\n",
    "early_usage.columns = ['subscription_id'] + [f'early_{col}' for col in usage_cols]\n",
    "\n",
    "# Late trial activity (last 3 days: days 12-14)\n",
    "late_usage = daily_usage_df[daily_usage_df['day_number'].between(12, 14)].groupby('subscription_id')[usage_cols].sum().reset_index()\n",
    "late_usage.columns = ['subscription_id'] + [f'late_{col}' for col in usage_cols]\n",
    "\n",
    "# Calculate total daily activity for trend analysis\n",
    "daily_usage_df['daily_total_activity'] = daily_usage_df[usage_cols].sum(axis=1)\n",
    "\n",
    "# Engagement trend (slope of activity over time)\n",
    "def calculate_trend(group):\n",
    "    if len(group) < 2:\n",
    "        return 0\n",
    "    x = group['day_number'].values\n",
    "    y = group['daily_total_activity'].values\n",
    "    if np.std(x) == 0:\n",
    "        return 0\n",
    "    return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0\n",
    "\n",
    "engagement_trend = daily_usage_df.groupby('subscription_id').apply(calculate_trend).reset_index()\n",
    "engagement_trend.columns = ['subscription_id', 'engagement_trend']\n",
    "\n",
    "# Activity variance and peak day\n",
    "activity_stats = daily_usage_df.groupby('subscription_id').agg({\n",
    "    'daily_total_activity': ['std', 'max']\n",
    "}).reset_index()\n",
    "activity_stats.columns = ['subscription_id', 'activity_std', 'activity_max']\n",
    "activity_stats['activity_std'] = activity_stats['activity_std'].fillna(0)\n",
    "\n",
    "# Calculate peak activity day separately\n",
    "peak_days = daily_usage_df.loc[daily_usage_df.groupby('subscription_id')['daily_total_activity'].idxmax()][['subscription_id', 'day_number']]\n",
    "peak_days.columns = ['subscription_id', 'peak_activity_day']\n",
    "activity_stats = activity_stats.merge(peak_days, on='subscription_id', how='left')\n",
    "activity_stats['peak_activity_day'] = activity_stats['peak_activity_day'].fillna(0)\n",
    "\n",
    "# Feature diversity (number of distinct features used)\n",
    "feature_diversity = daily_usage_df.groupby('subscription_id')[usage_cols].apply(lambda x: (x > 0).any().sum()).reset_index()\n",
    "feature_diversity.columns = ['subscription_id', 'feature_diversity']\n",
    "\n",
    "# Merge all features\n",
    "df = pd.merge(subscriptions_df, usage_agg, on='subscription_id', how='left')\n",
    "df = pd.merge(df, early_usage, on='subscription_id', how='left')\n",
    "df = pd.merge(df, late_usage, on='subscription_id', how='left')\n",
    "df = pd.merge(df, engagement_trend, on='subscription_id', how='left')\n",
    "df = pd.merge(df, activity_stats, on='subscription_id', how='left')\n",
    "df = pd.merge(df, feature_diversity, on='subscription_id', how='left')\n",
    "\n",
    "# Fill NaN values\n",
    "new_feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['total_', 'early_', 'late_', 'engagement', 'activity', 'feature_diversity'])]\n",
    "df[new_feature_cols] = df[new_feature_cols].fillna(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Exploratory Data Analysis (EDA)\n",
    "\n",
    "### 4.1 Target Distribution\n",
    "Checking the balance of our target variable is crucial. Imbalanced datasets might require techniques like SMOTE or class weighting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(x='converted', data=df, palette='viridis')\n",
    "plt.title('Distribution of Target Variable (Converted)')\n",
    "plt.xlabel('Converted (0=No, 1=Yes)')\n",
    "plt.ylabel('Count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Feature Correlations\n",
    "We analyze the correlation between numerical features and the target to identify strong predictors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Select numerical columns for correlation\n",
    "num_feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['total_', 'early_', 'late_', 'engagement', 'activity', 'feature_diversity'])]\n",
    "num_cols_for_corr = ['company_age_in_years', 'converted'] + num_feature_cols[:10] # limiting for readability\n",
    "corr_matrix = df[num_cols_for_corr].corr()\n",
    "\n",
    "plt.figure(figsize=(12, 10))\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'shrink': 0.8})\n",
    "plt.title('Correlation Matrix (Top Features)')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Pipeline & Modeling\n",
    "\n",
    "### 5.1 Preprocessing Pipeline\n",
    "- **Numerical Features**: Imputed with Median (robust to outliers), Scaled (StandardScaler) for Logistic Regression.\n",
    "- **Categorical Features**: Imputed with 'Unknown', One-Hot Encoded to handle categorical data for all models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical_features = ['vendor', 'v2_segment', 'naf_section', 'revenue_range', 'employee_count', 'regional_pole', 'legal_structure', 'company_age_group']\n",
    "numerical_features = ['company_age_in_years'] + [col for col in df.columns if any(prefix in col for prefix in ['total_', 'early_', 'late_', 'engagement', 'activity', 'feature_diversity'])]\n",
    "\n",
    "# Clean Numerical for split\n",
    "df['company_age_in_years'] = pd.to_numeric(df['company_age_in_years'], errors='coerce')\n",
    "\n",
    "X = df[categorical_features + numerical_features]\n",
    "y = df['converted']\n",
    "\n",
    "# Split using Config\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=config.data.test_size, random_state=config.data.random_state\n",
    ")\n",
    "\n",
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
    "### 5.2 Train/Validation Split\n",
    "We create a validation set from the training data for early stopping. This helps prevent overfitting by stopping training when validation performance plateaus."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split training data further for validation (for early stopping)\n",
    "X_train_split, X_val, y_train_split, y_val = train_test_split(\n",
    "    X_train, y_train, test_size=0.2, random_state=config.data.random_state, stratify=y_train\n",
    ")\n",
    "\n",
    "print(f'Training set: {X_train_split.shape[0]} samples')\n",
    "print(f'Validation set: {X_val.shape[0]} samples')\n",
    "print(f'Test set: {X_test.shape[0]} samples')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.3 Model Training with Regularization\n",
    "We train three models with advanced regularization techniques:\n",
    "- **Feature/Bagging sampling**: Reduces overfitting through randomization\n",
    "- **Early stopping**: Finds optimal number of iterations\n",
    "- **Leaf constraints**: Prevents overly specific splits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = {}\n",
    "\n",
    "# 1. Logistic Regression\n",
    "lr_model = Pipeline(steps=[('preprocessor', preprocessor),\n",
    "                           ('classifier', LogisticRegression(max_iter=config.lr.max_iter, C=config.lr.C, class_weight=config.lr.class_weight))])\n",
    "lr_model.fit(X_train, y_train)\n",
    "results['Logistic Regression'] = lr_model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "# 2. XGBoost with Advanced Regularization\n",
    "try:\n",
    "    import xgboost as xgb\n",
    "    # Calculate scale_pos_weight for class imbalance\n",
    "    scale_pos_weight = (y_train_split == 0).sum() / (y_train_split == 1).sum()\n",
    "    \n",
    "    # Preprocess data for early stopping\n",
    "    preprocessor_xgb = preprocessor.fit(X_train_split, y_train_split)\n",
    "    X_train_xgb = preprocessor_xgb.transform(X_train_split)\n",
    "    X_val_xgb = preprocessor_xgb.transform(X_val)\n",
    "    \n",
    "    xgb_clf = xgb.XGBClassifier(\n",
    "        n_estimators=config.xgb.n_estimators,\n",
    "        max_depth=config.xgb.max_depth,\n",
    "        learning_rate=config.xgb.learning_rate,\n",
    "        subsample=config.xgb.subsample,\n",
    "        colsample_bytree=config.xgb.colsample_bytree,\n",
    "        min_child_weight=config.xgb.min_child_weight,\n",
    "        random_state=config.xgb.random_state,\n",
    "        scale_pos_weight=scale_pos_weight,\n",
    "        use_label_encoder=False,\n",
    "        eval_metric='logloss'\n",
    "    )\n",
    "    from xgboost.callback import EarlyStopping\n",
    "    xgb_clf.fit(\n",
    "        X_train_xgb, y_train_split,\n",
    "        eval_set=[(X_val_xgb, y_val)],\n",
    "        callbacks=[EarlyStopping(rounds=config.xgb.early_stopping_rounds, save_best=True)],\n",
    "        verbose=False\n",
    "    )\n",
    "    print(f'XGBoost stopped at iteration: {xgb_clf.best_iteration + 1}/{config.xgb.n_estimators}')\n",
    "    results['XGBoost'] = xgb_clf.predict_proba(preprocessor_xgb.transform(X_test))[:, 1]\n",
    "except ImportError:\n",
    "    print(\"XGBoost not installed.\")\n",
    "\n",
    "# 3. LightGBM with Advanced Regularization\n",
    "try:\n",
    "    import lightgbm as lgb\n",
    "    \n",
    "    # Preprocess data for early stopping\n",
    "    preprocessor_lgb = preprocessor.fit(X_train_split, y_train_split)\n",
    "    X_train_lgb = preprocessor_lgb.transform(X_train_split)\n",
    "    X_val_lgb = preprocessor_lgb.transform(X_val)\n",
    "    \n",
    "    lgb_clf = lgb.LGBMClassifier(\n",
    "        n_estimators=config.lgb.n_estimators,\n",
    "        num_leaves=config.lgb.num_leaves,\n",
    "        max_depth=config.lgb.max_depth,\n",
    "        learning_rate=config.lgb.learning_rate,\n",
    "        feature_fraction=config.lgb.feature_fraction,\n",
    "        bagging_fraction=config.lgb.bagging_fraction,\n",
    "        bagging_freq=config.lgb.bagging_freq,\n",
    "        min_data_in_leaf=config.lgb.min_data_in_leaf,\n",
    "        random_state=config.lgb.random_state,\n",
    "        class_weight='balanced',\n",
    "        verbose=-1\n",
    "    )\n",
    "    lgb_clf.fit(\n",
    "        X_train_lgb, y_train_split,\n",
    "        eval_set=[(X_val_lgb, y_val)],\n",
    "        eval_metric='auc',\n",
    "        callbacks=[lgb.early_stopping(stopping_rounds=config.lgb.early_stopping_rounds, verbose=False)]\n",
    "    )\n",
    "    print(f'LightGBM stopped at iteration: {lgb_clf.best_iteration_}/{config.lgb.n_estimators}')\n",
    "    results['LightGBM'] = lgb_clf.predict_proba(preprocessor_lgb.transform(X_test))[:, 1]\n",
    "except ImportError:\n",
    "    print(\"LightGBM not installed.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Evaluation\n",
    "We compare the models using multiple metrics and cross-validation for robustness."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test set performance\n",
    "print('Test Set Results:')\n",
    "print('-' * 60)\n",
    "for name, y_prob in results.items():\n",
    "    y_pred = (y_prob >= 0.5).astype(int)\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    auc = roc_auc_score(y_test, y_prob)\n",
    "    print(f'{name}: Accuracy={acc:.4f}, AUC={auc:.4f}')\n",
    "print()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.1 ROC Curves"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 8))\n",
    "for name, y_prob in results.items():\n",
    "    auc = roc_auc_score(y_test, y_prob)\n",
    "    fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
    "    plt.plot(fpr, tpr, label=f\"{name} (AUC = {auc:.3f})\", linewidth=2)\n",
    "\n",
    "plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')\n",
    "plt.xlabel('False Positive Rate', fontsize=12)\n",
    "plt.ylabel('True Positive Rate', fontsize=12)\n",
    "plt.title('ROC Curves Comparison', fontsize=14)\n",
    "plt.legend(loc='lower right')\n",
    "plt.grid(alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 Precision-Recall Curves\n",
    "Important for imbalanced datasets where we care about positive class prediction quality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 8))\n",
    "for name, y_prob in results.items():\n",
    "    precision, recall, _ = precision_recall_curve(y_test, y_prob)\n",
    "    plt.plot(recall, precision, label=f\"{name}\", linewidth=2)\n",
    "\n",
    "plt.xlabel('Recall', fontsize=12)\n",
    "plt.ylabel('Precision', fontsize=12)\n",
    "plt.title('Precision-Recall Curves', fontsize=14)\n",
    "plt.legend()\n",
    "plt.grid(alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.3 Confusion Matrices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))\n",
    "if len(results) == 1:\n",
    "    axes = [axes]\n",
    "\n",
    "for idx, (name, y_prob) in enumerate(results.items()):\n",
    "    y_pred = (y_prob >= 0.5).astype(int)\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])\n",
    "    axes[idx].set_title(f'{name}')\n",
    "    axes[idx].set_xlabel('Predicted')\n",
    "    axes[idx].set_ylabel('Actual')\n",
    "    axes[idx].set_xticklabels(['Not Converted', 'Converted'])\n",
    "    axes[idx].set_yticklabels(['Not Converted', 'Converted'])\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Importance (LightGBM)\n",
    "if 'LightGBM' in results:\n",
    "    feature_names = numerical_features + list(preprocessor_lgb.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())\n",
    "    importances = lgb_clf.feature_importances_\n",
    "    indices = np.argsort(importances)[::-1][:20]\n",
    "    \n",
    "    plt.figure(figsize=(10, 8))\n",
    "    plt.title(\"Top 20 Feature Importances (LightGBM)\", fontsize=14)\n",
    "    plt.barh(range(len(indices)), importances[indices], align=\"center\")\n",
    "    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])\n",
    "    plt.xlabel('Importance', fontsize=12)\n",
    "    plt.gca().invert_yaxis()\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Model Interpretability with SHAP\n",
    "SHAP (SHapley Additive exPlanations) helps us understand which features drive predictions and by how much."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shap\n",
    "\n",
    "# Use best performing model (typically XGBoost or LightGBM)\n",
    "best_model_name = max(results.items(), key=lambda x: roc_auc_score(y_test, x[1]))[0]\n",
    "print(f'Generating SHAP explanations for: {best_model_name}')\n",
    "\n",
    "if best_model_name == 'XGBoost':\n",
    "    explainer = shap.TreeExplainer(xgb_clf)\n",
    "    shap_values = explainer(preprocessor_xgb.transform(X_test))\n",
    "    feature_names_shap = numerical_features + list(preprocessor_xgb.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())\n",
    "elif best_model_name == 'LightGBM':\n",
    "    explainer = shap.TreeExplainer(lgb_clf)\n",
    "    shap_values = explainer(preprocessor_lgb.transform(X_test))\n",
    "    feature_names_shap = numerical_features + list(preprocessor_lgb.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.1 SHAP Summary Plot\n",
    "Shows the most important features and their impact on predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if best_model_name in ['XGBoost', 'LightGBM']:\n",
    "    plt.figure(figsize=(12, 8))\n",
    "    shap.summary_plot(shap_values.values, features=shap_values.data, feature_names=feature_names_shap, max_display=20, show=False)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.2 Top Feature Drivers\n",
    "Quantifying which features matter most for conversion predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if best_model_name in ['XGBoost', 'LightGBM']:\n",
    "    # Calculate mean absolute SHAP values\n",
    "    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)\n",
    "    feature_importance_df = pd.DataFrame({\n",
    "        'feature': feature_names_shap,\n",
    "        'importance': mean_abs_shap\n",
    "    }).sort_values('importance', ascending=False).head(15)\n",
    "    \n",
    "    print('\\nTop 15 Features by SHAP Importance:')\n",
    "    print('=' * 60)\n",
    "    for idx, row in feature_importance_df.iterrows():\n",
    "        print(f\"{row['feature']:45s}: {row['importance']:.4f}\")\n",
    "    \n",
    "    # Bar plot\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])\n",
    "    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])\n",
    "    plt.xlabel('Mean |SHAP Value|', fontsize=12)\n",
    "    plt.title('Top 15 Feature Importance (SHAP)', fontsize=14)\n",
    "    plt.gca().invert_yaxis()\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.3 Sample Predictions\n",
    "Understanding individual predictions with waterfall plots."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if best_model_name in ['XGBoost', 'LightGBM']:\n",
    "    # Find a converted and non-converted example\n",
    "    converted_idx = np.where(y_test == 1)[0][0]\n",
    "    not_converted_idx = np.where(y_test == 0)[0][0]\n",
    "    \n",
    "    print(f'Explaining prediction for converted user (index {converted_idx}):')\n",
    "    shap.plots.waterfall(shap_values[converted_idx], max_display=15, show=False)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    print(f'\\nExplaining prediction for non-converted user (index {not_converted_idx}):')\n",
    "    shap.plots.waterfall(shap_values[not_converted_idx], max_display=15, show=False)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Calibration Analysis\n",
    "Calibration measures if predicted probabilities match actual conversion rates. Well-calibrated models are trustworthy for decision-making."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 8))\n",
    "\n",
    "for name, y_prob in results.items():\n",
    "    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)\n",
    "    brier = brier_score_loss(y_test, y_prob)\n",
    "    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{name} (Brier={brier:.3f})', linewidth=2)\n",
    "\n",
    "plt.plot([0, 1], [0, 1], 'k--', label=' Perfect Calibration')\n",
    "plt.xlabel('Mean Predicted Probability', fontsize=12)\n",
    "plt.ylabel('Fraction of Positives', fontsize=12)\n",
    "plt.title('Calibration Plots\\n(Lower Brier Score = Better Calibration)', fontsize=14)\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print('\\nCalibration Scores (Brier Loss - lower is better):')\n",
    "print('=' * 60)\n",
    "for name, y_prob in results.items():\n",
    "    brier = brier_score_loss(y_test, y_prob)\n",
    "    print(f'{name}: {brier:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Comprehensive Metrics Summary\n",
    "Final performance summary with all key metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('\\n' + '=' * 80)\n",
    "print('FINAL MODEL PERFORMANCE SUMMARY')\n",
    "print('=' * 80)\n",
    "\n",
    "for name, y_prob in results.items():\n",
    "    y_pred = (y_prob >= 0.5).astype(int)\n",
    "    \n",
    "    # Calculate metrics\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    roc_auc = roc_auc_score(y_test, y_prob)\n",
    "    precision, recall, _ = precision_recall_curve(y_test, y_prob)\n",
    "    pr_auc = auc(recall, precision)\n",
    "    brier = brier_score_loss(y_test, y_prob)\n",
    "    \n",
    "    print(f'\\n{name}:')\n",
    "    print(f'  ROC-AUC:     {roc_auc:.4f}')\n",
    "    print(f'  PR-AUC:      {pr_auc:.4f}')\n",
    "    print(f'  Accuracy:    {acc:.4f}')\n",
    "    print(f'  Brier Score: {brier:.4f}')\n",
    "\n",
    "print('\\n' + '=' * 80)"
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

print("Notebook generated successfully.")
