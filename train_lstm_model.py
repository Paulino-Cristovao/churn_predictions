"""
Run LSTM Model Training with Improved Architecture
"""
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')

from models.lstm.train import (
    prepare_sequence_data, 
    train_lstm_model, 
    evaluate_model,
    plot_training_curves
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Using device: {device}')

# Load data
print("📂 Loading data...")
daily_usage_df = pd.read_csv('Data/daily_usage.csv')
subscriptions_df = pd.read_csv('Data/subscriptions.csv')

# Preprocess
subscriptions_df['trial_starts_at'] = pd.to_datetime(subscriptions_df['trial_starts_at'], errors='coerce')
subscriptions_df['trial_ends_at'] = pd.to_datetime(subscriptions_df['trial_ends_at'], errors='coerce')
daily_usage_df['day_date'] = pd.to_datetime(daily_usage_df['day_date'], errors='coerce')

subscriptions_df = subscriptions_df.dropna(subset=['trial_starts_at', 'trial_ends_at'])
subscriptions_df['trial_duration'] = (subscriptions_df['trial_ends_at'] - subscriptions_df['trial_starts_at']).dt.days
subscriptions_df = subscriptions_df[subscriptions_df['trial_duration'] == 15].copy()
subscriptions_df['converted'] = subscriptions_df['first_paid_invoice_paid_at'].notnull().astype(int)

usage_cols = [col for col in daily_usage_df.columns if col.startswith('nb_')]
print(f'📊 Using {len(usage_cols)} usage features')

# Prepare sequences
print("🔄 Preparing sequential data...")
X_seq, y_seq = prepare_sequence_data(daily_usage_df, subscriptions_df, usage_cols)
print(f'Sequence shape: {X_seq.shape}')
print(f'Conversion rate: {y_seq.mean():.2%}')

# Split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f'Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}')

# Train with improved hyperparameters
print("\\n🚀 Training improved LSTM model...")
print("="*70)
model, history = train_lstm_model(
    X_train, y_train, X_val, y_val,
    input_size=len(usage_cols),
    hidden_size=128,  # Increased from 64
    num_layers=3,     # Increased from 2
    dropout=0.4,      # Increased regularization
    batch_size=16,    # Smaller batches
    learning_rate=0.0005,  # Lower LR
    num_epochs=200,   # More epochs
    patience=25,      # More patience
    device=device,
    save_dir='models/lstm'
)

# Evaluate
print("\\n📊 Evaluating on test set...")
print("="*70)
# Load best model
model.load_state_dict(torch.load('models/lstm/best_model.pt'))
results = evaluate_model(model, X_test, y_test, device=device, save_dir='models/lstm')

print("\\nIMPROVED LSTM/GRU - TEST RESULTS")
print("="*70)
print(f"Accuracy:    {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
print(f"ROC-AUC:     {results['test_auc']:.4f}")
print(f"PR-AUC:      {results['test_pr_auc']:.4f}")
print(f"Brier Score: {results['test_brier']:.4f}")
print(f"Best Val AUC: {history['best_val_auc']:.4f}")
print(f"Epochs: {history['epochs_trained']}")
print("="*70)

# Plot
plot_training_curves(history, save_dir='models/lstm')

print("\\n✅ Training complete!")
print(f"✅ Model saved: models/lstm/best_model.pt")
print(f"✅ Results saved: models/lstm/results.json")
print(f"✅ Plots saved: models/lstm/training_curves.png")
