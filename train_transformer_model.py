"""
Train Transformer Model for Churn Prediction
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, brier_score_loss
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.append('.')

from models.transformer.model import TransformerChurnModel
from models.lstm.train import SequenceDataset, prepare_sequence_data

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

# Create datasets
train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)
test_dataset = SequenceDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Transformer model
model = TransformerChurnModel(
    input_dim=len(usage_cols),
    d_model=64,      # Projected dimension (divisible by num_heads=4)
    num_heads=4,
    ff_dim=128,
    num_layers=2,
    dropout=0.2
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)  # Optimal from LR search

# Training loop
num_epochs = 100  # Increased for better convergence
patience = 15     # Increased patience
best_val_auc = 0
counter = 0
train_losses = []
val_aucs = []

print("\n🚀 Training Transformer model...")
print("="*70)

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).cpu().numpy()
            val_probs.extend(outputs)
            val_labels.extend(batch_y.numpy())
    
    val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
    val_aucs.append(val_auc)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:3d}/{num_epochs}: Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}')
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'models/transformer/best_model.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

print(f'\nBest validation AUC: {best_val_auc:.4f}')

# Load best model and evaluate
model.load_state_dict(torch.load('models/transformer/best_model.pt'))
model.eval()

test_probs, test_labels = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x).cpu().numpy()
        test_probs.extend(outputs)
        test_labels.extend(batch_y.numpy())

test_probs = torch.tensor(test_probs).numpy()
test_labels = torch.tensor(test_labels).numpy()
test_preds = (test_probs >= 0.5).astype(int)

# Calculate metrics
test_auc = roc_auc_score(test_labels, test_probs)
test_acc = accuracy_score(test_labels, test_preds)
precision, recall, _ = precision_recall_curve(test_labels, test_probs)
test_pr_auc = auc(recall, precision)
test_brier = brier_score_loss(test_labels, test_probs)

print("\n" + "="*70)
print("TRANSFORMER MODEL - TEST RESULTS")
print("="*70)
print(f'Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'ROC-AUC:     {test_auc:.4f}')
print(f'PR-AUC:      {test_pr_auc:.4f}')
print(f'Brier Score: {test_brier:.4f}')
print(f'Best Val AUC: {best_val_auc:.4f}')
print(f'Epochs: {epoch + 1}')
print("="*70)

# Save results
results = {
    'test_auc': float(test_auc),
    'test_accuracy': float(test_acc),
    'test_pr_auc': float(test_pr_auc),
    'test_brier': float(test_brier),
    'best_val_auc': float(best_val_auc),
    'epochs_trained': epoch + 1
}

os.makedirs('models/transformer', exist_ok=True)
with open('models/transformer/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to models/transformer/results.json")
print(f"✅ Best model saved to models/transformer/best_model.pt")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, len(train_losses) + 1)

ax1.plot(epochs, train_losses, label='Train Loss', color='steelblue', linewidth=2)
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Loss', fontweight='bold')
ax1.set_title('Training Loss', fontweight='bold', fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs, val_aucs, label='Validation AUC', color='coral', linewidth=2)
ax2.axhline(y=best_val_auc, color='green', linestyle='--', 
           label=f"Best: {best_val_auc:.4f}", alpha=0.7, linewidth=2)
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('AUC', fontweight='bold')
ax2.set_title('Validation AUC', fontweight='bold', fontsize=13)
ax2.legend()
ax2.grid(alpha=0.3)

fig.suptitle(f'Transformer Training Progress - {epoch+1} Epochs', 
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('models/transformer/training_curves.png', dpi=300, bbox_inches='tight')
print(f"✅ Training curves saved to models/transformer/training_curves.png")

print("\n✅ Training complete!")
