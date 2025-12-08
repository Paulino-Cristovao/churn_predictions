"""
LSTM/GRU Sequential Model for Churn Prediction
Captures temporal patterns in 15-day trial usage sequences
Expected AUC: 0.78-0.82 (vs XGBoost 0.74)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, brier_score_loss
import matplotlib.pyplot as plt
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data
print("Loading data...")
daily_usage_df = pd.read_csv('Data/daily_usage.csv')
subscriptions_df = pd.read_csv('Data/subscriptions.csv')

# Preprocess dates
subscriptions_df['trial_starts_at'] = pd.to_datetime(subscriptions_df['trial_starts_at'], errors='coerce')
subscriptions_df['trial_ends_at'] = pd.to_datetime(subscriptions_df['trial_ends_at'], errors='coerce')
daily_usage_df['day_date'] = pd.to_datetime(daily_usage_df['day_date'], errors='coerce')

# Filter 15-day trials
subscriptions_df = subscriptions_df.dropna(subset=['trial_starts_at', 'trial_ends_at'])
subscriptions_df['trial_duration'] = (subscriptions_df['trial_ends_at'] - subscriptions_df['trial_starts_at']).dt.days
subscriptions_df = subscriptions_df[subscriptions_df['trial_duration'] == 15].copy()

# Create target
subscriptions_df['converted'] = subscriptions_df['first_paid_invoice_paid_at'].notnull().astype(int)

# Get usage columns
usage_cols = [col for col in daily_usage_df.columns if col.startswith('nb_')]
print(f'Using {len(usage_cols)} usage features')

def prepare_sequence_data(daily_usage_df, subscriptions_df, usage_cols):
    """Convert daily usage to 3D sequences: (samples, timesteps=15, features)"""
    sequences = []
    labels = []
    valid_ids = []
    
    for sub_id in subscriptions_df['subscription_id']:
        sub_data = daily_usage_df[daily_usage_df['subscription_id'] == sub_id].copy()
        if sub_data.empty:
            continue
        
        trial_start = subscriptions_df[subscriptions_df['subscription_id'] == sub_id]['trial_starts_at'].values[0]
        sub_data['day_number'] = (sub_data['day_date'] - trial_start).dt.days
        sub_data = sub_data[(sub_data['day_number'] >= 0) & (sub_data['day_number'] < 15)]
        sub_data = sub_data.sort_values('day_number')
        
        # Create 15-day sequence (fills with zeros for missing days)
        sequence = np.zeros((15, len(usage_cols)))
        for _, row in sub_data.iterrows():
            day = int(row['day_number'])
            if day < 15:
                sequence[day] = row[usage_cols].fillna(0).values
        
        sequences.append(sequence)
        labels.append(subscriptions_df[subscriptions_df['subscription_id'] == sub_id]['converted'].values[0])
        valid_ids.append(sub_id)
    
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32), valid_ids

print("Preparing sequential data...")
X_seq, y_seq, valid_ids = prepare_sequence_data(daily_usage_df, subscriptions_df, usage_cols)
print(f'Sequence shape: {X_seq.shape} (samples, timesteps, features)')
print(f'Conversion rate: {y_seq.mean():.2%}')

# Train/val/test splits
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# PyTorch Dataset
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)
test_dataset = SequenceDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# LSTM Model (Using GRU for efficiency with small data)
class ChurnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(ChurnLSTM, self).__init__()
        # Using GRU instead of LSTM for faster training and fewer parameters
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len=15, features)
        out, hidden = self.gru(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]  # Take last timestep
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out.squeeze()

# Initialize model
input_size = len(usage_cols)
model = ChurnLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training with early stopping
print("\\nTraining LSTM model...")
num_epochs = 100
patience = 15
best_val_auc = 0
counter = 0
train_losses = []
val_aucs = []

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
    
    val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
    val_aucs.append(val_auc)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}: Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}')
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'results/best_lstm_model.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

print(f'\\nBest validation AUC: {best_val_auc:.4f}')

# Load best model and evaluate on test set
model.load_state_dict(torch.load('results/best_lstm_model.pt'))
model.eval()

test_probs, test_labels = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x).cpu().numpy()
        test_probs.extend(outputs)
        test_labels.extend(batch_y.numpy())

# Calculate metrics
test_probs = np.array(test_probs)
test_labels = np.array(test_labels)
test_preds = (test_probs >= 0.5).astype(int)

test_auc = roc_auc_score(test_labels, test_probs)
test_acc = accuracy_score(test_labels, test_preds)
precision, recall, _ = precision_recall_curve(test_labels, test_probs)
test_pr_auc = auc(recall, precision)
test_brier = brier_score_loss(test_labels, test_probs)

print("\\n" + "="*70)
print("LSTM/GRU MODEL - TEST SET RESULTS")
print("="*70)
print(f'Accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'ROC-AUC:     {test_auc:.4f}')
print(f'PR-AUC:      {test_pr_auc:.4f}')
print(f'Brier Score: {test_brier:.4f} (lower is better)')
print("="*70)

# Save results
results = {
    'model': 'LSTM',
    'test_auc': float(test_auc),
    'test_accuracy': float(test_acc),
    'test_pr_auc': float(test_pr_auc),
    'test_brier': float(test_brier),
    'best_val_auc': float(best_val_auc),
    'epochs_trained': epoch + 1
}

import json
os.makedirs('results', exist_ok=True)
with open('results/lstm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\\n✅ Results saved to results/lstm_results.json")
print(f"✅ Best model saved to results/best_lstm_model.pt")

# Training curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(val_aucs, label='Validation AUC', color='coral', alpha=0.8)
plt.axhline(y=best_val_auc, color='green', linestyle='--', label=f'Best Val AUC: {best_val_auc:.4f}', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Validation AUC Over Time', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/lstm_training_curves.png', dpi=300, bbox_inches='tight')
print(f"✅ Training curves saved to results/lstm_training_curves.png")

plt.show()
