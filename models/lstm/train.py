"""
LSTM Model Training Script
Handles data preparation, training loop, and evaluation
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
import json
import os

class SequenceDataset(Dataset):
    """PyTorch Dataset for sequential churn data"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_sequence_data(daily_usage_df, subscriptions_df, usage_cols):
    """
    Convert daily usage to 3D sequences
    Returns: (samples, timesteps=15, features)
    """
    sequences = []
    labels = []
    
    for sub_id in subscriptions_df['subscription_id']:
        sub_data = daily_usage_df[daily_usage_df['subscription_id'] == sub_id].copy()
        if sub_data.empty:
            continue
        
        trial_start = subscriptions_df[subscriptions_df['subscription_id'] == sub_id]['trial_starts_at'].values[0]
        sub_data['day_number'] = (sub_data['day_date'] - trial_start).dt.days
        sub_data = sub_data[(sub_data['day_number'] >= 0) & (sub_data['day_number'] < 15)]
        sub_data = sub_data.sort_values('day_number')
        
        # Create 15-day sequence
        sequence = np.zeros((15, len(usage_cols)))
        for _, row in sub_data.iterrows():
            day = int(row['day_number'])
            if day < 15:
                sequence[day] = row[usage_cols].fillna(0).values.astype(float)
        
        sequences.append(sequence)
        labels.append(subscriptions_df[subscriptions_df['subscription_id'] == sub_id]['converted'].values[0])
    
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

def train_lstm_model(
    X_train, y_train, X_val, y_val,
    input_size,
    hidden_size=64,      # GRU default
    num_layers=2,        # GRU default
    dropout=0.3,         # GRU default
    batch_size=32,       # Standard batch size
    learning_rate=0.05,  # Optimal from LR search (tested 0.01, 0.001, 0.05, 0.005)
    num_epochs=100,      # Increased for better convergence
    patience=10,         # Early stopping patience
    device='cpu',
    save_dir='models/lstm'
):
    """
    Train GRU model with optimized hyperparameters for small datasets
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize simpler GRU model
    from models.lstm.model import GRUModel
    model = GRUModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer (Adamax for sparse data)
    criterion = nn.BCELoss()
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_auc = 0
    counter = 0
    train_losses = []
    val_aucs = []
    learning_rates = []
    
    print("\\nStarting training...")
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
            
            # Gradient clipping for stability
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
        
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        val_aucs.append(val_auc)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d}/{num_epochs}: Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}')
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'{save_dir}/best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_aucs': val_aucs,
        'learning_rates': learning_rates,
        'best_val_auc': float(best_val_auc),
        'epochs_trained': epoch + 1
    }
    
    return model, history

def evaluate_model(model, X_test, y_test, device='cpu', save_dir='models/lstm'):
    """Evaluate model on test set"""
    test_dataset = SequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).cpu().numpy()
            test_probs.extend(outputs)
            test_labels.extend(batch_y.numpy())
    
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_preds = (test_probs >= 0.5).astype(int)
    
    # Metrics
    test_auc = roc_auc_score(test_labels, test_probs)
    test_acc = accuracy_score(test_labels, test_preds)
    precision, recall, _ = precision_recall_curve(test_labels, test_probs)
    test_pr_auc = auc(recall, precision)
    test_brier = brier_score_loss(test_labels, test_probs)
    
    results = {
        'test_auc': float(test_auc),
        'test_accuracy': float(test_acc),
        'test_pr_auc': float(test_pr_auc),
        'test_brier': float(test_brier)
    }
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_training_curves(history, save_dir='models/lstm'):
    """Generate training visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss curve
    axes[0, 0].plot(epochs, history['train_losses'], label='Train Loss', color='steelblue', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontweight='bold')
    axes[0, 0].set_title('Training Loss', fontweight='bold', fontsize=13)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # AUC curve
    axes[0, 1].plot(epochs, history['val_aucs'], label='Validation AUC', color='coral', linewidth=2)
    axes[0, 1].axhline(y=history['best_val_auc'], color='green', linestyle='--', 
                       label=f"Best: {history['best_val_auc']:.4f}", alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontweight='bold')
    axes[0, 1].set_ylabel('AUC', fontweight='bold')
    axes[0, 1].set_title('Validation AUC', fontweight='bold', fontsize=13)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['learning_rates'], label='Learning Rate', color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontweight='bold')
    axes[1, 0].set_ylabel('Learning Rate', fontweight='bold')
    axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold', fontsize=13)
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Combined view
    ax2 = axes[1, 1]
    ax2.plot(epochs, history['val_aucs'], label='Val AUC', color='coral', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('AUC', fontweight='bold', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax2.set_title('AUC vs Loss', fontweight='bold', fontsize=13)
    ax2.grid(alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, history['train_losses'], label='Train Loss', color='steelblue', linewidth=2, alpha=0.7)
    ax2_twin.set_ylabel('Loss', fontweight='bold', color='steelblue')
    ax2_twin.tick_params(axis='y', labelcolor='steelblue')
    
    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Add overall title with training info
    fig.suptitle(f'GRU Training Progress - {history["epochs_trained"]} Epochs (Early Stopped)', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to {save_dir}/training_curves.png")
    
    return fig
