"""
GRU Churn Prediction Model - Simple and Effective
Optimized for small datasets (415 samples)
"""
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    Simple GRU model for churn prediction
    More efficient than LSTM, less prone to overfitting on small data
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len=15, features)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out.squeeze()
