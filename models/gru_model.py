"""
LSTM/GRU Model for Churn Prediction
"""
import torch
import torch.nn as nn


class GRUChurnModel(nn.Module):
    """GRU-based model for predicting churn from sequential usage data."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(GRUChurnModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 16)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # GRU output
        gru_out, hidden = self.gru(x)
        
        # Take the last timestep output
        last_output = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze()
