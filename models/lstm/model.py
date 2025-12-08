"""
LSTM/GRU Churn Prediction Model - Improved Architecture
Bidirectional LSTM with Attention for Sequential Pattern Learning
"""
import torch
import torch.nn as nn

class ImprovedChurnLSTM(nn.Module):
    """
    Improved Bidirectional LSTM with attention mechanism
    for churn prediction on 15-day trial sequences
    """
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.4):
        super(ImprovedChurnLSTM, self).__init__()
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Captures both forward and backward patterns
        )
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def attention_net(self, lstm_output):
        """
        Attention mechanism to focus on important timesteps
        """
        # lstm_output: (batch, seq_len, hidden*2)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: (batch, seq_len, 1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context: (batch, hidden*2)
        return context, attention_weights
    
    def forward(self, x):
        # x shape: (batch, seq_len=15, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden*2)
        
        # Apply attention
        context, attention_weights = self.attention_net(lstm_out)
        # context: (batch, hidden*2)
        
        # Batch normalization
        out = self.bn1(context)
        out = self.dropout(out)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.bn2(out)
        out = self.dropout(out)
        
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        
        out = self.sigmoid(self.fc3(out))
        return out.squeeze()
