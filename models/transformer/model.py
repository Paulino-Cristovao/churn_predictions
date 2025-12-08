"""
Transformer Encoder Model for Churn Prediction
Multi-head self-attention for capturing temporal patterns
"""
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Adds positional information to sequence embeddings
    Uses sin/cos functions at different frequencies
    """
    def __init__(self, d_model, max_len=15):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)[:, :d_model//2]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        return x + self.pe[:, :x.size(1), :]

class TransformerChurnModel(nn.Module):
    """
    Transformer Encoder for churn prediction
    Uses multi-head self-attention to capture dependencies across trial days
    """
    def __init__(self, input_dim, d_model=64, num_heads=4, ff_dim=128, 
                 num_layers=2, dropout=0.2):
        super().__init__()
        
        # Project input to d_model dimension (must be divisible by num_heads)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for temporal order
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # Use projected dimension
            nhead=num_heads,
            dim_feedforward=ff_dim, 
            dropout=dropout,
            activation='relu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch, seq_len=15, features=19)
        
        # Project to d_model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding with self-attention
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling over sequence
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # Classification head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x.squeeze()
