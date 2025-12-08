"""
Configuration and hyperparameters for churn prediction models
"""

# Model hyperparameters
MODEL_CONFIG = {
    'gru': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10
    },
    
    'transformer': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10
    }
}

# Data configuration
DATA_CONFIG = {
    'sequence_length': 15,  # 15-day trial period
    'val_split': 0.2,
    'random_seed': 42
}

# Training configuration
TRAINING_CONFIG = {
    'device': 'cuda' if __name__ != '__main__' else 'cpu',  # Will be set in notebook
    'save_dir': '../results/models/',
    'metrics_dir': '../results/metrics/',
    'figures_dir': '../results/figures/'
}
