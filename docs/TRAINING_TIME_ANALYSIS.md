# Training Time Comparison

From our training runs, here are the approximate training times for each model:

| Model | Training Time | Parameters | Epochs | Notes |
|-------|---------------|------------|--------|-------|
| **Logistic Regression** | <1 min | ~400 | N/A | Sklearn implementation, very fast |
| **XGBoost** | ~2-3 min | ~500 trees | ~180-220 | Early stopping, CPU-optimized |
| **LightGBM** | ~1.5-2 min | ~500 trees | ~150-200 | Faster than XGBoost, histogram-based |
| **GRU** | ~5-7 min | 42,337 | 17 (max 100) | PyTorch CPU, early stopped |
| **Transformer** | ~6-8 min | 70,337 | 16 (max 100) | PyTorch CPU, self-attention overhead |

## Key Observations

### Speed vs Performance Trade-off
- **Fastest**: LightGBM (1.5-2 min) - 72.5% AUC
- **Best Performance**: XGBoost (2-3 min) - 73.7% AUC  
- **Best PR-AUC**: GRU (5-7 min) - 79.76% PR-AUC

### Training Efficiency
1. **Tree Models** (1-3 min):
   - Train directly on engineered features
   - Parallel tree construction
   - Early stopping prevents overtraining

2. **Deep Learning** (5-8 min):
   - Sequential epoch-based training
   - Gradient descent optimization
   - Longer but learns temporal patterns
   - CPU-only (no GPU acceleration)

### Production Considerations
- **Retraining frequency**: Monthly recommended
- **XGBoost**: Best balance of speed (2-3 min) and performance (73.7% AUC)
- **GRU**: Worth extra time for precision-recall tasks (79.76% PR-AUC)
- **With GPU**: Deep learning models would train 5-10x faster

### Hyperparameter Search Time
- **GRU LR Search**: ~20 min (4 LRs × 50 epochs each)
- **Transformer LR Search**: ~25 min (4 LRs × 50 epochs each)
- **Total optimization time**: ~45 min one-time cost

---

**Recommendation**: For production, use **XGBoost** (best speed/performance balance). Deploy **GRU** for precision-critical campaigns where the extra 3-5 min training time is justified by superior PR-AUC.
