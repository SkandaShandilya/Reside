# Training pipeline package
"""
Training Pipeline for Activity Recognition Models

Pipeline:
1. Load existing dataset + verified CCTV samples
2. Split into train/validation sets
3. Train model with configurable architecture
4. Evaluate on validation set
5. Compare with previous model version
6. Save model checkpoint (never overwrite)
7. Deploy if performance is acceptable

IMPORTANT:
- Never automatically treat AI predictions as ground truth
- Only verified (human-labeled) samples are used for training
- Every trained model gets a unique version number
- Previous models are never overwritten (rollback support)
"""
