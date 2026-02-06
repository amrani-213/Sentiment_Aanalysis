"""
Training module for sentiment analysis
Comprehensive training utilities including trainers, losses, and metrics
"""
# Trainer components
from .trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
)

# Loss functions
from .losses import (
    FocalLoss,
    LabelSmoothingLoss,
    WeightedCrossEntropyLoss,
    CombinedLoss,
    compute_class_weights,
    get_loss_function
)

# Evaluation metrics
from .metrics import (
    calculate_metrics,
    print_metrics,
    generate_classification_report,
    calculate_top_k_accuracy,
    calculate_calibration_error,
    calculate_confidence_metrics,
    compare_models
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Amrani Bouabdellah'

# Public API - defines what gets imported with `from src.training import *`
__all__ = [
    # Trainer
    'Trainer',
    'create_optimizer',
    'create_scheduler',
    
    # Losses
    'FocalLoss',
    'LabelSmoothingLoss',
    'WeightedCrossEntropyLoss',
    'CombinedLoss',
    'compute_class_weights',
    'get_loss_function',
    
    # Metrics
    'calculate_metrics',
    'print_metrics',
    'generate_classification_report',
    'calculate_top_k_accuracy',
    'calculate_calibration_error',
    'calculate_confidence_metrics',
    'compare_models',
]

