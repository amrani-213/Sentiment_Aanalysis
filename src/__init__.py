"""
Sentiment Analysis Deep Learning Project
Main package initializer
"""
# Data modules
from .data.preprocessing import EnhancedTextPreprocessor, prepare_data
from .data.augmentation import TextAugmenter, augment_training_data
from .data.dataset import (
    SentimentDataset,
    TransformerDataset,
    create_data_loaders,
    create_transformer_data_loaders,
    get_class_weights
)

# Model modules - Baseline
from .models.baseline.bilstm_attention import create_bilstm_model
from .models.baseline.custom_transformer import create_custom_transformer
from .models.baseline.fasttext import create_fasttext_model

# Model modules - Pretrained
from .models.pretrained.roberta import create_roberta_model, get_roberta_tokenizer
from .models.pretrained.bertweet import create_bertweet_model, get_bertweet_tokenizer

# Model modules - Ensemble
from .models.ensemble.voting_ensemble import (
    VotingEnsemble,
    create_voting_ensemble,
    create_weighted_ensemble
)
from .models.ensemble.stacking_ensemble import (
    StackingEnsemble,
    create_stacking_ensemble
)

# Training modules
from .training.trainer import Trainer
from .training.metrics import calculate_metrics, print_metrics
from .training.losses import (
    FocalLoss,
    LabelSmoothingLoss,
    WeightedCrossEntropyLoss,
    CombinedLoss,
    get_loss_function,
    compute_class_weights
)

# Evaluation modules
from .evaluation.evaluator import ModelEvaluator, EnsembleEvaluator
from .evaluation.error_analysis import ErrorAnalyzer
from .evaluation.visualizer import SentimentVisualizer

# Utility modules
from .utils.config import set_seed, get_device, load_config, save_config
from .utils.logger import setup_logger
from .utils.helpers import (
    count_parameters,
    save_model,
    load_model,
    EarlyStopping,
    format_time
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Ayoub Asri'
__description__ = 'Sentiment Analysis with Deep Learning - ENSSEA Project 2026'

# Public API
__all__ = [
    # Data
    'EnhancedTextPreprocessor',
    'prepare_data',
    'TextAugmenter',
    'augment_training_data',
    'SentimentDataset',
    'TransformerDataset',
    'create_data_loaders',
    'create_transformer_data_loaders',
    'get_class_weights',
    
    # Baseline Models
    'create_bilstm_model',
    'create_custom_transformer',
    'create_fasttext_model',
    
    # Pretrained Models
    'create_roberta_model',
    'get_roberta_tokenizer',
    'create_bertweet_model',
    'get_bertweet_tokenizer',
    
    # Ensemble Models
    'VotingEnsemble',
    'create_voting_ensemble',
    'create_weighted_ensemble',
    'StackingEnsemble',
    'create_stacking_ensemble',
    
    # Training
    'Trainer',
    'calculate_metrics',
    'print_metrics',
    'FocalLoss',
    'LabelSmoothingLoss',
    'WeightedCrossEntropyLoss',
    'CombinedLoss',
    'get_loss_function',
    'compute_class_weights',
    
    # Evaluation
    'ModelEvaluator',
    'EnsembleEvaluator',
    'ErrorAnalyzer',
    'SentimentVisualizer',
    
    # Utilities
    'set_seed',
    'get_device',
    'load_config',
    'save_config',
    'setup_logger',
    'count_parameters',
    'save_model',
    'load_model',
    'EarlyStopping',
    'format_time',
]