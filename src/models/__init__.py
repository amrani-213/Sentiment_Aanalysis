"""
Model module for sentiment analysis
Comprehensive collection of baseline, pretrained, and ensemble architectures
"""
# Baseline models
from .baseline.bilstm_attention import (
    BiLSTMMultiHeadAttention,
    create_bilstm_model
)
from .baseline.custom_transformer import (
    CustomTransformer,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    TransformerEncoderLayer,
    create_custom_transformer
)
from .baseline.fasttext import (
    FastTextModel,
    CharNGramExtractor,
    create_fasttext_model
)

# Pretrained transformer models
from .pretrained.roberta import (
    RoBERTaClassifier,
    create_roberta_model,
    get_roberta_tokenizer,
    tokenize_for_roberta
)
from .pretrained.bertweet import (
    BERTweetClassifier,
    create_bertweet_model,
    get_bertweet_tokenizer,
    tokenize_for_bertweet
)

# Ensemble models
from .ensemble.voting_ensemble import (
    VotingEnsemble,
    WeightedVotingEnsemble,
    create_voting_ensemble,
    create_weighted_ensemble
)
from .ensemble.stacking_ensemble import (
    StackingEnsemble,
    create_stacking_ensemble
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Amrani Bouabdellah'
__description__ = 'Sentiment Analysis Models - ENSSEA Project 2026'

# Public API - defines what gets imported with `from src.models import *`
__all__ = [
    # Baseline Models
    'BiLSTMMultiHeadAttention',
    'create_bilstm_model',
    'CustomTransformer',
    'PositionalEncoding',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerEncoderLayer',
    'create_custom_transformer',
    'FastTextModel',
    'CharNGramExtractor',
    'create_fasttext_model',
    
    # Pretrained Models
    'RoBERTaClassifier',
    'create_roberta_model',
    'get_roberta_tokenizer',
    'tokenize_for_roberta',
    'BERTweetClassifier',
    'create_bertweet_model',
    'get_bertweet_tokenizer',
    'tokenize_for_bertweet',
    
    # Ensemble Models
    'VotingEnsemble',
    'WeightedVotingEnsemble',
    'create_voting_ensemble',
    'create_weighted_ensemble',
    'StackingEnsemble',
    'create_stacking_ensemble',
]