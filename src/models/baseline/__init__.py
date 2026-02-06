"""
Baseline models for sentiment analysis
Traditional deep learning architectures without pretrained transformers
"""
# BiLSTM with Multi-Head Attention
from .bilstm_attention import (
    BiLSTMMultiHeadAttention,
    create_bilstm_model
)

# Custom Transformer (from scratch)
from .custom_transformer import (
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    TransformerEncoderLayer,
    CustomTransformer,
    create_custom_transformer
)

# FastText model
from .fasttext import (
    CharNGramExtractor,
    FastTextModel,
    create_fasttext_model
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Amrani Bouabdellah'

# Public API - defines what gets imported with `from src.models.baseline import *`
__all__ = [
    # BiLSTM
    'BiLSTMMultiHeadAttention',
    'create_bilstm_model',
    
    # Custom Transformer
    'PositionalEncoding',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerEncoderLayer',
    'CustomTransformer',
    'create_custom_transformer',
    
    # FastText
    'CharNGramExtractor',
    'FastTextModel',
    'create_fasttext_model',
]