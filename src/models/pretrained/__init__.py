"""
Pretrained transformer models for sentiment analysis
RoBERTa and BERTweet implementations with factory functions
"""
# RoBERTa model
from .roberta import (
    RoBERTaClassifier,
    create_roberta_model,
    get_roberta_tokenizer,
    tokenize_for_roberta
)

# BERTweet model
from .bertweet import (
    BERTweetClassifier,
    create_bertweet_model,
    get_bertweet_tokenizer,
    tokenize_for_bertweet
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Amrani Bouabdellah'

# Public API - defines what gets imported with `from src.models.pretrained import *`
__all__ = [
    # RoBERTa
    'RoBERTaClassifier',
    'create_roberta_model',
    'get_roberta_tokenizer',
    'tokenize_for_roberta',
    
    # BERTweet
    'BERTweetClassifier',
    'create_bertweet_model',
    'get_bertweet_tokenizer',
    'tokenize_for_bertweet',
]