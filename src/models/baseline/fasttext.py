"""
FastText Model for Sentiment Analysis
Enhanced with character n-grams and efficient word representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import re
from typing import Optional, Tuple, Dict, Any

class FastTextModel(nn.Module):
    """
    FastText-inspired model for sentiment analysis
    Uses averaged word embeddings with character n-grams
    """
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        num_classes=3,
        dropout=0.3,
        padding_idx=0,
        use_char_ngrams=True,
        ngram_vocab_size=10000
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_classes: Number of output classes
            dropout: Dropout rate
            padding_idx: Padding token index
            use_char_ngrams: Whether to use character n-grams
            ngram_vocab_size: Size of n-gram vocabulary
        """
        super(FastTextModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.use_char_ngrams = use_char_ngrams
        self.use_sentiment_features = False
        self.word_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        if use_char_ngrams:
            self.ngram_embedding = nn.Embedding(
                ngram_vocab_size,
                embedding_dim,
                padding_idx=0
            )
        
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc2 = nn.Linear(embedding_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.uniform_(self.word_embedding.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        if self.use_char_ngrams:
            nn.init.uniform_(self.ngram_embedding.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim)
        
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x, ngram_indices=None, mask=None, **kwargs):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len)
            ngram_indices: Character n-gram indices (batch_size, seq_len, num_ngrams)
            mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len = x.size()
        
        if mask is None:
            mask = (x != 0).float()  
        
        word_emb = self.word_embedding(x) 
        
        if self.use_char_ngrams and ngram_indices is not None:
            ngram_emb = self.ngram_embedding(ngram_indices)  
            ngram_emb = ngram_emb.mean(dim=2) 
            
            combined_emb = (word_emb + ngram_emb) / 2
        else:
            combined_emb = word_emb
        
        mask_expanded = mask.unsqueeze(-1).expand(combined_emb.size())
        sum_embeddings = torch.sum(combined_emb * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  
        
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits


class CharNGramExtractor:
    """
    Extract character n-grams from words
    FastText-style character n-gram features
    """
    
    def __init__(self, min_n=3, max_n=6, max_vocab_size=10000):
        """
        Args:
            min_n: Minimum n-gram length
            max_n: Maximum n-gram length
            max_vocab_size: Maximum vocabulary size for n-grams
        """
        self.min_n = min_n
        self.max_n = max_n
        self.max_vocab_size = max_vocab_size
        self.ngram2idx = {}
        self.idx2ngram = {}
        self.ngram_counts = Counter()
    
    def extract_ngrams(self, word):
        """
        Extract character n-grams from a word
        
        Args:
            word: Input word
        
        Returns:
            List of n-grams
        """
        word = f"<{word}>"
        ngrams = []
        
        for n in range(self.min_n, min(len(word), self.max_n) + 1):
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])
        
        return ngrams
    
    def build_vocab(self, texts, word2idx):
        """
        Build n-gram vocabulary from texts
        
        Args:
            texts: List of texts
            word2idx: Word to index mapping
        """
        print("Building character n-gram vocabulary...")
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word in word2idx:
                    ngrams = self.extract_ngrams(word)
                    self.ngram_counts.update(ngrams)
        
        self.ngram2idx['<PAD>'] = 0
        self.idx2ngram[0] = '<PAD>'
        
        for idx, (ngram, count) in enumerate(self.ngram_counts.most_common(self.max_vocab_size - 1), 1):
            self.ngram2idx[ngram] = idx
            self.idx2ngram[idx] = ngram
        
        print(f"N-gram vocabulary size: {len(self.ngram2idx)}")
    
    def word_to_ngram_indices(self, word, num_ngrams=10):
        """
        Convert word to n-gram indices
        
        Args:
            word: Input word
            num_ngrams: Number of n-grams to return (fixed size)
        
        Returns:
            List of n-gram indices
        """
        ngrams = self.extract_ngrams(word)
        indices = [self.ngram2idx.get(ng, 0) for ng in ngrams]
        
        if len(indices) < num_ngrams:
            indices += [0] * (num_ngrams - len(indices))
        else:
            indices = indices[:num_ngrams]
        
        return indices


def create_fasttext_model(
    vocab_size,
    embedding_dim=100,
    num_classes=3,
    dropout=0.3,
    padding_idx=0,
    use_char_ngrams=True,
    ngram_vocab_size=10000
):
    config = {
        
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'num_classes': num_classes,
        'dropout': dropout,
        'padding_idx': padding_idx,
        'use_char_ngrams': use_char_ngrams,
        'ngram_vocab_size': ngram_vocab_size
    }
    
    model = FastTextModel(**config)
    
    return model, config 


if __name__ == "__main__":
    print("="*80)
    print("TESTING FASTTEXT MODEL")
    print("="*80)
    
    vocab_size = 10000
    batch_size = 16
    seq_len = 100
    num_ngrams = 10
    
    model = create_fasttext_model(
        vocab_size=vocab_size,
        embedding_dim=100,
        num_classes=3,
        dropout=0.3,
        use_char_ngrams=True,
        ngram_vocab_size=10000
    )
    
    print(f"\nModel Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\nInput shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape (without n-grams): {output.shape}")
    
    dummy_ngrams = torch.randint(0, 10000, (batch_size, seq_len, num_ngrams))
    output_with_ngrams = model(dummy_input, ngram_indices=dummy_ngrams)
    print(f"Output shape (with n-grams): {output_with_ngrams.shape}")
    
    print("\n" + "="*80)
    print("TESTING CHARACTER N-GRAM EXTRACTOR")
    print("="*80)
    
    extractor = CharNGramExtractor(min_n=3, max_n=6)
    
    test_word = "amazing"
    ngrams = extractor.extract_ngrams(test_word)
    print(f"\nWord: '{test_word}'")
    print(f"Character n-grams: {ngrams}")
    
    test_texts = ["this is amazing", "wonderful movie", "terrible experience"]
    word2idx = {'this': 1, 'is': 2, 'amazing': 3, 'wonderful': 4, 'movie': 5, 'terrible': 6, 'experience': 7}
    extractor.build_vocab(test_texts, word2idx)
    
    indices = extractor.word_to_ngram_indices("amazing", num_ngrams=10)
    print(f"\nN-gram indices for 'amazing': {indices}")
    
    print("\n✅ FastText model tested successfully!")
    print("\nThis model uses:")
    print("  - Averaged word embeddings")
    print("  - Character n-gram features (optional)")
    print("  - Efficient and fast training")
