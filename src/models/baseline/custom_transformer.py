"""
Custom Transformer Model from Scratch
Demonstrates understanding of transformer architecture without using nn.Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Any, Dict  


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    Uses sinusoidal functions to encode position information
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from scratch
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention
        
        Args:
            Q: Queries (batch_size, num_heads, seq_len, d_k)
            K: Keys (batch_size, num_heads, seq_len, d_k)
            V: Values (batch_size, num_heads, seq_len, d_k)
            mask: Attention mask
        
        Returns:
            Output and attention weights
        """
     
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        

        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Attention mask
        
        Returns:
            Output tensor and attention weights
        """
        batch_size = query.size(0)
        
        Q = self.W_q(query)  
        K = self.W_k(key)
        V = self.W_v(value)
        
      
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
 
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Output tensor
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask
        
        Returns:
            Output tensor
        """
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class CustomTransformer(nn.Module):
    """
    Custom Transformer for Sentiment Analysis (built from scratch)
    
    Architecture:
    1. Embedding layer
    2. Positional encoding
    3. Multiple transformer encoder layers
    4. Global pooling
    5. Classification head
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        num_classes: int = 3,
        max_len: int = 512,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            num_classes: Number of output classes
            max_len: Maximum sequence length
            dropout: Dropout rate
            padding_idx: Padding token index
        """
        self.use_sentiment_features = False
        
        super(CustomTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder_layers: nn.ModuleList = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len)
            mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len = x.size()
        
        if mask is None:
            mask = (x != 0).float()  
        
        x = self.embedding(x) * math.sqrt(self.d_model)

        x = self.pos_encoding(x)
        
   
        attn_mask = mask.unsqueeze(1).unsqueeze(2)
  
        for encoder_layer in list(self.encoder_layers):
            x = encoder_layer(x, attn_mask)
      
        mask_expanded = mask.unsqueeze(-1).expand(x.size())
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # (batch_size, d_model)
    
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits


def create_custom_transformer(
    vocab_size,
    d_model=256,
    num_heads=4,
    num_layers=4,
    d_ff=1024,
    num_classes=3,
    max_len=512,
    dropout=0.1,
    padding_idx=0
):
    config = {
        
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
        'num_classes': num_classes,
        'max_len': max_len,
        'dropout': dropout,
        'padding_idx': padding_idx
    }
    
    model = CustomTransformer(**config)
    
    return model, config  


if __name__ == "__main__":
    print("="*80)
    print("TESTING CUSTOM TRANSFORMER MODEL")
    print("="*80)
    
    vocab_size = 10000
    batch_size = 16
    seq_len = 100
    
    model = create_custom_transformer(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        num_classes=3,
        dropout=0.1
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
    print(f"Output shape: {output.shape}")
    
    mask = (dummy_input != 0).float()
    output_with_mask = model(dummy_input, mask=mask)
    print(f"Output with mask shape: {output_with_mask.shape}")
    
    print("\n✅ Custom Transformer model tested successfully!")
    print("\nThis model demonstrates:")
    print("  - Custom multi-head attention from scratch")
    print("  - Positional encoding")
    print("  - Transformer encoder layers")
    print("  - Residual connections and layer normalization")
