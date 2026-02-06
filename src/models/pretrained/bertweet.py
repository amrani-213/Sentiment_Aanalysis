"""
BERTweet Model for Sentiment Analysis
Twitter-specific BERT model with enhanced regularization
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


class BERTweetClassifier(nn.Module):
    """
    BERTweet-based sentiment classifier
    Enhanced with better regularization and optional focal loss
    """
    
    def __init__(
        self,
        model_name='vinai/bertweet-base',
        num_classes=3,
        dropout=0.5,
        freeze_bert=False,
        freeze_layers=0
    ):
        """
        Args:
            model_name: Pretrained model name
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT layers
            freeze_layers: Number of layers to freeze (from bottom)
        """
        super(BERTweetClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained BERTweet
        self.bertweet = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bertweet.config.hidden_size
        
        # Freeze layers if specified
        if freeze_bert:
            for param in self.bertweet.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            # Freeze bottom n layers
            for layer in self.bertweet.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Classification head with enhanced regularization
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_size // 2, num_classes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size // 2)
        
        # Initialize classification head
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for classification head"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Token type IDs (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # BERTweet encoding
        outputs = self.bertweet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if hasattr(self.bertweet.config, 'type_vocab_size') else None
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Classification head
        x = self.dropout1(pooled_output)
        x = torch.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        
        return logits
    
    def get_attention_weights(self, input_ids, attention_mask=None):
        """
        Get attention weights for visualization
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Attention weights from all layers
        """
        with torch.no_grad():
            outputs = self.bertweet(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            return outputs.attentions


def create_bertweet_model(
    model_name='vinai/bertweet-base',
    num_classes=3,
    dropout=0.5,
    freeze_bert=False,
    freeze_layers=0
):
    """
    Factory function to create BERTweet model
    
    Args:
        model_name: Pretrained model name
        num_classes: Number of output classes
        dropout: Dropout rate
        freeze_bert: Whether to freeze BERT layers
        freeze_layers: Number of layers to freeze
    
    Returns:
        BERTweetClassifier model
    """
    model = BERTweetClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_bert=freeze_bert,
        freeze_layers=freeze_layers
    )
    
    return model


def get_bertweet_tokenizer(model_name='vinai/bertweet-base'):
    """
    Get BERTweet tokenizer
    
    Args:
        model_name: Pretrained model name
    
    Returns:
        Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return tokenizer


def tokenize_for_bertweet(texts, tokenizer, max_length=128):
    """
    Tokenize texts for BERTweet
    
    Args:
        texts: List of texts
        tokenizer: BERTweet tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with input_ids, attention_mask, token_type_ids
    """
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return encoded


if __name__ == "__main__":
    print("="*80)
    print("TESTING BERTWEET MODEL")
    print("="*80)
    
    # Create model (this will download ~550MB on first run)
    try:
        print("\nCreating BERTweet model...")
        model = create_bertweet_model(
            model_name='vinai/bertweet-base',
            num_classes=3,
            dropout=0.5
        )
        
        print(f"\nModel Architecture:")
        print(f"BERTweet Base + Classification Head")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = get_bertweet_tokenizer()
        
        # Test tokenization
        test_texts = [
            "This movie is absolutely amazing! 😍",
            "Terrible experience, would not recommend.",
            "It was okay, nothing special."
        ]
        
        print("\nTest Texts:")
        for i, text in enumerate(test_texts, 1):
            print(f"{i}. {text}")
        
        # Tokenize
        encoded = tokenize_for_bertweet(test_texts, tokenizer, max_length=64)
        
        print(f"\nTokenized shapes:")
        print(f"input_ids: {encoded['input_ids'].shape}")
        print(f"attention_mask: {encoded['attention_mask'].shape}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        model.eval()
        with torch.no_grad():
            logits = model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
        
        print(f"Output logits shape: {logits.shape}")
        
        # Get predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        print(f"\nPredictions:")
        class_names = ['Negative', 'Neutral', 'Positive']
        for i, (text, pred, prob) in enumerate(zip(test_texts, preds, probs)):
            print(f"{i+1}. {class_names[pred.item()]} (confidence: {prob[pred].item():.3f})")
        
        print("\n✅ BERTweet model tested successfully!")
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("This is expected if you don't have internet connection or transformers library.")
        print("The model will work when properly set up with requirements.txt")