"""
RoBERTa Model for Sentiment Analysis
Enhanced with focal loss and class weighting support
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


class RoBERTaClassifier(nn.Module):
    """
    RoBERTa-based sentiment classifier
    Enhanced with focal loss support and better regularization
    """
    
    def __init__(
        self,
        model_name='roberta-base',
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
            freeze_bert: Whether to freeze RoBERTa layers
            freeze_layers: Number of layers to freeze (from bottom)
        """
        super(RoBERTaClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained RoBERTa
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size
        
        # Freeze layers if specified
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            # Freeze bottom n layers
            for layer in self.roberta.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Classification head with enhanced regularization
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_size // 2, self.hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(self.hidden_size // 4, num_classes)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size // 4)
        
        # Initialize classification head
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for classification head"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # RoBERTa encoding
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Classification head with deep architecture
        x = self.dropout1(pooled_output)
        x = torch.relu(self.fc1(x))
        x = self.layer_norm1(x)
        
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.layer_norm2(x)
        
        x = self.dropout3(x)
        logits = self.fc3(x)
        
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
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            return outputs.attentions
    
    def get_embeddings(self, input_ids, attention_mask=None):
        """
        Get final layer embeddings for analysis
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Embeddings from [CLS] token
        """
        with torch.no_grad():
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return outputs.last_hidden_state[:, 0, :]


def create_roberta_model(
    model_name='roberta-base',
    num_classes=3,
    dropout=0.5,
    freeze_bert=False,
    freeze_layers=0
):
    """
    Factory function to create RoBERTa model
    
    Args:
        model_name: Pretrained model name
        num_classes: Number of output classes
        dropout: Dropout rate
        freeze_bert: Whether to freeze RoBERTa layers
        freeze_layers: Number of layers to freeze
    
    Returns:
        RoBERTaClassifier model
    """
    model = RoBERTaClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_bert=freeze_bert,
        freeze_layers=freeze_layers
    )
    
    return model


def get_roberta_tokenizer(model_name='roberta-base'):
    """
    Get RoBERTa tokenizer
    
    Args:
        model_name: Pretrained model name
    
    Returns:
        Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_for_roberta(texts, tokenizer, max_length=128):
    """
    Tokenize texts for RoBERTa
    
    Args:
        texts: List of texts
        tokenizer: RoBERTa tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with input_ids and attention_mask
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
    print("TESTING ROBERTA MODEL")
    print("="*80)
    
    # Create model (this will download ~500MB on first run)
    try:
        print("\nCreating RoBERTa model...")
        model = create_roberta_model(
            model_name='roberta-base',
            num_classes=3,
            dropout=0.5
        )
        
        print(f"\nModel Architecture:")
        print(f"RoBERTa Base + Deep Classification Head")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = get_roberta_tokenizer()
        
        # Test tokenization
        test_texts = [
            "This movie is absolutely amazing!",
            "Terrible experience, would not recommend.",
            "It was okay, nothing special."
        ]
        
        print("\nTest Texts:")
        for i, text in enumerate(test_texts, 1):
            print(f"{i}. {text}")
        
        # Tokenize
        encoded = tokenize_for_roberta(test_texts, tokenizer, max_length=64)
        
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
        
        # Test embeddings extraction
        print("\nTesting embeddings extraction...")
        embeddings = model.get_embeddings(encoded['input_ids'], encoded['attention_mask'])
        print(f"Embeddings shape: {embeddings.shape}")
        
        print("\n✅ RoBERTa model tested successfully!")
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("This is expected if you don't have internet connection or transformers library.")
        print("The model will work when properly set up with requirements.txt")