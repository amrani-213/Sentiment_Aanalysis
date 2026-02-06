"""
Unit tests for model architectures
Tests forward passes, output shapes, and model components
"""
import pytest
import torch
import torch.nn as nn


class TestBiLSTMModel:
    """Test BiLSTM with Multi-Head Attention model."""
    
    def test_model_forward_pass(self, small_bilstm_model, sample_batch):
        """Model should complete forward pass without errors."""
        model = small_bilstm_model
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                sample_batch['text'],
                sentiment_scores=sample_batch['sentiment_score']
            )
        
        assert outputs is not None
        assert outputs.shape[0] == sample_batch['text'].shape[0]
    
    def test_output_shape(self, small_bilstm_model):
        """Output should have shape (batch_size, num_classes)."""
        model = small_bilstm_model
        batch_size = 8
        seq_len = 20
        
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        sentiment = torch.randn(batch_size, 4)
        
        outputs = model(inputs, sentiment_scores=sentiment)
        
        assert outputs.shape == (batch_size, 3)  # 3 classes
    
    def test_model_parameters_trainable(self, small_bilstm_model):
        """Model should have trainable parameters."""
        trainable_params = sum(
            p.numel() for p in small_bilstm_model.parameters() 
            if p.requires_grad
        )
        
        assert trainable_params > 0
    
    def test_model_with_different_batch_sizes(self, small_bilstm_model):
        """Model should handle different batch sizes."""
        model = small_bilstm_model
        
        for batch_size in [1, 4, 16, 32]:
            inputs = torch.randint(0, 100, (batch_size, 20))
            sentiment = torch.randn(batch_size, 4)
            
            outputs = model(inputs, sentiment_scores=sentiment)
            assert outputs.shape == (batch_size, 3)
    
    def test_model_without_sentiment_scores(self, small_bilstm_model):
        """Model should work without sentiment scores."""
        model = small_bilstm_model
        inputs = torch.randint(0, 100, (8, 20))
        
        # Should not raise error
        outputs = model(inputs, sentiment_scores=None)
        assert outputs.shape == (8, 3)
    
    def test_attention_weights_extraction(self, small_bilstm_model):
        """Model should be able to extract attention weights."""
        model = small_bilstm_model
        inputs = torch.randint(0, 100, (1, 20))
        sentiment = torch.randn(1, 4)
        
        attn_weights = model.get_attention_weights(inputs, sentiment)
        
        assert attn_weights is not None
        assert attn_weights.dim() >= 2  # At least 2D


class TestCustomTransformer:
    """Test custom transformer model."""
    
    def test_model_forward_pass(self, small_transformer_model):
        """Model should complete forward pass."""
        model = small_transformer_model
        batch_size = 8
        seq_len = 20
        
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        outputs = model(inputs)
        
        assert outputs.shape == (batch_size, 3)
    
    def test_output_shape(self, small_transformer_model):
        """Output shape should be correct."""
        model = small_transformer_model
        batch_size = 16
        seq_len = 30
        
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        outputs = model(inputs)
        
        assert outputs.shape == (batch_size, 3)
    
    def test_model_with_mask(self, small_transformer_model):
        """Model should handle attention masks."""
        model = small_transformer_model
        batch_size = 8
        seq_len = 20
        
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        mask = (inputs != 0).float()
        
        outputs = model(inputs, mask=mask)
        assert outputs.shape == (batch_size, 3)
    
    def test_positional_encoding(self, small_transformer_model):
        """Positional encoding should be applied."""
        # Get positional encoding module
        pos_encoding = small_transformer_model.pos_encoding
        
        batch_size = 4
        seq_len = 10
        d_model = small_transformer_model.d_model
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Apply positional encoding
        output = pos_encoding(x)
        
        assert output.shape == x.shape


class TestFastTextModel:
    """Test FastText model."""
    
    def test_model_forward_pass(self, small_fasttext_model):
        """Model should complete forward pass."""
        model = small_fasttext_model
        batch_size = 8
        seq_len = 20
        
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        outputs = model(inputs)
        
        assert outputs.shape == (batch_size, 3)
    
    def test_output_shape(self, small_fasttext_model):
        """Output should have correct shape."""
        model = small_fasttext_model
        
        for batch_size in [1, 8, 16]:
            inputs = torch.randint(0, 100, (batch_size, 20))
            outputs = model(inputs)
            assert outputs.shape == (batch_size, 3)
    
    def test_model_with_mask(self, small_fasttext_model):
        """Model should handle masks properly."""
        model = small_fasttext_model
        batch_size = 8
        seq_len = 20
        
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        mask = (inputs != 0).float()
        
        outputs = model(inputs, mask=mask)
        assert outputs.shape == (batch_size, 3)


class TestModelGradients:
    """Test gradient flow through models."""
    
    def test_bilstm_gradients_flow(self, small_bilstm_model):
        """Gradients should flow through BiLSTM."""
        model = small_bilstm_model
        model.train()
        
        inputs = torch.randint(0, 100, (4, 20))
        sentiment = torch.randn(4, 4)
        targets = torch.randint(0, 3, (4,))
        
        # Forward pass
        outputs = model(inputs, sentiment_scores=sentiment)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_transformer_gradients_flow(self, small_transformer_model):
        """Gradients should flow through transformer."""
        model = small_transformer_model
        model.train()
        
        inputs = torch.randint(0, 100, (4, 20))
        targets = torch.randint(0, 3, (4,))
        
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        
        # Check gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelSaveLoad:
    """Test model saving and loading."""
    
    def test_save_and_load_bilstm(self, small_bilstm_model, temp_model_path):
        """Model should be saveable and loadable."""
        model = small_bilstm_model
        
        # Save
        torch.save(model.state_dict(), temp_model_path)
        
        # Load in new model
        from src.models.baseline.bilstm_attention import create_bilstm_model
        new_model = create_bilstm_model(
            vocab_size=100,
            embedding_dim=32,
            hidden_dim=16,
            num_layers=1,
            num_classes=3
        )
        new_model.load_state_dict(torch.load(temp_model_path))
        
        # Test forward pass
        inputs = torch.randint(0, 100, (4, 20))
        sentiment = torch.randn(4, 4)
        
        with torch.no_grad():
            outputs = new_model(inputs, sentiment_scores=sentiment)
        
        assert outputs.shape == (4, 3)
    
    def test_save_and_load_transformer(self, small_transformer_model, temp_model_path):
        """Transformer should be saveable and loadable."""
        model = small_transformer_model
        
        torch.save(model.state_dict(), temp_model_path)
        
        from src.models.baseline.custom_transformer import create_custom_transformer
        new_model = create_custom_transformer(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=2,
            num_classes=3
        )
        new_model.load_state_dict(torch.load(temp_model_path))
        
        inputs = torch.randint(0, 100, (4, 20))
        
        with torch.no_grad():
            outputs = new_model(inputs)
        
        assert outputs.shape == (4, 3)


class TestModelOutput:
    """Test model output properties."""
    
    def test_output_is_logits(self, small_bilstm_model):
        """Model output should be logits (not probabilities)."""
        model = small_bilstm_model
        inputs = torch.randint(0, 100, (8, 20))
        sentiment = torch.randn(8, 4)
        
        outputs = model(inputs, sentiment_scores=sentiment)
        
        # Logits can be any real number
        # Probabilities would be [0, 1] and sum to 1
        # Check that not all values are in [0, 1]
        assert outputs.max() > 1.0 or outputs.min() < 0.0
    
    def test_softmax_gives_probabilities(self, small_bilstm_model):
        """Applying softmax should give valid probabilities."""
        model = small_bilstm_model
        inputs = torch.randint(0, 100, (8, 20))
        sentiment = torch.randn(8, 4)
        
        logits = model(inputs, sentiment_scores=sentiment)
        probs = torch.softmax(logits, dim=1)
        
        # Check probability properties
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        # Sum to 1 (allow small numerical error)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


# ============================================================================
# Integration Tests
# ============================================================================

class TestModelIntegration:
    """Integration tests for models with realistic scenarios."""
    
    @pytest.mark.slow
    def test_train_step(self, small_bilstm_model):
        """Model should be able to perform training step."""
        model = small_bilstm_model
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create batch
        batch_size = 16
        inputs = torch.randint(0, 100, (batch_size, 20))
        sentiment = torch.randn(batch_size, 4)
        targets = torch.randint(0, 3, (batch_size,))
        
        # Training step
        optimizer.zero_grad()
        outputs = model(inputs, sentiment_scores=sentiment)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Loss should be computed
        assert loss.item() > 0
    
    @pytest.mark.slow
    def test_eval_step(self, small_bilstm_model):
        """Model should be able to perform evaluation."""
        model = small_bilstm_model
        model.eval()
        
        batch_size = 16
        inputs = torch.randint(0, 100, (batch_size, 20))
        sentiment = torch.randn(batch_size, 4)
        targets = torch.randint(0, 3, (batch_size,))
        
        with torch.no_grad():
            outputs = model(inputs, sentiment_scores=sentiment)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        # Check predictions
        assert preds.shape == targets.shape
        assert (preds >= 0).all() and (preds < 3).all()


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("batch_size,seq_len", [
    (1, 10),
    (4, 20),
    (16, 50),
    (32, 100),
])
def test_various_input_shapes(small_bilstm_model, batch_size, seq_len):
    """Model should handle various input shapes."""
    model = small_bilstm_model
    inputs = torch.randint(0, 100, (batch_size, seq_len))
    sentiment = torch.randn(batch_size, 4)
    
    outputs = model(inputs, sentiment_scores=sentiment)
    assert outputs.shape == (batch_size, 3)


@pytest.mark.parametrize("model_fixture", [
    "small_bilstm_model",
    "small_transformer_model",
    "small_fasttext_model",
])
def test_all_models_forward_pass(model_fixture, request):
    """All models should be able to forward pass."""
    model = request.getfixturevalue(model_fixture)
    
    batch_size = 8
    seq_len = 20
    inputs = torch.randint(0, 100, (batch_size, seq_len))
    
    # Handle models that need sentiment scores
    if model_fixture == "small_bilstm_model":
        sentiment = torch.randn(batch_size, 4)
        outputs = model(inputs, sentiment_scores=sentiment)
    else:
        outputs = model(inputs)
    
    assert outputs.shape == (batch_size, 3)