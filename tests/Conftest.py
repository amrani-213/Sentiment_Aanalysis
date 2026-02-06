"""
Shared pytest fixtures for all tests
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "I absolutely love this product! It's amazing and wonderful! 😊",
        "This is terrible. Waste of money. Very disappointed 😞",
        "It's okay. Nothing special. Average quality.",
        "Best purchase ever! Highly recommend! Fantastic!",
        "Horrible experience. Do not buy this. Awful quality."
    ]


@pytest.fixture
def sample_labels():
    """Provide corresponding labels (0=negative, 1=neutral, 2=positive)."""
    return np.array([2, 0, 1, 2, 0])


@pytest.fixture
def sample_short_texts():
    """Short texts for quick testing."""
    return [
        "Great!",
        "Bad.",
        "Okay"
    ]


@pytest.fixture
def sample_short_labels():
    """Labels for short texts."""
    return np.array([2, 0, 1])


# ============================================================================
# Preprocessor Fixtures
# ============================================================================

@pytest.fixture
def basic_preprocessor():
    """Create a basic preprocessor instance."""
    from src.data.preprocessing import EnhancedTextPreprocessor
    
    return EnhancedTextPreprocessor(
        vocab_size=1000,
        max_length=50,
        min_freq=1,
        use_spell_check=False,  # Disable for speed
        use_lemmatization=False
    )


@pytest.fixture
def preprocessor_with_vocab(basic_preprocessor, sample_texts):
    """Create preprocessor with built vocabulary."""
    basic_preprocessor.build_vocabulary(sample_texts)
    return basic_preprocessor


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def small_bilstm_model():
    """Create small BiLSTM model for testing."""
    from src.models.baseline.bilstm_attention import create_bilstm_model
    
    return create_bilstm_model(
        vocab_size=100,
        embedding_dim=32,
        hidden_dim=16,
        num_layers=1,
        num_classes=3,
        dropout=0.1,
        num_attention_heads=2,
        use_sentiment_features=True
    )


@pytest.fixture
def small_transformer_model():
    """Create small transformer model for testing."""
    from src.models.baseline.custom_transformer import create_custom_transformer
    
    return create_custom_transformer(
        vocab_size=100,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=64,
        num_classes=3,
        max_len=50,
        dropout=0.1
    )


@pytest.fixture
def small_fasttext_model():
    """Create small FastText model for testing."""
    from src.models.baseline.fasttext import create_fasttext_model
    
    return create_fasttext_model(
        vocab_size=100,
        embedding_dim=16,
        num_classes=3,
        dropout=0.1,
        use_char_ngrams=False
    )


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 4
    seq_len = 20
    
    return {
        'text': torch.randint(0, 100, (batch_size, seq_len)),
        'label': torch.randint(0, 3, (batch_size,)),
        'sentiment_score': torch.randn(batch_size, 4)
    }


@pytest.fixture
def sample_dataset_dict():
    """Create sample data dictionary."""
    return {
        'train': {
            'texts': np.array([
                "Great product",
                "Terrible quality",
                "It's okay",
                "Love it",
                "Hate this"
            ] * 10),  # 50 samples
            'labels': np.array([2, 0, 1, 2, 0] * 10)
        },
        'val': {
            'texts': np.array([
                "Amazing",
                "Awful",
                "Fine"
            ] * 5),  # 15 samples
            'labels': np.array([2, 0, 1] * 5)
        },
        'test': {
            'texts': np.array([
                "Excellent",
                "Poor",
                "Average"
            ] * 5),  # 15 samples
            'labels': np.array([2, 0, 1] * 5)
        }
    }


# ============================================================================
# Augmenter Fixtures
# ============================================================================

@pytest.fixture
def basic_augmenter():
    """Create basic text augmenter."""
    from src.data.augmentation import TextAugmenter
    
    return TextAugmenter(
        aug_methods=['swap', 'delete'],  # Fast methods only
        aug_p=0.1
    )


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_model_path(tmp_path):
    """Provide temporary path for saving models."""
    return tmp_path / "test_model.pt"


@pytest.fixture
def temp_vocab_path(tmp_path):
    """Provide temporary path for saving vocabulary."""
    return tmp_path / "test_vocab.pkl"


# ============================================================================
# Loss Function Fixtures
# ============================================================================

@pytest.fixture
def sample_logits():
    """Sample model outputs (logits)."""
    batch_size = 8
    num_classes = 3
    return torch.randn(batch_size, num_classes)


@pytest.fixture
def sample_targets():
    """Sample target labels."""
    batch_size = 8
    return torch.randint(0, 3, (batch_size,))


# ============================================================================
# Metric Fixtures
# ============================================================================

@pytest.fixture
def sample_predictions():
    """Sample predictions for metrics."""
    return np.array([2, 0, 1, 2, 0, 1, 2, 0, 1, 2])


@pytest.fixture
def sample_true_labels():
    """Sample true labels for metrics."""
    return np.array([2, 0, 1, 2, 1, 1, 2, 0, 0, 2])


@pytest.fixture
def sample_probabilities():
    """Sample probability distributions."""
    # 10 samples, 3 classes
    probs = np.random.rand(10, 3)
    # Normalize to sum to 1
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get computing device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ============================================================================
# Parametrize Helpers
# ============================================================================

# Common test cases for parametrization
SENTIMENT_TEST_CASES = [
    ("I love this!", 2),  # positive
    ("This is terrible", 0),  # negative
    ("It's okay", 1),  # neutral
    ("Amazing product! Best ever!", 2),
    ("Worst purchase. Never again.", 0),
    ("Could be better. Could be worse.", 1),
]

EDGE_CASE_TEXTS = [
    "",  # empty
    "   ",  # whitespace only
    "a",  # single character
    "A" * 1000,  # very long
    "!!!???",  # only punctuation
    "😊😊😊",  # only emojis
    "http://example.com",  # only URL
    "@user #hashtag",  # only mentions/hashtags
]