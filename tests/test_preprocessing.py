"""
Unit tests for preprocessing module
Tests text cleaning, vocabulary building, tokenization, and VADER features
"""
import pytest
import torch
import numpy as np
from src.data.preprocessing import EnhancedTextPreprocessor, prepare_data


class TestTextCleaning:
    """Test text cleaning functionality."""
    
    def test_clean_text_removes_urls(self, basic_preprocessor):
        """URLs should be removed from text."""
        text = "Check this out https://example.com great site!"
        cleaned = basic_preprocessor.clean_text(text)
        
        assert "https://" not in cleaned
        assert "example.com" not in cleaned
        assert "great" in cleaned
    
    def test_clean_text_removes_mentions(self, basic_preprocessor):
        """@mentions should be removed."""
        text = "Hey @user how are you?"
        cleaned = basic_preprocessor.clean_text(text)
        
        assert "@user" not in cleaned
        assert "hey" in cleaned
        assert "you" in cleaned
    
    def test_clean_text_removes_hashtags(self, basic_preprocessor):
        """#hashtags should be removed."""
        text = "This is #awesome and #great"
        cleaned = basic_preprocessor.clean_text(text)
        
        assert "#awesome" not in cleaned
        assert "#great" not in cleaned
    
    def test_clean_text_lowercases(self, basic_preprocessor):
        """Text should be lowercased."""
        text = "THIS IS UPPERCASE TEXT"
        cleaned = basic_preprocessor.clean_text(text)
        
        assert cleaned == cleaned.lower()
        assert "THIS" not in cleaned
        assert "this" in cleaned
    
    def test_clean_text_removes_extra_spaces(self, basic_preprocessor):
        """Multiple spaces should be reduced to single space."""
        text = "This    has     many      spaces"
        cleaned = basic_preprocessor.clean_text(text)
        
        assert "    " not in cleaned
        assert "many" in cleaned
    
    def test_clean_text_handles_emojis(self, basic_preprocessor):
        """Emojis should be handled properly."""
        text = "I love this! 😊❤️"
        cleaned = basic_preprocessor.clean_text(text)
        
        # Should have some text remaining
        assert len(cleaned) > 0
        assert "love" in cleaned
    
    def test_clean_text_empty_input(self, basic_preprocessor):
        """Empty input should return empty string."""
        text = ""
        cleaned = basic_preprocessor.clean_text(text)
        
        assert cleaned == ""
    
    def test_clean_text_whitespace_only(self, basic_preprocessor):
        """Whitespace-only input should return empty string."""
        text = "     \n\t  "
        cleaned = basic_preprocessor.clean_text(text)
        
        assert cleaned.strip() == ""


class TestVocabularyBuilding:
    """Test vocabulary building and management."""
    
    def test_build_vocabulary_creates_word2idx(self, basic_preprocessor, sample_texts):
        """Vocabulary building should create word to index mapping."""
        basic_preprocessor.build_vocabulary(sample_texts)
        
        assert len(basic_preprocessor.word2idx) > 2  # At least <PAD>, <UNK>, + words
        assert '<PAD>' in basic_preprocessor.word2idx
        assert '<UNK>' in basic_preprocessor.word2idx
    
    def test_build_vocabulary_respects_vocab_size(self, sample_texts):
        """Vocabulary should not exceed max vocab_size."""
        preprocessor = EnhancedTextPreprocessor(vocab_size=10)
        preprocessor.build_vocabulary(sample_texts)
        
        assert len(preprocessor.word2idx) <= 10
    
    def test_build_vocabulary_respects_min_freq(self, sample_texts):
        """Words below min_freq should be excluded."""
        preprocessor = EnhancedTextPreprocessor(vocab_size=1000, min_freq=5)
        preprocessor.build_vocabulary(sample_texts * 2)  # Duplicate to increase freq
        
        # All words in vocab should appear at least min_freq times
        assert '<PAD>' in preprocessor.word2idx
        assert '<UNK>' in preprocessor.word2idx
    
    def test_get_vocab_size(self, preprocessor_with_vocab):
        """get_vocab_size should return correct vocabulary size."""
        vocab_size = preprocessor_with_vocab.get_vocab_size()
        
        assert vocab_size == len(preprocessor_with_vocab.word2idx)
        assert vocab_size > 2  # At least <PAD>, <UNK>, + words
    
    def test_get_padding_idx(self, preprocessor_with_vocab):
        """<PAD> token should have index 0."""
        pad_idx = preprocessor_with_vocab.get_padding_idx()
        
        assert pad_idx == 0
        assert preprocessor_with_vocab.word2idx['<PAD>'] == 0
    
    def test_vocabulary_save_and_load(self, basic_preprocessor, sample_texts, temp_vocab_path):
        """Vocabulary should be saveable and loadable."""
        # Build and save
        basic_preprocessor.build_vocabulary(sample_texts)
        original_vocab = basic_preprocessor.word2idx.copy()
        basic_preprocessor.save_vocabulary(temp_vocab_path)
        
        # Load in new preprocessor
        new_preprocessor = EnhancedTextPreprocessor()
        new_preprocessor.load_vocabulary(temp_vocab_path)
        
        assert new_preprocessor.word2idx == original_vocab
        assert new_preprocessor.idx2word == basic_preprocessor.idx2word


class TestTokenization:
    """Test text to sequence conversion."""
    
    def test_text_to_sequence_converts_text(self, preprocessor_with_vocab):
        """Text should be converted to sequence of indices."""
        text = "this is a test"
        sequence = preprocessor_with_vocab.text_to_sequence(text)
        
        assert isinstance(sequence, list)
        assert all(isinstance(idx, int) for idx in sequence)
        assert len(sequence) > 0
    
    def test_text_to_sequence_unknown_words(self, preprocessor_with_vocab):
        """Unknown words should map to <UNK> token."""
        text = "unknownword12345 anotherunkown98765"
        sequence = preprocessor_with_vocab.text_to_sequence(text)
        
        unk_idx = preprocessor_with_vocab.word2idx['<UNK>']
        # At least one token should be UNK
        assert unk_idx in sequence
    
    def test_text_to_sequence_empty_text(self, preprocessor_with_vocab):
        """Empty text should return empty sequence."""
        text = ""
        sequence = preprocessor_with_vocab.text_to_sequence(text)
        
        assert len(sequence) == 0
    
    def test_pad_sequence_pads_to_length(self, preprocessor_with_vocab):
        """Sequence should be padded to max_length."""
        sequence = [1, 2, 3]
        padded = preprocessor_with_vocab.pad_sequence(sequence)
        
        assert len(padded) == preprocessor_with_vocab.max_length
        # First elements should be original sequence
        assert padded[:3] == sequence
        # Rest should be padding
        pad_idx = preprocessor_with_vocab.get_padding_idx()
        assert all(x == pad_idx for x in padded[3:])
    
    def test_pad_sequence_truncates_long_sequences(self):
        """Long sequences should be truncated."""
        preprocessor = EnhancedTextPreprocessor(max_length=5)
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        padded = preprocessor.pad_sequence(sequence)
        
        assert len(padded) == 5
        assert padded == [1, 2, 3, 4, 5]
    
    def test_batch_to_sequences(self, preprocessor_with_vocab, sample_short_texts):
        """Batch conversion should work for multiple texts."""
        sequences = preprocessor_with_vocab.batch_to_sequences(sample_short_texts)
        
        assert len(sequences) == len(sample_short_texts)
        assert all(isinstance(seq, list) for seq in sequences)


class TestVADERFeatures:
    """Test VADER sentiment feature extraction."""
    
    def test_compute_vader_features_returns_four_scores(self, basic_preprocessor):
        """VADER should return 4 scores: compound, pos, neu, neg."""
        text = "This is a test"
        scores = basic_preprocessor.compute_vader_features(text)
        
        assert len(scores) == 4
        assert all(isinstance(score, float) for score in scores)
    
    def test_vader_positive_text_high_compound(self, basic_preprocessor):
        """Positive text should have positive compound score."""
        text = "I absolutely love this! Amazing! Wonderful! Fantastic!"
        scores = basic_preprocessor.compute_vader_features(text)
        
        compound = scores[0]
        assert compound > 0.5  # Strong positive
    
    def test_vader_negative_text_low_compound(self, basic_preprocessor):
        """Negative text should have negative compound score."""
        text = "This is terrible! Awful! Horrible! Disgusting! Worst ever!"
        scores = basic_preprocessor.compute_vader_features(text)
        
        compound = scores[0]
        assert compound < -0.5  # Strong negative
    
    def test_vader_neutral_text_near_zero(self, basic_preprocessor):
        """Neutral text should have compound score near zero."""
        text = "This is a table. It has four legs."
        scores = basic_preprocessor.compute_vader_features(text)
        
        compound = scores[0]
        assert -0.3 < compound < 0.3  # Near neutral
    
    def test_vader_scores_in_valid_range(self, basic_preprocessor):
        """All VADER scores should be in valid ranges."""
        text = "This is a test with mixed emotions good and bad"
        scores = basic_preprocessor.compute_vader_features(text)
        
        compound, pos, neu, neg = scores
        
        # Compound: [-1, 1]
        assert -1 <= compound <= 1
        # Pos, neu, neg: [0, 1]
        assert 0 <= pos <= 1
        assert 0 <= neu <= 1
        assert 0 <= neg <= 1
        # Pos + neu + neg should ≈ 1
        assert abs((pos + neu + neg) - 1.0) < 0.01
    
    def test_vader_empty_text(self, basic_preprocessor):
        """Empty text should return neutral scores."""
        text = ""
        scores = basic_preprocessor.compute_vader_features(text)
        
        compound = scores[0]
        assert compound == 0.0  # Neutral


class TestDataPreparation:
    """Test data preparation and splitting."""
    
    @pytest.mark.slow
    def test_prepare_data_creates_splits(self, tmp_path):
        """prepare_data should create train/val/test splits."""
        # Create tiny test dataset
        import pandas as pd
        df = pd.DataFrame({
            'text': ['good'] * 50 + ['bad'] * 50 + ['okay'] * 50,
            'label': [2] * 50 + [0] * 50 + [1] * 50,
            'sentiment': ['positive'] * 50 + ['negative'] * 50 + ['neutral'] * 50
        })
        
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Prepare data
        data_dict = prepare_data(
            str(csv_path),
            test_size=0.2,
            val_size=0.2,
            random_state=42
        )
        
        # Check splits exist
        assert 'train' in data_dict
        assert 'val' in data_dict
        assert 'test' in data_dict
        
        # Check each split has texts and labels
        for split in ['train', 'val', 'test']:
            assert 'texts' in data_dict[split]
            assert 'labels' in data_dict[split]
            assert len(data_dict[split]['texts']) > 0
            assert len(data_dict[split]['labels']) > 0
    
    @pytest.mark.slow
    def test_prepare_data_split_proportions(self, tmp_path):
        """Split proportions should be approximately correct."""
        import pandas as pd
        total_samples = 1000
        df = pd.DataFrame({
            'text': ['test'] * total_samples,
            'label': [1] * total_samples,
            'sentiment': ['neutral'] * total_samples
        })
        
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        data_dict = prepare_data(
            str(csv_path),
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        train_size = len(data_dict['train']['texts'])
        val_size = len(data_dict['val']['texts'])
        test_size = len(data_dict['test']['texts'])
        
        total = train_size + val_size + test_size
        
        # Check proportions (allow some rounding error)
        assert abs(test_size / total - 0.2) < 0.05
        assert abs(val_size / total - 0.1) < 0.05
        assert abs(train_size / total - 0.7) < 0.05


class TestPreprocessorIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_full_pipeline(self, basic_preprocessor, sample_texts):
        """Test complete preprocessing pipeline."""
        # Build vocabulary
        basic_preprocessor.build_vocabulary(sample_texts)
        
        # Process a text
        test_text = "This is a great product! Love it! 😊"
        
        # Clean
        cleaned = basic_preprocessor.clean_text(test_text)
        assert len(cleaned) > 0
        
        # Convert to sequence
        sequence = basic_preprocessor.text_to_sequence(cleaned)
        assert len(sequence) > 0
        
        # Pad
        padded = basic_preprocessor.pad_sequence(sequence)
        assert len(padded) == basic_preprocessor.max_length
        
        # VADER features
        vader = basic_preprocessor.compute_vader_features(test_text)
        assert len(vader) == 4
        assert vader[0] > 0  # Positive text
    
    def test_preprocessing_is_deterministic(self, basic_preprocessor, sample_texts):
        """Same input should always give same output."""
        basic_preprocessor.build_vocabulary(sample_texts)
        
        text = "Test text for determinism"
        
        # Process multiple times
        result1 = basic_preprocessor.text_to_sequence(text)
        result2 = basic_preprocessor.text_to_sequence(text)
        result3 = basic_preprocessor.text_to_sequence(text)
        
        assert result1 == result2 == result3


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("text,should_contain", [
    ("Great product!", "great"),
    ("  spaces  ", "spaces"),
    ("UPPERCASE", "uppercase"),
    ("@user mentioned", "mentioned"),
])
def test_clean_text_various_inputs(text, should_contain):
    """Test cleaning with various input types."""
    preprocessor = EnhancedTextPreprocessor()
    cleaned = preprocessor.clean_text(text)
    assert should_contain in cleaned


@pytest.mark.parametrize("positive_text", [
    "I love this!",
    "Amazing! Wonderful! Fantastic!",
    "Best product ever! Highly recommend!",
    "So happy with this purchase! 😊",
])
def test_vader_detects_positive_sentiment(positive_text):
    """VADER should detect positive sentiment."""
    preprocessor = EnhancedTextPreprocessor()
    scores = preprocessor.compute_vader_features(positive_text)
    compound = scores[0]
    assert compound > 0.05, f"Expected positive compound for '{positive_text}', got {compound}"


@pytest.mark.parametrize("negative_text", [
    "I hate this!",
    "Terrible! Awful! Horrible!",
    "Worst product ever! Do not buy!",
    "Very disappointed 😞",
])
def test_vader_detects_negative_sentiment(negative_text):
    """VADER should detect negative sentiment."""
    preprocessor = EnhancedTextPreprocessor()
    scores = preprocessor.compute_vader_features(negative_text)
    compound = scores[0]
    assert compound < -0.05, f"Expected negative compound for '{negative_text}', got {compound}"