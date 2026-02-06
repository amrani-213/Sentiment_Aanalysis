"""
Unit tests for data augmentation
"""
import pytest
import numpy as np
from src.data.augmentation import TextAugmenter, augment_training_data


class TestTextAugmenter:
    """Test TextAugmenter class."""
    
    def test_augmenter_initialization(self):
        """Augmenter should initialize with specified methods."""
        augmenter = TextAugmenter(
            aug_methods=['swap', 'delete'],
            aug_p=0.1
        )
        
        assert 'swap' in augmenter.aug_methods
        assert 'delete' in augmenter.aug_methods
        assert augmenter.aug_p == 0.1
    
    def test_random_swap(self, basic_augmenter):
        """Random swap should swap words."""
        text = "this is a test"
        augmented = basic_augmenter.random_swap(text, n=1)
        
        # Should be different (with high probability)
        # But have same words
        assert len(augmented.split()) == len(text.split())
    
    def test_random_deletion(self, basic_augmenter):
        """Random deletion should delete some words."""
        text = "this is a test with many words"
        augmented = basic_augmenter.random_deletion(text, p=0.3)
        
        # Should have fewer or same words
        assert len(augmented.split()) <= len(text.split())
        # Should not be empty
        assert len(augmented) > 0
    
    def test_augment_text_swap(self, basic_augmenter):
        """Augment with swap method."""
        text = "this is a test"
        augmented = basic_augmenter.augment_text(text, method='swap')
        
        assert isinstance(augmented, str)
        assert len(augmented) > 0
    
    def test_augment_text_delete(self, basic_augmenter):
        """Augment with delete method."""
        text = "this is a test"
        augmented = basic_augmenter.augment_text(text, method='delete')
        
        assert isinstance(augmented, str)
        assert len(augmented) > 0
    
    def test_augment_dataset(self, basic_augmenter, sample_short_texts, sample_short_labels):
        """Should augment entire dataset."""
        aug_texts, aug_labels = basic_augmenter.augment_dataset(
            sample_short_texts,
            sample_short_labels,
            n_aug=1,
            keep_original=True
        )
        
        # Should have more samples
        assert len(aug_texts) > len(sample_short_texts)
        assert len(aug_labels) > len(sample_short_labels)
        # Should have same number of texts and labels
        assert len(aug_texts) == len(aug_labels)
    
    def test_augment_minority_classes(self, basic_augmenter):
        """Should balance classes by augmenting minority."""
        texts = np.array(["pos"] * 10 + ["neg"] * 5 + ["neu"] * 3)
        labels = np.array([2] * 10 + [0] * 5 + [1] * 3)
        
        aug_texts, aug_labels = basic_augmenter.augment_minority_classes(
            texts, labels, target_ratio=1.0
        )
        
        # Should have more samples
        assert len(aug_texts) > len(texts)
        # Classes should be more balanced
        from collections import Counter
        counts = Counter(aug_labels)
        # Minority classes should have more samples
        assert counts[1] > 3  # Neutral was 3
        assert counts[0] > 5  # Negative was 5


class TestAugmentationFunction:
    """Test augment_training_data function."""
    
    def test_augment_training_data(self, sample_dataset_dict):
        """Should augment training data in dataset dict."""
        original_size = len(sample_dataset_dict['train']['texts'])
        
        augmented_dict = augment_training_data(
            sample_dataset_dict,
            n_aug=1,
            balance_classes=False,
            aug_methods=['swap', 'delete']
        )
        
        # Should have more training samples
        assert len(augmented_dict['train']['texts']) > original_size
        # Val and test should be unchanged
        assert len(augmented_dict['val']['texts']) == len(sample_dataset_dict['val']['texts'])
        assert len(augmented_dict['test']['texts']) == len(sample_dataset_dict['test']['texts'])


@pytest.mark.parametrize("method", ['swap', 'delete'])
def test_augmentation_methods(method):
    """All augmentation methods should work."""
    augmenter = TextAugmenter(aug_methods=[method], aug_p=0.1)
    text = "this is a test with several words"
    
    augmented = augmenter.augment_text(text, method=method)
    
    assert isinstance(augmented, str)
    assert len(augmented) > 0