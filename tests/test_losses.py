"""
Unit tests for loss functions
"""
import pytest
import torch
import numpy as np
from src.training.losses import (
    FocalLoss,
    LabelSmoothingLoss,
    WeightedCrossEntropyLoss,
    CombinedLoss,
    compute_class_weights,
    get_loss_function
)


class TestFocalLoss:
    """Test Focal Loss implementation."""
    
    def test_focal_loss_initialization(self):
        """Focal loss should initialize correctly."""
        loss_fn = FocalLoss(gamma=2.0)
        assert loss_fn.gamma == 2.0
    
    def test_focal_loss_forward(self, sample_logits, sample_targets):
        """Focal loss should compute loss."""
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(sample_logits, sample_targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_focal_loss_with_alpha(self, sample_logits, sample_targets):
        """Focal loss should work with class weights."""
        alpha = [1.0, 1.5, 2.0]
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        loss = loss_fn(sample_logits, sample_targets)
        
        assert loss.item() > 0


class TestLabelSmoothingLoss:
    """Test Label Smoothing Loss."""
    
    def test_label_smoothing_initialization(self):
        """Label smoothing should initialize correctly."""
        loss_fn = LabelSmoothingLoss(num_classes=3, smoothing=0.1)
        assert loss_fn.smoothing == 0.1
        assert loss_fn.num_classes == 3
    
    def test_label_smoothing_forward(self, sample_logits, sample_targets):
        """Label smoothing should compute loss."""
        loss_fn = LabelSmoothingLoss(num_classes=3, smoothing=0.1)
        loss = loss_fn(sample_logits, sample_targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestWeightedCrossEntropyLoss:
    """Test Weighted Cross-Entropy Loss."""
    
    def test_weighted_ce_initialization(self):
        """Weighted CE should initialize with weights."""
        weights = [1.0, 1.5, 2.0]
        loss_fn = WeightedCrossEntropyLoss(weights)
        
        assert torch.all(loss_fn.class_weights == torch.FloatTensor(weights))
    
    def test_weighted_ce_forward(self, sample_logits, sample_targets):
        """Weighted CE should compute loss."""
        weights = [1.0, 1.5, 2.0]
        loss_fn = WeightedCrossEntropyLoss(weights)
        loss = loss_fn(sample_logits, sample_targets)
        
        assert loss.item() > 0


class TestComputeClassWeights:
    """Test class weight computation."""
    
    def test_compute_weights_inverse_freq(self):
        """Should compute inverse frequency weights."""
        labels = np.array([0, 0, 0, 1, 1, 2])  # Imbalanced
        weights = compute_class_weights(labels, method='inverse_freq', num_classes=3)
        
        assert len(weights) == 3
        # Class 0 (most frequent) should have lowest weight
        assert weights[0] < weights[1]
        assert weights[0] < weights[2]
    
    def test_compute_weights_effective_num(self):
        """Should compute effective number weights."""
        labels = np.array([0] * 10 + [1] * 5 + [2] * 2)
        weights = compute_class_weights(labels, method='effective_num', num_classes=3)
        
        assert len(weights) == 3
        assert all(w > 0 for w in weights)


class TestGetLossFunction:
    """Test loss function factory."""
    
    def test_get_cross_entropy(self):
        """Should return cross entropy loss."""
        loss_fn = get_loss_function('cross_entropy')
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    
    def test_get_focal_loss(self):
        """Should return focal loss."""
        loss_fn = get_loss_function('focal', gamma=2.0)
        assert isinstance(loss_fn, FocalLoss)
    
    def test_get_label_smoothing(self):
        """Should return label smoothing loss."""
        loss_fn = get_loss_function('label_smoothing', num_classes=3, smoothing=0.1)
        assert isinstance(loss_fn, LabelSmoothingLoss)
    
    def test_unknown_loss_type(self):
        """Should raise error for unknown loss type."""
        with pytest.raises(ValueError):
            get_loss_function('unknown_loss')