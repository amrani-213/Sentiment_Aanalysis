"""
Unit tests for metrics calculation
"""
import pytest
import numpy as np
import torch
from src.training.metrics import (
    calculate_metrics,
    print_metrics,
    calculate_top_k_accuracy,
    calculate_calibration_error,
    calculate_confidence_metrics,
    compare_models
)


class TestCalculateMetrics:
    """Test metrics calculation."""
    
    def test_basic_metrics(self, sample_true_labels, sample_predictions):
        """Should calculate basic metrics."""
        metrics = calculate_metrics(sample_true_labels, sample_predictions)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics
        
        # Metrics should be in [0, 1]
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
    
    def test_metrics_with_probabilities(self, sample_true_labels, sample_predictions, sample_probabilities):
        """Should calculate ROC-AUC when probabilities provided."""
        metrics = calculate_metrics(
            sample_true_labels,
            sample_predictions,
            y_prob=sample_probabilities
        )
        
        assert 'roc_auc_macro' in metrics
        assert 0 <= metrics['roc_auc_macro'] <= 1
    
    def test_per_class_metrics(self, sample_true_labels, sample_predictions):
        """Should calculate per-class metrics."""
        metrics = calculate_metrics(sample_true_labels, sample_predictions)
        
        assert 'per_class' in metrics
        for class_name in ['Negative', 'Neutral', 'Positive']:
            assert class_name in metrics['per_class']
            class_metrics = metrics['per_class'][class_name]
            assert 'precision' in class_metrics
            assert 'recall' in class_metrics
            assert 'f1' in class_metrics
    
    def test_perfect_predictions(self):
        """Perfect predictions should give 1.0 accuracy."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])  # Perfect
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['f1_macro'] == 1.0
    
    def test_worst_predictions(self):
        """Completely wrong predictions should give low scores."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])  # All wrong
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] < 0.5


class TestTopKAccuracy:
    """Test top-k accuracy calculation."""
    
    def test_top_2_accuracy(self):
        """Should calculate top-2 accuracy."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [0.8, 0.15, 0.05],  # Correct (0 is top-1)
            [0.3, 0.4, 0.3],    # Correct (1 is top-1)
            [0.6, 0.3, 0.1]     # Wrong (2 not in top-2)
        ])
        
        top2_acc = calculate_top_k_accuracy(y_true, y_prob, k=2)
        
        # 2/3 correct in top-2
        assert abs(top2_acc - 2/3) < 0.01


class TestConfidenceMetrics:
    """Test confidence metrics calculation."""
    
    def test_confidence_metrics(self, sample_probabilities):
        """Should calculate confidence metrics."""
        metrics = calculate_confidence_metrics(sample_probabilities)
        
        assert 'mean_confidence' in metrics
        assert 'std_confidence' in metrics
        assert 'min_confidence' in metrics
        assert 'max_confidence' in metrics
        
        # Confidence should be in [0, 1]
        assert 0 <= metrics['mean_confidence'] <= 1
        assert 0 <= metrics['min_confidence'] <= 1
        assert 0 <= metrics['max_confidence'] <= 1