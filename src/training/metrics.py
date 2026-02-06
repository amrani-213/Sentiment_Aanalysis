"""
Evaluation Metrics for Sentiment Analysis
Comprehensive metrics for model evaluation
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, matthews_corrcoef
)
from typing import Dict, List, Tuple, Optional


def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None, average='macro'):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        y_prob: Prediction probabilities (numpy array, optional)
        class_names: List of class names (default: ['Negative', 'Neutral', 'Positive'])
        average: Averaging method for metrics ('macro', 'weighted', 'micro')
    
    Returns:
        Dict containing all metrics
    """
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Prepare metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'confusion_matrix': cm,
        'per_class': {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': precision_per_class[i] if i < len(precision_per_class) else 0,
            'recall': recall_per_class[i] if i < len(recall_per_class) else 0,
            'f1': f1_per_class[i] if i < len(f1_per_class) else 0,
            'support': np.sum(y_true == i)
        }
    
    # Add ROC-AUC if probabilities are provided
    if y_prob is not None:
        try:
            # One-vs-Rest ROC-AUC
            roc_auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            metrics['roc_auc_macro'] = roc_auc_ovr
            
            # Per-class ROC-AUC
            n_classes = y_prob.shape[1]
            roc_auc_per_class = []
            for i in range(n_classes):
                if np.sum(y_true == i) > 0:
                    roc_auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
                    roc_auc_per_class.append(roc_auc)
                else:
                    roc_auc_per_class.append(0.0)
            
            for i, class_name in enumerate(class_names):
                if i < len(roc_auc_per_class):
                    metrics['per_class'][class_name]['roc_auc'] = roc_auc_per_class[i]
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
    
    return metrics


def print_metrics(metrics, model_name='Model'):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        model_name: Name of the model
    """
    print("="*80)
    print(f"{model_name} - EVALUATION METRICS")
    print("="*80)
    
    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"  Precision (Weight): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (Weight):    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (Weight):  {metrics['f1_weighted']:.4f}")
    print(f"  MCC:                {metrics['mcc']:.4f}")
    
    if 'roc_auc_macro' in metrics:
        print(f"  ROC-AUC (Macro):    {metrics['roc_auc_macro']:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<15} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} "
              f"{class_metrics['f1']:<12.4f} "
              f"{class_metrics['support']:<10}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("-"*80)
    cm = metrics['confusion_matrix']
    print(cm)
    print()


def calculate_top_k_accuracy(y_true, y_prob, k=2):
    """
    Calculate top-k accuracy
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        k: Top-k value
    
    Returns:
        Top-k accuracy score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Get top-k predictions
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def calculate_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)
    Measures how well predicted probabilities match actual outcomes
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Get predicted class and confidence
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    
    # Calculate ECE
    ece = 0.0
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(accuracies[mask])
            bin_confidence = np.mean(confidences[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)
    
    return ece


def calculate_confidence_metrics(y_prob):
    """
    Calculate metrics related to prediction confidence
    
    Args:
        y_prob: Prediction probabilities
    
    Returns:
        Dict with confidence metrics
    """
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Get max probabilities (confidence)
    confidences = np.max(y_prob, axis=1)
    
    metrics = {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'median_confidence': np.median(confidences)
    }
    
    return metrics


def compare_models(models_metrics: Dict[str, Dict]) -> Dict:
    """
    Compare multiple models and rank them
    
    Args:
        models_metrics: Dictionary mapping model names to their metrics
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'accuracy': {},
        'f1_macro': {},
        'f1_weighted': {},
        'precision_macro': {},
        'recall_macro': {}
    }
    
    for model_name, metrics in models_metrics.items():
        comparison['accuracy'][model_name] = metrics['accuracy']
        comparison['f1_macro'][model_name] = metrics['f1_macro']
        comparison['f1_weighted'][model_name] = metrics['f1_weighted']
        comparison['precision_macro'][model_name] = metrics['precision_macro']
        comparison['recall_macro'][model_name] = metrics['recall_macro']
    
    # Rank models
    rankings = {}
    for metric_name, scores in comparison.items():
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings[metric_name] = sorted_models
    
    return {
        'scores': comparison,
        'rankings': rankings
    }


def generate_classification_report(y_true, y_pred, class_names=None, output_dict=False):
    """
    Generate sklearn classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dict: Return as dictionary
    
    Returns:
        Classification report (string or dict)
    """
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    return classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


if __name__ == "__main__":
    # Test metrics functions
    print("="*80)
    print("TESTING METRICS FUNCTIONS")
    print("="*80)
    
    # Create dummy data
    n_samples = 100
    n_classes = 3
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize to probabilities
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Print metrics
    print_metrics(metrics, model_name='Test Model')
    
    # Test top-k accuracy
    top2_acc = calculate_top_k_accuracy(y_true, y_prob, k=2)
    print(f"\nTop-2 Accuracy: {top2_acc:.4f}")
    
    # Test calibration error
    ece = calculate_calibration_error(y_true, y_prob)
    print(f"Expected Calibration Error: {ece:.4f}")
    
    # Test confidence metrics
    conf_metrics = calculate_confidence_metrics(y_prob)
    print("\nConfidence Metrics:")
    for key, value in conf_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ All metrics functions tested successfully!")