"""
Custom Loss Functions for Sentiment Analysis
Includes Focal Loss, Label Smoothing, and Weighted Cross-Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: Lin et al. (2017) - Focal Loss for Dense Object Detection
    
    Args:
        alpha: Weighting factor for each class (list or tensor)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.FloatTensor(alpha)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class indices
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get the probability of the target class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
        pt = (probs * targets_one_hot).sum(dim=1)  # p_t in the paper
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss
    Prevents overconfidence and improves model calibration
    
    Args:
        num_classes: Number of classes
        smoothing: Smoothing parameter (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class indices
        """
        # Get log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Calculate loss
        loss = torch.sum(-true_dist * log_probs, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for class imbalance
    
    Args:
        class_weights: Weights for each class (list, array, or tensor)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, class_weights, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        if isinstance(class_weights, (list, np.ndarray)):
            class_weights = torch.FloatTensor(class_weights)
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class indices
        """
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        return F.cross_entropy(
            inputs, 
            targets, 
            weight=self.class_weights,
            reduction=self.reduction
        )


class CombinedLoss(nn.Module):
    """
    Combined loss function (e.g., Focal + Label Smoothing)
    
    Args:
        focal_weight: Weight for focal loss (default: 0.5)
        smoothing_weight: Weight for label smoothing (default: 0.5)
        alpha: Alpha parameter for focal loss
        gamma: Gamma parameter for focal loss
        smoothing: Smoothing parameter for label smoothing
        num_classes: Number of classes
    """
    
    def __init__(self, focal_weight=0.5, smoothing_weight=0.5, 
                 alpha=None, gamma=2.0, smoothing=0.1, num_classes=3):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.smoothing_weight = smoothing_weight
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smoothing_loss = LabelSmoothingLoss(
            num_classes=num_classes, 
            smoothing=smoothing
        )
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        smoothing = self.smoothing_loss(inputs, targets)
        
        return self.focal_weight * focal + self.smoothing_weight * smoothing


def compute_class_weights(labels, method='inverse_freq', num_classes=3):
    """
    Compute class weights for handling imbalanced datasets
    
    Args:
        labels: Array or tensor of class labels
        method: 'inverse_freq' or 'effective_num'
        num_classes: Number of classes
    
    Returns:
        torch.Tensor: Class weights
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Count samples per class
    class_counts = np.bincount(labels, minlength=num_classes)
    
    if method == 'inverse_freq':
        # Inverse frequency weighting
        total_samples = len(labels)
        weights = total_samples / (num_classes * class_counts)
        
    elif method == 'effective_num':
        # Effective number of samples (Cui et al., 2019)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * num_classes
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return torch.FloatTensor(weights)


def get_loss_function(loss_type='cross_entropy', class_weights=None, 
                      num_classes=3, **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_type: 'cross_entropy', 'focal', 'label_smoothing', 'weighted', or 'combined'
        class_weights: Class weights for weighted/focal loss
        num_classes: Number of classes
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        Loss function
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    
    elif loss_type == 'weighted':
        if class_weights is None:
            raise ValueError("class_weights must be provided for weighted loss")
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    
    elif loss_type == 'combined':
        gamma = kwargs.get('gamma', 2.0)
        smoothing = kwargs.get('smoothing', 0.1)
        focal_weight = kwargs.get('focal_weight', 0.5)
        smoothing_weight = kwargs.get('smoothing_weight', 0.5)
        
        return CombinedLoss(
            focal_weight=focal_weight,
            smoothing_weight=smoothing_weight,
            alpha=class_weights,
            gamma=gamma,
            smoothing=smoothing,
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("="*80)
    print("TESTING LOSS FUNCTIONS")
    print("="*80)
    
    # Create dummy data
    batch_size = 32
    num_classes = 3
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test standard cross-entropy
    ce_loss = nn.CrossEntropyLoss()
    print(f"\nCross-Entropy Loss: {ce_loss(inputs, targets).item():.4f}")
    
    # Test focal loss
    focal_loss = FocalLoss(gamma=2.0)
    print(f"Focal Loss (γ=2.0): {focal_loss(inputs, targets).item():.4f}")
    
    # Test label smoothing
    ls_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    print(f"Label Smoothing Loss (ε=0.1): {ls_loss(inputs, targets).item():.4f}")
    
    # Test weighted loss
    class_weights = compute_class_weights(targets.numpy())
    print(f"\nClass Weights: {class_weights.numpy()}")
    weighted_loss = WeightedCrossEntropyLoss(class_weights)
    print(f"Weighted Cross-Entropy Loss: {weighted_loss(inputs, targets).item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedLoss(
        focal_weight=0.5,
        smoothing_weight=0.5,
        gamma=2.0,
        smoothing=0.1,
        num_classes=num_classes
    )
    print(f"Combined Loss: {combined_loss(inputs, targets).item():.4f}")
    
    print("\n✅ All loss functions tested successfully!")