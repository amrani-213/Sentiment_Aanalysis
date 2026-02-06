"""
Helper utilities for model training and evaluation
UPDATED: Enhanced save_model() with metadata support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json


def count_parameters(model):
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def save_model(model, optimizer, epoch, metrics, save_path, 
               model_type=None, model_config=None):
    """
    Save model with complete metadata for reliable loading
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (can be None)
        epoch: Current epoch
        metrics: Training/validation metrics dict
        save_path: Path to save checkpoint
        model_type: Type of model ('bilstm', 'transformer', 'fasttext', 'roberta', 'bertweet')
        model_config: Model configuration dict with all hyperparameters
    
    Example:
        >>> model, config = create_bilstm_model(vocab_size=10000, embedding_dim=300)
        >>> save_model(model, optimizer, epoch, metrics, 'model.pt',
        ...            model_type='bilstm', model_config=config)
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint with metadata
    checkpoint = {
        # Model state
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'metrics': metrics,
        
        # Model metadata (NEW)
        'model_type': model_type,
        'model_config': model_config,
        
        # Version info
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        'save_path': str(save_path)
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    # Save metadata separately for quick inspection (without loading weights)
    if model_type is not None:
        metadata_path = str(save_path).replace('.pt', '_metadata.json')
        metadata = {
            'model_type': model_type,
            'model_config': model_config,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp'],
            'pytorch_version': checkpoint['pytorch_version']
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_model(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    Load model from checkpoint (legacy function - consider using ModelLoader)
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer to load state into (optional)
        device: Device to load on
    
    Returns:
        model: Model with loaded weights
        epoch: Saved epoch number
        metrics: Saved metrics dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        if checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return model, epoch, metrics


def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=0.0, **kwargs):
    """
    Get optimizer by name
    
    Args:
        model: PyTorch model
        optimizer_name: 'adam', 'adamw', or 'sgd'
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(optimizer, scheduler_name='none', num_training_steps=None, 
                  num_warmup_steps=None, **kwargs):
    """
    Get learning rate scheduler by name
    
    Args:
        optimizer: Optimizer
        scheduler_name: 'none', 'linear', 'cosine', 'step', 'reduce_on_plateau'
        num_training_steps: Total training steps (for linear/cosine)
        num_warmup_steps: Warmup steps (for linear/cosine)
        **kwargs: Additional scheduler arguments
    
    Returns:
        Scheduler instance or None
    """
    if scheduler_name.lower() == 'none':
        return None
    
    elif scheduler_name.lower() == 'linear':
        if num_training_steps is None or num_warmup_steps is None:
            raise ValueError("Linear scheduler requires num_training_steps and num_warmup_steps")
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    elif scheduler_name.lower() == 'cosine':
        if num_training_steps is None or num_warmup_steps is None:
            raise ValueError("Cosine scheduler requires num_training_steps and num_warmup_steps")
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name.lower() == 'reduce_on_plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' (whether lower or higher is better)
        verbose: Print messages
    
    Example:
        >>> early_stopping = EarlyStopping(patience=5, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stopping(val_loss):
        ...         print("Early stopping triggered!")
        ...         break
    """
    
    def __init__(self, patience=5, min_delta=0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Check if should stop based on new score
        
        Args:
            score: Current validation score
        
        Returns:
            bool: True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"Validation score improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
        
        return self.early_stop


def format_time(seconds):
    """
    Format seconds into human-readable string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    
    Example:
        >>> format_time(3665)
        '1h 1m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def plot_history(history, save_path=None, figsize=(12, 5)):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot (optional)
        figsize: Figure size
    
    Example:
        >>> history = {
        ...     'train_loss': [0.5, 0.4, 0.3],
        ...     'val_loss': [0.6, 0.5, 0.4],
        ...     'train_acc': [0.7, 0.8, 0.85],
        ...     'val_acc': [0.65, 0.75, 0.8]
        ... }
        >>> plot_history(history, 'training_curves.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test the new save_model function
    print("="*80)
    print("TESTING ENHANCED SAVE_MODEL")
    print("="*80)
    
    # Create a dummy model
    model = nn.Linear(10, 3)
    optimizer = optim.Adam(model.parameters())
    
    # Model config example
    config = {
        'vocab_size': 10000,
        'embedding_dim': 300,
        'hidden_dim': 256,
        'num_classes': 3
    }
    
    metrics = {
        'accuracy': 0.85,
        'f1_macro': 0.83
    }
    
    # Save with metadata
    save_model(
        model, optimizer, epoch=10, metrics=metrics,
        save_path='test_model.pt',
        model_type='bilstm',
        model_config=config
    )
    
    print("\n✅ Model saved with metadata!")
    print("Files created:")
    print("  - test_model.pt (checkpoint)")
    print("  - test_model_metadata.json (quick inspection)")
    
    # Check metadata file
    with open('test_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"\nMetadata contents:")
    print(json.dumps(metadata, indent=2))
    
    # Clean up
    import os
    os.remove('test_model.pt')
    os.remove('test_model_metadata.json')
    print("\n✅ Test complete!")