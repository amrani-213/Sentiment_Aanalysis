"""
Trainer Module for Sentiment Analysis
Handles model training with advanced features like gradient accumulation,
mixed precision, warmup scheduling, and early stopping

FIXED: evaluate() method now returns 'loss' in metrics dictionary
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time

from .metrics import calculate_metrics


class Trainer:
    """
    Universal trainer for all sentiment analysis models
    
    Features:
    - Gradient accumulation for large effective batch sizes
    - Mixed precision training (FP16)
    - Learning rate warmup and scheduling
    - Early stopping
    - Gradient clipping
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        device='cuda',
        logger=None,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        use_amp=False
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            loss_fn: Loss function
            device: Device to train on
            logger: Logger instance
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        
        # Initialize AMP scaler if needed
        if use_amp:
            self.scaler = GradScaler()
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def train(
        self,
        num_epochs,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        scheduler_type='cosine',
        early_stopping_patience=None,
        save_path=None,
        verbose=True
    ):
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            warmup_ratio: Ratio of total steps to use for warmup
            scheduler_type: Type of LR scheduler ('cosine', 'linear', or None)
            early_stopping_patience: Patience for early stopping (None to disable)
            save_path: Path to save best model
            verbose: Whether to print progress
        
        Returns:
            trained_model: Best model
            history: Training history
        """
        # Create optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create scheduler
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        if scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = None
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Early stopping
        patience_counter = 0
        best_epoch = 0
        
        if self.logger:
            self.logger.info(f"\nStarting training for {num_epochs} epochs")
            self.logger.info(f"Learning rate: {learning_rate}")
            self.logger.info(f"Warmup steps: {warmup_steps}")
            self.logger.info(f"Total steps: {total_steps}")
        
        # Training loop
        for epoch in range(num_epochs):
            if self.logger:
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                self.logger.info(f"{'='*80}")
            
            # Train
            train_metrics = self.train_epoch(optimizer, scheduler, verbose=verbose)
            
            # Validate
            val_metrics = self.validate_epoch(verbose=verbose)
            
            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            if scheduler:
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Log metrics
            if self.logger:
                self.logger.info(f"\nEpoch {epoch + 1} Results:")
                self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                self.logger.info(f"  Val Loss:   {val_metrics['loss']:.4f}, Val Acc:   {val_metrics['accuracy']:.4f}")
            elif verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs}:")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                best_epoch = epoch + 1
                patience_counter = 0
                
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics, optimizer)
                    if self.logger:
                        self.logger.info(f"  [BEST] New best model! Val Loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping check
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if self.logger:
                    self.logger.info(f"\n[WARNING]  Early stopping triggered after {epoch + 1} epochs")
                    self.logger.info(f"Best epoch: {best_epoch}, Best val loss: {self.best_val_loss:.4f}")
                break
        
        # Load best model if saved
        if save_path and Path(save_path).exists():
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.logger:
                self.logger.info(f"\n[SUCCESS] Loaded best model from epoch {best_epoch}")
        
        return self.model, history
    
    def train_epoch(self, optimizer, scheduler=None, verbose=True):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training") if verbose else self.train_loader
        
        for step, batch in enumerate(pbar):
            # Get inputs based on batch type
            if 'input_ids' in batch:
                # Transformer model
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(inputs, attention_mask=attention_mask)
                        loss = self.loss_fn(logits, labels)
                else:
                    logits = self.model(inputs, attention_mask=attention_mask)
                    loss = self.loss_fn(logits, labels)
            else:
                # Standard model
                inputs = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # ✅ CRITICAL FIX: Only pass sentiment_scores to models that accept it
                if hasattr(self.model, 'use_sentiment_features') and self.model.use_sentiment_features:
                    sentiment_scores = batch['sentiment_score'].to(self.device)
                    if self.use_amp:
                        with autocast():
                            logits = self.model(inputs, sentiment_scores=sentiment_scores)
                            loss = self.loss_fn(logits, labels)
                    else:
                        logits = self.model(inputs, sentiment_scores=sentiment_scores)
                        loss = self.loss_fn(logits, labels)
                else:
                    if self.use_amp:
                        with autocast():
                            logits = self.model(inputs)
                            loss = self.loss_fn(logits, labels)
                    else:
                        logits = self.model(inputs)
                        loss = self.loss_fn(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'acc': f'{correct/total:.4f}'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self, verbose=True):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc="Validation") if verbose else self.val_loader
        
        with torch.no_grad():
            for batch in pbar:
                # Get inputs based on batch type
                if 'input_ids' in batch:
                    # Transformer model
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    logits = self.model(inputs, attention_mask=attention_mask)
                else:
                    # Standard model
                    inputs = batch['text'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # ✅ CRITICAL FIX: Only pass sentiment_scores to models that accept it
                    if hasattr(self.model, 'use_sentiment_features') and self.model.use_sentiment_features:
                        sentiment_scores = batch['sentiment_score'].to(self.device)
                        logits = self.model(inputs, sentiment_scores=sentiment_scores)
                    else:
                        logits = self.model(inputs)
                
                loss = self.loss_fn(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{correct/total:.4f}'
                    })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def evaluate(self, test_loader, return_predictions=False):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test DataLoader
            return_predictions: Whether to return predictions
        
        Returns:
            metrics: Dictionary with evaluation metrics (including 'loss')
            predictions (optional): Predictions and labels
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0  # ✅ ADDED: Track loss during evaluation
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Get inputs based on batch type
                if 'input_ids' in batch:
                    # Transformer model
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    logits = self.model(inputs, attention_mask=attention_mask)
                else:
                    # Standard model
                    inputs = batch['text'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # ✅ CRITICAL FIX: Only pass sentiment_scores to models that accept it
                    if hasattr(self.model, 'use_sentiment_features') and self.model.use_sentiment_features:
                        sentiment_scores = batch['sentiment_score'].to(self.device)
                        logits = self.model(inputs, sentiment_scores=sentiment_scores)
                    else:
                        logits = self.model(inputs)
                
                # ✅ ADDED: Calculate loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        
        # ✅ CRITICAL FIX: Add loss to metrics dictionary
        metrics['loss'] = total_loss / len(test_loader)
        
        if return_predictions:
            return metrics, {
                'predictions': all_preds,
                'labels': all_labels,
                'probabilities': all_probs
            }
        
        return metrics
    
    def save_checkpoint(self, path, epoch, metrics, optimizer=None):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


def create_optimizer(model, optimizer_type='adamw', learning_rate=1e-3, weight_decay=0.01, **kwargs):
    """
    Create optimizer
    
    Args:
        model: PyTorch model
        optimizer_type: 'adam', 'adamw', or 'sgd'
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer, scheduler_type='cosine', num_training_steps=None, 
                     num_warmup_steps=None, **kwargs):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        scheduler_type: 'cosine', 'linear', 'step', or None
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        **kwargs: Additional scheduler arguments
    
    Returns:
        Scheduler or None
    """
    if scheduler_type is None or scheduler_type.lower() == 'none':
        return None
    
    if scheduler_type.lower() == 'cosine':
        if num_training_steps is None or num_warmup_steps is None:
            raise ValueError("Cosine scheduler requires num_training_steps and num_warmup_steps")
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    elif scheduler_type.lower() == 'linear':
        if num_training_steps is None or num_warmup_steps is None:
            raise ValueError("Linear scheduler requires num_training_steps and num_warmup_steps")
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    elif scheduler_type.lower() == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    
    elif scheduler_type.lower() == 'reduce_on_plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    print("="*80)
    print("TRAINER MODULE")
    print("="*80)
    print("\nFeatures:")
    print("  ✅ Gradient accumulation")
    print("  ✅ Mixed precision training (AMP)")
    print("  ✅ Learning rate warmup and scheduling")
    print("  ✅ Early stopping")
    print("  ✅ Gradient clipping")
    print("  ✅ Model checkpointing")
    print("  ✅ Comprehensive metrics tracking")
    print("  ✅ FIXED: evaluate() now returns 'loss' in metrics")
    print("\nReady to use!")