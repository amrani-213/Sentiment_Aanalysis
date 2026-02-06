"""
Stacking Ensemble for Sentiment Analysis
Uses meta-learner to combine base model predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with meta-learner
    
    Architecture:
    1. Base models make predictions (frozen)
    2. Meta-learner learns to combine base predictions
    3. Final prediction from meta-learner
    """
    
    def __init__(
        self,
        base_models,
        num_classes=3,
        meta_model_type='mlp',
        meta_hidden_dim=64,
        device='cpu'
    ):
        """
        Args:
            base_models: List of trained base models
            num_classes: Number of output classes
            meta_model_type: 'mlp', 'linear', or 'logistic'
            meta_hidden_dim: Hidden dimension for MLP meta-learner
            device: Device to run on
        """
        super(StackingEnsemble, self).__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.num_base_models = len(base_models)
        self.num_classes = num_classes
        self.device = device
        
        # Freeze base models
        for model in self.base_models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        
        # Create meta-learner
        # Input: predictions from all base models (num_base_models * num_classes)
        meta_input_dim = self.num_base_models * num_classes
        
        if meta_model_type == 'linear':
            self.meta_learner = nn.Linear(meta_input_dim, num_classes)
            
        elif meta_model_type == 'logistic':
            self.meta_learner = nn.Sequential(
                nn.Linear(meta_input_dim, num_classes),
                nn.Softmax(dim=1)
            )
            
        elif meta_model_type == 'mlp':
            self.meta_learner = nn.Sequential(
                nn.Linear(meta_input_dim, meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(meta_hidden_dim, meta_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(meta_hidden_dim // 2, num_classes)
            )
        else:
            raise ValueError(f"Unknown meta_model_type: {meta_model_type}")
        
        self.meta_learner.to(device)
        
        print(f"\n{'='*80}")
        print(f"STACKING ENSEMBLE INITIALIZED")
        print(f"{'='*80}")
        print(f"Number of base models: {self.num_base_models}")
        print(f"Meta-learner type: {meta_model_type}")
        print(f"Meta-learner input dim: {meta_input_dim}")
        print(f"Meta-learner hidden dim: {meta_hidden_dim if meta_model_type == 'mlp' else 'N/A'}")
    
    def get_base_predictions(self, inputs, **kwargs):
        """
        Get predictions from all base models
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments
        
        Returns:
            Concatenated predictions from all base models
        """
        all_probs = []
        
        with torch.no_grad():
            for model in self.base_models:
                logits = model(inputs, **kwargs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
        
        # Concatenate: (batch_size, num_base_models * num_classes)
        concatenated = torch.cat(all_probs, dim=1)
        
        return concatenated
    
    def forward(self, inputs, **kwargs):
        """
        Forward pass through stacking ensemble
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments for base models
        
        Returns:
            Logits from meta-learner
        """
        # Get base model predictions
        base_predictions = self.get_base_predictions(inputs, **kwargs)
        
        # Meta-learner prediction
        logits = self.meta_learner(base_predictions)
        
        return logits
    
    def predict(self, inputs, **kwargs):
        """
        Make predictions
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments
        
        Returns:
            predictions: Predicted class indices
            probabilities: Prediction probabilities
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(inputs, **kwargs)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        return predictions, probs
    
    def train_meta_learner(
        self,
        train_loader,
        val_loader,
        epochs=20,
        lr=0.001,
        verbose=True
    ):
        """
        Train the meta-learner
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print training progress
        
        Returns:
            training_history: Dictionary with losses and accuracies
        """
        print(f"\n{'='*80}")
        print(f"TRAINING META-LEARNER")
        print(f"{'='*80}")
        
        # Optimizer and loss
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
            
            for batch in pbar:
                # Get inputs based on model type
                if 'input_ids' in batch:
                    inputs = batch['input_ids'].to(self.device)
                    kwargs = {'attention_mask': batch['attention_mask'].to(self.device)}
                else:
                    inputs = batch['text'].to(self.device)
                    kwargs = {'sentiment_scores': batch['sentiment_score'].to(self.device)}
                
                labels = batch['label'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.forward(inputs, **kwargs)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{train_correct/train_total:.4f}'
                    })
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.meta_learner.state_dict().copy()
                if verbose:
                    print(f"  ✅ New best validation accuracy: {best_val_acc:.4f}")
        
        # Restore best model
        if best_state is not None:
            self.meta_learner.load_state_dict(best_state)
            print(f"\n✅ Meta-learner trained! Best val accuracy: {best_val_acc:.4f}")
        
        return history
    
    def evaluate(self, data_loader, criterion):
        """
        Evaluate the ensemble
        
        Args:
            data_loader: DataLoader
            criterion: Loss function
        
        Returns:
            loss: Average loss
            accuracy: Accuracy
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Get inputs
                if 'input_ids' in batch:
                    inputs = batch['input_ids'].to(self.device)
                    kwargs = {'attention_mask': batch['attention_mask'].to(self.device)}
                else:
                    inputs = batch['text'].to(self.device)
                    kwargs = {'sentiment_scores': batch['sentiment_score'].to(self.device)}
                
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.forward(inputs, **kwargs)
                loss = criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy


def create_stacking_ensemble(
    base_models,
    train_loader,
    val_loader,
    num_classes=3,
    meta_model_type='mlp',
    meta_hidden_dim=64,
    epochs=20,
    lr=0.001,
    device='cpu',
    verbose=True
):
    """
    Factory function to create and train stacking ensemble
    
    Args:
        base_models: List of trained base models
        train_loader: Training DataLoader (for meta-learner)
        val_loader: Validation DataLoader
        num_classes: Number of classes
        meta_model_type: Type of meta-learner ('mlp', 'linear', 'logistic')
        meta_hidden_dim: Hidden dimension for MLP
        epochs: Training epochs for meta-learner
        lr: Learning rate
        device: Device to run on
        verbose: Print training progress
    
    Returns:
        Trained StackingEnsemble
    """
    # Create ensemble
    ensemble = StackingEnsemble(
        base_models=base_models,
        num_classes=num_classes,
        meta_model_type=meta_model_type,
        meta_hidden_dim=meta_hidden_dim,
        device=device
    )
    
    # Train meta-learner
    history = ensemble.train_meta_learner(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        verbose=verbose
    )
    
    return ensemble, history


if __name__ == "__main__":
    print("="*80)
    print("TESTING STACKING ENSEMBLE")
    print("="*80)
    
    print("\nStacking Ensemble module loaded successfully!")
    print("\nFeatures:")
    print("  ✅ Meta-learner (MLP, Linear, Logistic)")
    print("  ✅ Automatic training on validation set")
    print("  ✅ Base model prediction combination")
    print("  ✅ Frozen base models (efficient)")
    
    print("\nArchitecture:")
    print("  1. Base models → Probability predictions")
    print("  2. Concatenate all predictions")
    print("  3. Meta-learner → Final prediction")
    
    print("\nTo use this module:")
    print("  1. Train base models separately")
    print("  2. Create stacking ensemble with create_stacking_ensemble()")
    print("  3. Meta-learner trains on base model outputs")
    print("  4. Use ensemble.predict() for inference")
    
    print("\n✅ Stacking Ensemble module ready!")