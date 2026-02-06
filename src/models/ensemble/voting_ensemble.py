"""
Voting Ensemble for Sentiment Analysis
Combines predictions from multiple models using soft/hard voting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


class VotingEnsemble:
    """
    Ensemble multiple models using voting strategies
    
    Supports:
    - Hard voting (majority vote)
    - Soft voting (average probabilities)
    - Weighted voting (with model-specific weights)
    """
    
    def __init__(self, models, weights=None, voting='soft', device='cpu'):
        """
        Args:
            models: List of trained models
            weights: List of weights for each model (optional)
            voting: 'soft' or 'hard'
            device: Device to run inference on
        """
        self.models = models
        self.voting = voting
        self.device = device
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
            model.to(device)
        
        # Initialize weights
        if weights is None:
            self.weights = [1.0] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"\n{'='*80}")
        print(f"VOTING ENSEMBLE INITIALIZED")
        print(f"{'='*80}")
        print(f"Number of models: {len(self.models)}")
        print(f"Voting strategy: {self.voting}")
        print(f"Model weights: {[f'{w:.3f}' for w in self.weights]}")
    
    def predict_soft_voting(self, inputs, **kwargs):
        """
        Soft voting: Average predicted probabilities
        
        Args:
            inputs: Model inputs (batch)
            **kwargs: Additional arguments for models
        
        Returns:
            predictions: Predicted class indices
            probabilities: Averaged probability distributions
        """
        all_probs = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                # Get logits from model
                logits = model(inputs, **kwargs)
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=1)
                
                # Apply weight
                weighted_probs = probs * self.weights[i]
                all_probs.append(weighted_probs)
        
        # Average probabilities
        avg_probs = torch.stack(all_probs).sum(dim=0)
        
        # Get predictions
        predictions = torch.argmax(avg_probs, dim=1)
        
        return predictions, avg_probs
    
    def predict_hard_voting(self, inputs, **kwargs):
        """
        Hard voting: Majority vote on predicted classes
        
        Args:
            inputs: Model inputs (batch)
            **kwargs: Additional arguments for models
        
        Returns:
            predictions: Predicted class indices
            vote_counts: Vote distribution for each sample
        """
        batch_size = inputs.shape[0] if torch.is_tensor(inputs) else inputs['input_ids'].shape[0]
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                # Get logits from model
                logits = model(inputs, **kwargs)
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                all_predictions.append(preds.cpu().numpy())
        
        # Stack predictions: (num_models, batch_size)
        all_predictions = np.array(all_predictions)
        
        # Majority vote for each sample
        final_predictions = []
        vote_counts = []
        
        for i in range(batch_size):
            votes = all_predictions[:, i]
            
            # Count votes
            vote_counter = Counter(votes)
            
            # Get majority vote
            majority_vote = vote_counter.most_common(1)[0][0]
            final_predictions.append(majority_vote)
            vote_counts.append(dict(vote_counter))
        
        return torch.LongTensor(final_predictions), vote_counts
    
    def predict(self, inputs, **kwargs):
        """
        Predict using the configured voting strategy
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments for models
        
        Returns:
            predictions: Predicted class indices
            additional_info: Probabilities (soft) or vote counts (hard)
        """
        if self.voting == 'soft':
            return self.predict_soft_voting(inputs, **kwargs)
        elif self.voting == 'hard':
            return self.predict_hard_voting(inputs, **kwargs)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting}")
    
    def predict_with_confidence(self, inputs, **kwargs):
        """
        Predict with confidence scores
        
        Args:
            inputs: Model inputs
            **kwargs: Additional arguments
        
        Returns:
            predictions: Predicted classes
            confidences: Confidence scores
            all_probs: All model probabilities (for analysis)
        """
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(inputs, **kwargs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
        
        # Stack: (num_models, batch_size, num_classes)
        all_probs = torch.stack(all_probs)
        
        if self.voting == 'soft':
            # Average probabilities
            avg_probs = all_probs.mean(dim=0)
            predictions = torch.argmax(avg_probs, dim=1)
            confidences = torch.max(avg_probs, dim=1)[0]
        else:
            # Hard voting with confidence based on agreement
            predictions_per_model = torch.argmax(all_probs, dim=2)  # (num_models, batch_size)
            
            # Majority vote
            predictions = []
            confidences = []
            
            for i in range(predictions_per_model.shape[1]):
                votes = predictions_per_model[:, i].cpu().numpy()
                vote_counter = Counter(votes)
                majority_vote = vote_counter.most_common(1)[0][0]
                agreement = vote_counter[majority_vote] / len(self.models)
                
                predictions.append(majority_vote)
                confidences.append(agreement)
            
            predictions = torch.LongTensor(predictions)
            confidences = torch.FloatTensor(confidences)
            avg_probs = all_probs.mean(dim=0)
        
        return predictions, confidences, avg_probs


class WeightedVotingEnsemble(VotingEnsemble):
    """
    Weighted voting ensemble with automatic weight optimization
    """
    
    def __init__(self, models, val_loader, device='cpu', voting='soft'):
        """
        Args:
            models: List of trained models
            val_loader: Validation data loader for weight optimization
            device: Device to run on
            voting: 'soft' or 'hard'
        """
        # Initialize with equal weights
        super().__init__(models, weights=None, voting=voting, device=device)
        
        # Optimize weights on validation set
        self.optimize_weights(val_loader)
    
    def optimize_weights(self, val_loader):
        """
        Optimize ensemble weights using validation set performance
        
        Args:
            val_loader: Validation DataLoader
        """
        print(f"\nOptimizing ensemble weights on validation set...")
        
        # Get individual model accuracies
        model_accuracies = []
        
        for i, model in enumerate(self.models):
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if 'input_ids' in batch:
                        # Transformer model
                        inputs = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        logits = model(inputs, attention_mask=attention_mask)
                    else:
                        # Standard model
                        inputs = batch['text'].to(self.device)
                        logits = model(inputs, sentiment_scores=batch['sentiment_score'].to(self.device))
                    
                    labels = batch['label'].to(self.device)
                    preds = torch.argmax(logits, dim=1)
                    
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            accuracy = correct / total
            model_accuracies.append(accuracy)
            print(f"  Model {i+1} accuracy: {accuracy:.4f}")
        
        # Set weights proportional to accuracy
        total_accuracy = sum(model_accuracies)
        self.weights = [acc / total_accuracy for acc in model_accuracies]
        
        print(f"\nOptimized weights: {[f'{w:.3f}' for w in self.weights]}")


def create_voting_ensemble(
    models,
    voting='soft',
    weights=None,
    device='cpu'
):
    """
    Factory function to create voting ensemble
    
    Args:
        models: List of trained models
        voting: 'soft' or 'hard'
        weights: Optional model weights
        device: Device to run on
    
    Returns:
        VotingEnsemble instance
    """
    ensemble = VotingEnsemble(
        models=models,
        weights=weights,
        voting=voting,
        device=device
    )
    
    return ensemble


def create_weighted_ensemble(
    models,
    val_loader,
    voting='soft',
    device='cpu'
):
    """
    Factory function to create weighted voting ensemble
    
    Args:
        models: List of trained models
        val_loader: Validation DataLoader
        voting: 'soft' or 'hard'
        device: Device to run on
    
    Returns:
        WeightedVotingEnsemble instance
    """
    ensemble = WeightedVotingEnsemble(
        models=models,
        val_loader=val_loader,
        voting=voting,
        device=device
    )
    
    return ensemble


if __name__ == "__main__":
    print("="*80)
    print("TESTING VOTING ENSEMBLE")
    print("="*80)
    
    # This is a placeholder test
    # Real testing requires trained models
    
    print("\nVoting Ensemble module loaded successfully!")
    print("\nFeatures:")
    print("  ✅ Soft voting (average probabilities)")
    print("  ✅ Hard voting (majority vote)")
    print("  ✅ Weighted voting")
    print("  ✅ Automatic weight optimization")
    print("  ✅ Confidence scoring")
    
    print("\nTo use this module:")
    print("  1. Train multiple models")
    print("  2. Create ensemble with create_voting_ensemble()")
    print("  3. Use ensemble.predict() for inference")
    
    print("\n✅ Voting Ensemble module ready!")