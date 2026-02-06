"""
Comprehensive Model Evaluator
Handles evaluation of all models with detailed metrics and type-safe structures
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, TypedDict
from tqdm import tqdm
import json
import os


# ✅ CRITICAL FIX: Define precise nested types to prevent type checker confusion
class ErrorByClassStats(TypedDict):
    total: int
    errors: int
    error_rate: float

class PerClassMetrics(TypedDict):
    precision: float
    recall: float
    f1: float
    support: int

class ErrorExample(TypedDict):
    text: str
    true_label: str
    predicted_label: str
    confidence: float
    probabilities: List[float]


class ModelEvaluator:
    """
    Comprehensive evaluator for sentiment analysis models
    
    Features:
    - Accuracy, precision, recall, F1 scores
    - Confusion matrix
    - Per-class metrics
    - Prediction confidence analysis
    - Error analysis
    """
    
    def __init__(self, model, device='cpu', class_names: Optional[List[str]] = None):
        """
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
            class_names: List of class names (default: ['Negative', 'Neutral', 'Positive'])
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        if class_names is None:
            self.class_names = ['Negative', 'Neutral', 'Positive']
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def evaluate(self, data_loader, return_predictions: bool = True, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on data loader with type-safe structure
        
        Returns:
            Dictionary with strictly typed results
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"EVALUATING MODEL")
            print(f"{'='*80}")
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_texts = []
        
        with torch.no_grad():
            iterator = tqdm(data_loader, desc="Evaluating") if verbose else data_loader
            
            for batch in iterator:
                # Get inputs based on model type
                if 'input_ids' in batch:
                    # Transformer model
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    logits = self.model(inputs, attention_mask=attention_mask)
                else:
                    # Standard model
                    inputs = batch['text'].to(self.device)
                    sentiment_scores = batch['sentiment_score'].to(self.device)
                    logits = self.model(inputs, sentiment_scores=sentiment_scores)
                
                labels = batch['label'].to(self.device)
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                
                if 'raw_text' in batch:
                    all_texts.extend(batch['raw_text'])
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        from src.training.metrics import calculate_metrics
        
        metrics = calculate_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities,
            class_names=self.class_names
        )
        
        if verbose:
            from src.training.metrics import print_metrics
            print_metrics(metrics, model_name=self.__class__.__name__)
        
        # ✅ FIX: Build results with explicit type separation (no mixed-type dict ambiguity)
        results: Dict[str, Any] = {
            'metrics': metrics,
            'labels': all_labels,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
        }
        
        if return_predictions and all_texts:
            results['texts'] = all_texts
        
        return results
    
    def evaluate_with_error_analysis(self, data_loader, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate with detailed error analysis using type-safe structures
        
        Returns:
            Dictionary with metrics and strictly typed error analysis
        """
        results = self.evaluate(data_loader, return_predictions=True, verbose=verbose)
        
        # Extract results with explicit typing
        labels = results['labels']
        predictions = results['predictions']
        probabilities = results['probabilities']
        texts = results.get('texts', [])
        
        # Identify errors
        errors = labels != predictions
        error_indices = np.where(errors)[0]
        
        # ✅ FIX: Build error_analysis with EXPLICIT SEPARATION of nested structures
        # Avoid mixed-type dict initialization that confuses type checker
        
        # Top-level scalar metrics (all floats/ints)
        error_summary = {
            'total_samples': int(len(labels)),
            'total_errors': int(errors.sum()),
            'error_rate': float(errors.sum() / len(labels)),
        }
        
        # Nested structure 1: errors_by_class (dict of typed dicts)
        errors_by_class: Dict[str, ErrorByClassStats] = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            class_total = int(class_mask.sum())
            class_errors = int(errors[class_mask].sum())
            
            # ✅ Explicitly construct typed dict
            stats: ErrorByClassStats = {
                'total': class_total,
                'errors': class_errors,
                'error_rate': float(class_errors / max(class_total, 1))
            }
            errors_by_class[class_name] = stats
        
        # Nested structure 2: confusion_pairs (dict of ints)
        confusion_pairs: Dict[str, int] = {}
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    pair_count = int(((labels == i) & (predictions == j)).sum())
                    if pair_count > 0:
                        pair_name = f"{self.class_names[i]} → {self.class_names[j]}"
                        confusion_pairs[pair_name] = pair_count
        
        # Nested structure 3: error examples (list of typed dicts)
        error_examples: List[ErrorExample] = []
        if len(error_indices) > 0:
            error_confidences = probabilities[error_indices, predictions[error_indices]]
            top_error_indices = error_indices[np.argsort(error_confidences)[-10:]]
            
            for idx in reversed(top_error_indices):  # Highest confidence first
                example: ErrorExample = {
                    'text': str(texts[idx])[:200] if idx < len(texts) else "N/A",
                    'true_label': self.class_names[int(labels[idx])],
                    'predicted_label': self.class_names[int(predictions[idx])],
                    'confidence': float(probabilities[idx, predictions[idx]]),
                    'probabilities': [float(p) for p in probabilities[idx].tolist()]
                }
                error_examples.append(example)
        
        # ✅ FINAL FIX: Assemble error_analysis WITHOUT mixed-type initialization
        # Each component is built separately with explicit types, then combined
        error_analysis = {
            'summary': error_summary,          # Dict[str, Union[int, float]]
            'errors_by_class': errors_by_class,  # Dict[str, ErrorByClassStats]
            'confusion_pairs': confusion_pairs,  # Dict[str, int]
            'error_examples': error_examples     # List[ErrorExample]
        }
        
        results['error_analysis'] = error_analysis
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ERROR ANALYSIS")
            print(f"{'='*80}")
            print(f"Total errors: {error_summary['total_errors']} / {error_summary['total_samples']}")
            print(f"Error rate: {error_summary['error_rate']:.2%}")
            
            print(f"\nErrors by class:")
            for class_name, stats in errors_by_class.items():
                print(f"  {class_name}: {stats['errors']}/{stats['total']} ({stats['error_rate']:.2%})")
            
            print(f"\nTop confusion pairs:")
            sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
            for pair, count in sorted_pairs[:5]:
                print(f"  {pair}: {count}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], save_dir: str, model_name: str = 'model') -> None:
        """
        Save evaluation results to disk with type-safe serialization
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # ✅ FIX: Build metrics_to_save with explicit type separation
        metrics = results['metrics']
        
        # Scalar metrics (all floats)
        scalar_metrics = {
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'precision_weighted': float(metrics['precision_weighted']),
            'recall_weighted': float(metrics['recall_weighted']),
            'f1_weighted': float(metrics['f1_weighted']),
            'mcc': float(metrics['mcc']),
        }
        
        if 'roc_auc_macro' in metrics:
            scalar_metrics['roc_auc_macro'] = float(metrics['roc_auc_macro'])
        
        # Per-class metrics (nested dict with explicit typing)
        per_class_metrics: Dict[str, PerClassMetrics] = {}
        for class_name, class_metrics in metrics['per_class'].items():
            per_class_metrics[class_name] = {
                'precision': float(class_metrics['precision']),
                'recall': float(class_metrics['recall']),
                'f1': float(class_metrics['f1']),
                'support': int(class_metrics['support'])
            }
        
        # ✅ Assemble final metrics dict WITHOUT mixed-type initialization ambiguity
        metrics_to_save = {
            'scalar_metrics': scalar_metrics,
            'per_class': per_class_metrics,
        }
        
        metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"\n✅ Metrics saved to {metrics_path}")
        
        # Save confusion matrix
        cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.npy")
        np.save(cm_path, metrics['confusion_matrix'])
        
        # Save predictions
        predictions_path = os.path.join(save_dir, f"{model_name}_predictions.npz")
        np.savez(
            predictions_path,
            labels=results['labels'],
            predictions=results['predictions'],
            probabilities=results['probabilities']
        )
        
        print(f"✅ Predictions saved to {predictions_path}")
        
        # Save error analysis if available
        if 'error_analysis' in results:
            error_path = os.path.join(save_dir, f"{model_name}_error_analysis.json")
            
            # Convert numpy types to Python types for JSON serialization
            with open(error_path, 'w') as f:
                json.dump(results['error_analysis'], f, indent=2, default=str)
            
            print(f"✅ Error analysis saved to {error_path}")


class EnsembleEvaluator(ModelEvaluator):
    """
    Evaluator for ensemble models with diversity metrics
    """
    
    def __init__(self, ensemble, device='cpu', class_names: Optional[List[str]] = None):
        """
        Args:
            ensemble: Trained ensemble model
            device: Device to run on
            class_names: List of class names
        """
        super().__init__(ensemble, device, class_names)
        self.ensemble = ensemble
    
    def evaluate_with_diversity(self, data_loader, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate ensemble with diversity metrics
        
        Returns:
            Dictionary with metrics and diversity analysis
        """
        results = self.evaluate(data_loader, return_predictions=True, verbose=verbose)
        
        # Get individual model predictions for diversity analysis
        if hasattr(self.ensemble, 'models'):
            base_models = self.ensemble.models
        elif hasattr(self.ensemble, 'base_models'):
            base_models = self.ensemble.base_models
        else:
            print("Warning: Cannot compute diversity - no base models found")
            return results
        
        # Collect predictions from each base model
        all_base_predictions = []
        
        with torch.no_grad():
            for model in base_models:
                model_preds = []
                
                for batch in data_loader:
                    if 'input_ids' in batch:
                        inputs = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        logits = model(inputs, attention_mask=attention_mask)
                    else:
                        inputs = batch['text'].to(self.device)
                        sentiment_scores = batch['sentiment_score'].to(self.device)
                        logits = model(inputs, sentiment_scores=sentiment_scores)
                    
                    preds = torch.argmax(logits, dim=1)
                    model_preds.extend(preds.cpu().numpy())
                
                all_base_predictions.append(np.array(model_preds))
        
        # Calculate diversity metrics
        all_base_predictions = np.array(all_base_predictions)  # (num_models, num_samples)
        
        # Agreement rate
        agreement_matrix = np.zeros((len(base_models), len(base_models)))
        for i in range(len(base_models)):
            for j in range(len(base_models)):
                agreement_matrix[i, j] = float((all_base_predictions[i] == all_base_predictions[j]).mean())
        
        avg_pairwise_agreement = float(
            (agreement_matrix.sum() - len(base_models)) / (len(base_models) * (len(base_models) - 1))
        )
        
        # Diversity score (1 - agreement)
        diversity_score = 1.0 - avg_pairwise_agreement
        
        results['diversity_analysis'] = {
            'num_base_models': len(base_models),
            'avg_pairwise_agreement': avg_pairwise_agreement,
            'diversity_score': diversity_score,
            'agreement_matrix': agreement_matrix.tolist()
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ENSEMBLE DIVERSITY ANALYSIS")
            print(f"{'='*80}")
            print(f"Number of base models: {len(base_models)}")
            print(f"Average pairwise agreement: {avg_pairwise_agreement:.4f}")
            print(f"Diversity score: {diversity_score:.4f}")
        
        return results


if __name__ == "__main__":
    print("="*80)
    print("TESTING MODEL EVALUATOR")
    print("="*80)
    
    print("\nModelEvaluator module loaded successfully!")
    print("\nFeatures:")
    print("  ✅ Comprehensive metrics (accuracy, F1, precision, recall)")
    print("  ✅ Confusion matrix analysis")
    print("  ✅ Per-class metrics")
    print("  ✅ Error analysis with type-safe structures")
    print("  ✅ Confidence analysis")
    print("  ✅ Ensemble diversity metrics")
    print("  ✅ Save/load results")
    
    print("\nTo use this module:")
    print("  1. Create evaluator: evaluator = ModelEvaluator(model)")
    print("  2. Run evaluation: results = evaluator.evaluate(test_loader)")
    print("  3. Save results: evaluator.save_results(results, 'results/')")
    
    print("\n✅ Evaluator module ready!")