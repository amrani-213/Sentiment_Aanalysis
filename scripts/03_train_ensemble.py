"""
FIXED: scripts/03_train_ensemble.py
Fixes:
1. Legacy model loading with hyperparameter inference from checkpoints
2. Conditional sentiment_scores passing based on model type
"""

import torch
import argparse
import os
import pickle
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import EnhancedTextPreprocessor, prepare_data
from src.data.dataset import create_data_loaders
from src.models.ensemble.voting_ensemble import create_voting_ensemble, create_weighted_ensemble
from src.models.ensemble.stacking_ensemble import create_stacking_ensemble
from src.evaluation.evaluator import ModelEvaluator as Evaluator
from src.evaluation.visualizer import SentimentVisualizer as Visualizer
from src.utils.config import set_seed, get_device
from src.utils.model_loader import ModelLoader


def infer_model_hyperparams(state_dict, model_type):
    """
    Infer model hyperparameters from state dict
    
    Args:
        state_dict: Model state dictionary
        model_type: Type of model ('fasttext', 'bilstm', 'transformer', etc.)
    
    Returns:
        Dictionary of hyperparameters
    """
    params = {}
    
    if model_type == 'fasttext':
        params['vocab_size'] = state_dict['word_embedding.weight'].shape[0]
        params['embedding_dim'] = state_dict['word_embedding.weight'].shape[1]
        
    elif model_type == 'bilstm':
        # LSTM weight_ih_l0 shape is [4*hidden_size, embedding_dim]
        params['hidden_dim'] = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        params['embedding_dim'] = state_dict['embedding.weight'].shape[1]
        params['vocab_size'] = state_dict['embedding.weight'].shape[0]
        
        # Count LSTM layers by checking for weight_ih_l{N}
        num_layers = 0
        for key in state_dict.keys():
            if 'lstm.weight_ih_l' in key:
                layer_num = int(key.split('_l')[1][0]) + 1
                num_layers = max(num_layers, layer_num)
        params['num_layers'] = num_layers
        
        # Check if using sentiment features
        params['use_sentiment_features'] = 'sentiment_fc.weight' in state_dict
        
    elif model_type == 'transformer':
        params['d_model'] = state_dict['pos_encoding.pe'].shape[2]
        params['vocab_size'] = state_dict['embedding.weight'].shape[0]
        params['max_len'] = state_dict['pos_encoding.pe'].shape[1]
        
        # Count transformer layers
        num_layers = 0
        for key in state_dict.keys():
            if 'encoder_layers.' in key:
                layer_num = int(key.split('.')[1]) + 1
                num_layers = max(num_layers, layer_num)
        params['num_layers'] = num_layers
    
    return params


def load_trained_models(model_paths, device):
    """
    Load trained models with automatic type detection
    """
    models = []
    model_types = []  # Track model types for conditional argument passing
    
    for model_path in model_paths:
        print(f"Loading model from {model_path}...")
        
        try:
            # Use ModelLoader for automatic detection
            model = ModelLoader.load_model(model_path, device=device)
            models.append(model)
            
            # Get model info
            info = ModelLoader.get_model_info(model_path)
            model_type = info['model_type']
            model_types.append(model_type)
            
            print(f"  ✅ Loaded {model_type} model")
            if 'metrics' in info and info['metrics']:
                acc = info['metrics'].get('accuracy', 'N/A')
                print(f"     Accuracy: {acc:.4f}" if isinstance(acc, float) else f"     Accuracy: {acc}")
            
        except ValueError as e:
            # Handle legacy checkpoints
            print(f"  ⚠️  Warning: {e}")
            print(f"  Attempting legacy loading...")
            
            model, model_type = load_model_legacy(model_path, device)
            if model:
                models.append(model)
                model_types.append(model_type)
                print(f"  ✅ Loaded {model_type} using legacy method")
            else:
                print(f"  ❌ Failed to load {model_path}")
                continue
                
        except Exception as e:
            print(f"  ❌ Error loading {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not models:
        raise ValueError("No models could be loaded! Please check model paths.")
    
    return models, model_types


def load_model_legacy(model_path, device):
    """
    Legacy model loading with hyperparameter inference
    """
    import warnings
    warnings.warn("Using legacy model loading. Please re-train models with new save format.")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_path_lower = str(model_path).lower()
    
    # Get state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    try:
        # Determine model type from path
        if 'bertweet' in model_path_lower:
            from src.models.pretrained.bertweet import create_bertweet_model
            model = create_bertweet_model(num_classes=3)
            model_type = 'bertweet'
            
        elif 'roberta' in model_path_lower:
            from src.models.pretrained.roberta import create_roberta_model
            model = create_roberta_model(num_classes=3)
            model_type = 'roberta'
            
        elif 'bilstm' in model_path_lower:
            from src.models.baseline.bilstm_attention import create_bilstm_model
            
            # Infer hyperparameters
            params = infer_model_hyperparams(state_dict, 'bilstm')
            print(f"  Inferred BiLSTM params: {params}")
            
            model, _ = create_bilstm_model(
                vocab_size=params['vocab_size'],
                embedding_dim=params['embedding_dim'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                num_classes=3,
                use_sentiment_features=params['use_sentiment_features']
            )
            model_type = 'bilstm'
            
        elif 'transformer' in model_path_lower:
            from src.models.baseline.custom_transformer import create_custom_transformer
            
            # Infer hyperparameters
            params = infer_model_hyperparams(state_dict, 'transformer')
            print(f"  Inferred Transformer params: {params}")
            
            model, _ = create_custom_transformer(
                vocab_size=params['vocab_size'],
                d_model=params['d_model'],
                num_classes=3,
                max_len=params['max_len'],
                num_layers=params['num_layers']
            )
            model_type = 'transformer'
            
        else:  # fasttext
            from src.models.baseline.fasttext import create_fasttext_model
            
            # Infer hyperparameters
            params = infer_model_hyperparams(state_dict, 'fasttext')
            print(f"  Inferred FastText params: {params}")
            
            model, _ = create_fasttext_model(
                vocab_size=params['vocab_size'],
                embedding_dim=params['embedding_dim'],
                num_classes=3
            )
            model_type = 'fasttext'
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, model_type
        
    except Exception as e:
        print(f"Legacy loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def safe_model_forward(model, model_type, inputs, sentiment_scores=None, **kwargs):
    """
    Safely call model forward with appropriate arguments
    
    Args:
        model: The model to call
        model_type: Type of model ('fasttext', 'bilstm', 'transformer', etc.)
        inputs: Input tensor
        sentiment_scores: Sentiment scores (optional)
        **kwargs: Additional arguments
    
    Returns:
        Model outputs (logits)
    """
    # Models that support sentiment_scores
    if model_type in ['bilstm'] and hasattr(model, 'use_sentiment_features') and model.use_sentiment_features:
        return model(inputs, sentiment_scores=sentiment_scores, **kwargs)
    
    # Models that don't use sentiment_scores
    else:
        return model(inputs, **kwargs)


def train_ensemble(args):
    print("="*80)
    print("TRAINING ENSEMBLE MODELS")
    print("="*80)
    print(f"Ensemble type: {args.ensemble_type}")
    print(f"Number of models: {len(args.model_paths)}")
    print(f"Device: {args.device}")
    print("="*80)
    
    set_seed(args.seed)
    device = get_device(args.device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/ensemble/{args.ensemble_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading data...")
    data_dict = prepare_data(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    print("\nLoading base models...")
    base_models, model_types = load_trained_models(args.model_paths, device)
    
    print(f"\nLoaded {len(base_models)} models successfully")
    print(f"Model types: {model_types}")
    
    # Create wrapper class for models to handle sentiment_scores
    import torch.nn as nn
    
    class ModelWrapper(nn.Module):
        """Wrapper to handle sentiment_scores conditionally"""
        def __init__(self, model, model_type):
            super().__init__()
            self.model = model
            self.model_type = model_type
        
        def forward(self, inputs, **kwargs):
            return safe_model_forward(self.model, self.model_type, inputs, **kwargs)
        
        def eval(self):
            self.model.eval()
            return super().eval()
        
        def train(self, mode=True):
            self.model.train(mode)
            return super().train(mode)
    
    # Wrap models
    wrapped_models = [ModelWrapper(model, mtype) for model, mtype in zip(base_models, model_types)]
    
    if args.ensemble_type == 'voting':
        print("\nCreating voting ensemble...")
        ensemble = create_voting_ensemble(
            models=wrapped_models,
            voting=args.voting_strategy,
            device=device
        )
        
        print("Ensemble ready for inference")
        
    elif args.ensemble_type == 'weighted':
        print("\nPreparing data loaders for weight optimization...")
        
        preprocessor = EnhancedTextPreprocessor()
        vocab_path = args.vocab_path if args.vocab_path else 'results/baseline/vocabulary.pkl'
        
        if os.path.exists(vocab_path):
            preprocessor.load_vocabulary(vocab_path)
        else:
            print(f"Warning: Vocabulary not found at {vocab_path}, building new...")
            preprocessor.build_vocabulary(data_dict['train']['texts'])
        
        loaders = create_data_loaders(data_dict, preprocessor, batch_size=args.batch_size)
        
        print("\nCreating weighted ensemble...")
        ensemble = create_weighted_ensemble(
            models=wrapped_models,
            val_loader=loaders['val'],
            voting=args.voting_strategy,
            device=device
        )
        
    elif args.ensemble_type == 'stacking':
        print("\nPreparing data loaders for stacking...")
        
        preprocessor = EnhancedTextPreprocessor()
        vocab_path = args.vocab_path if args.vocab_path else 'results/baseline/vocabulary.pkl'
        
        if os.path.exists(vocab_path):
            preprocessor.load_vocabulary(vocab_path)
        else:
            preprocessor.build_vocabulary(data_dict['train']['texts'])
        
        loaders = create_data_loaders(data_dict, preprocessor, batch_size=args.batch_size)
        
        print("\nCreating and training stacking ensemble...")
        ensemble, history = create_stacking_ensemble(
            base_models=wrapped_models,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            num_classes=3,
            meta_model_type=args.meta_model_type,
            epochs=args.meta_epochs,
            device=device
        )
        
        with open(os.path.join(output_dir, 'meta_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        viz = Visualizer(output_dir)
        viz.plot_training_curves(history, 'Meta-Learner')
    
    else:
        raise ValueError(f"Unknown ensemble type: {args.ensemble_type}")
    
    print("\nEvaluating ensemble on test set...")
    
    if args.ensemble_type == 'stacking':
        test_loader = loaders['test']
    else:
        preprocessor = EnhancedTextPreprocessor()
        vocab_path = args.vocab_path if args.vocab_path else 'results/baseline/vocabulary.pkl'
        if os.path.exists(vocab_path):
            preprocessor.load_vocabulary(vocab_path)
        loaders = create_data_loaders(data_dict, preprocessor, batch_size=args.batch_size)
        test_loader = loaders['test']
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].squeeze().to(device)
            sentiment_scores = batch['sentiment_score'].to(device)
            
            if args.ensemble_type == 'voting' or args.ensemble_type == 'weighted':
                preds, probs = ensemble.predict(texts, sentiment_scores=sentiment_scores)
            else:
                logits = ensemble(texts, sentiment_scores=sentiment_scores)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
    import numpy as np
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    precision_macro = precision_score(all_labels, all_predictions, average='macro')
    recall_macro = recall_score(all_labels, all_predictions, average='macro')
    
    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    precision_per_class = precision_score(all_labels, all_predictions, average=None)
    recall_per_class = recall_score(all_labels, all_predictions, average=None)
    
    print(f"\nEnsemble Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro): {recall_macro:.4f}")
    print(f"  F1-Score (macro): {f1_macro:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                                target_names=['Negative', 'Neutral', 'Positive']))
    
    results = {
        'predictions': all_predictions,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_negative': f1_per_class[0],
        'f1_neutral': f1_per_class[1],
        'f1_positive': f1_per_class[2],
        'precision_negative': precision_per_class[0],
        'precision_neutral': precision_per_class[1],
        'precision_positive': precision_per_class[2],
        'recall_negative': recall_per_class[0],
        'recall_neutral': recall_per_class[1],
        'recall_positive': recall_per_class[2],
        'model_types': model_types,
        'ensemble_type': args.ensemble_type
    }
    
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    if args.ensemble_type == 'stacking':
        torch.save(ensemble.state_dict(), os.path.join(output_dir, 'ensemble.pt'))
    
    print(f"\nResults saved to: {output_dir}")
    print("Ensemble training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ensemble models')
    
    parser.add_argument('--data_path', type=str, default='data/raw/sentiment_dataset.csv')
    parser.add_argument('--model_paths', nargs='+', required=True,
                       help='Paths to trained model checkpoints')
    parser.add_argument('--vocab_path', type=str, default=None)
    
    parser.add_argument('--ensemble_type', type=str, required=True,
                       choices=['voting', 'weighted', 'stacking'])
    parser.add_argument('--voting_strategy', type=str, default='soft',
                       choices=['soft', 'hard'])
    parser.add_argument('--meta_model_type', type=str, default='mlp',
                       choices=['linear', 'logistic', 'mlp'])
    parser.add_argument('--meta_epochs', type=int, default=20)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    train_ensemble(args)