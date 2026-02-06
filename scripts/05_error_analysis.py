import torch
import argparse
import pickle
import os
from pathlib import Path
import sys

# Fix path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import EnhancedTextPreprocessor, prepare_data
from src.data.dataset import create_data_loaders
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.error_analysis import ErrorAnalyzer
from src.utils.config import set_seed, get_device


def infer_model_hyperparams(state_dict, model_type):
    """Infer model hyperparameters from state dict"""
    params = {}
    
    if model_type == 'fasttext':
        params['vocab_size'] = state_dict['word_embedding.weight'].shape[0]
        params['embedding_dim'] = state_dict['word_embedding.weight'].shape[1]
        
    elif model_type == 'bilstm':
        # LSTM weight_ih_l0 shape is [4*hidden_size, embedding_dim]
        params['hidden_dim'] = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        params['embedding_dim'] = state_dict['embedding.weight'].shape[1]
        params['vocab_size'] = state_dict['embedding.weight'].shape[0]
        
        # Count LSTM layers
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


def load_model_for_analysis(model_path, vocab_path, device):
    """
    Load model for error analysis with hyperparameter inference
    """
    from src.utils.model_loader import ModelLoader
    
    try:
        # Try new metadata-based loading first
        model = ModelLoader.load_model(model_path, device=device)
        info = ModelLoader.get_model_info(model_path)
        
        model_type = info['model_type']
        use_transformer = model_type in ['roberta', 'bertweet']
        
        print(f"✅ Loaded {model_type} model using metadata")
        
        return model, use_transformer
        
    except ValueError as e:
        # Fall back to legacy method with hyperparameter inference
        print(f"⚠️  Warning: {e}")
        print("Using legacy loading with hyperparameter inference...")
        
        checkpoint = torch.load(model_path, map_location=device)
        model_path_lower = str(model_path).lower()
        
        # Get state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Path-based detection (legacy)
        if 'bertweet' in model_path_lower:
            from src.models.pretrained.bertweet import create_bertweet_model
            model = create_bertweet_model(num_classes=3)
            use_transformer = True
            model_type = 'bertweet'
            
        elif 'roberta' in model_path_lower:
            from src.models.pretrained.roberta import create_roberta_model
            model = create_roberta_model(num_classes=3)
            use_transformer = True
            model_type = 'roberta'
            
        elif 'bilstm' in model_path_lower:
            from src.models.baseline.bilstm_attention import create_bilstm_model
            
            # Infer hyperparameters from checkpoint
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
            use_transformer = False
            model_type = 'bilstm'
            
        elif 'transformer' in model_path_lower:
            from src.models.baseline.custom_transformer import create_custom_transformer
            
            # Infer hyperparameters from checkpoint
            params = infer_model_hyperparams(state_dict, 'transformer')
            print(f"  Inferred Transformer params: {params}")
            
            model, _ = create_custom_transformer(
                vocab_size=params['vocab_size'],
                d_model=params['d_model'],
                num_classes=3,
                max_len=params['max_len'],
                num_layers=params['num_layers']
            )
            use_transformer = False
            model_type = 'transformer'
            
        else:  # fasttext
            from src.models.baseline.fasttext import create_fasttext_model
            
            # Infer hyperparameters from checkpoint
            params = infer_model_hyperparams(state_dict, 'fasttext')
            print(f"  Inferred FastText params: {params}")
            
            model, _ = create_fasttext_model(
                vocab_size=params['vocab_size'],
                embedding_dim=params['embedding_dim'],
                num_classes=3
            )
            use_transformer = False
            model_type = 'fasttext'
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded using legacy method with inferred params")
        
        return model, use_transformer


def analyze_model_errors(args):
    print("="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print("="*80)
    
    set_seed(args.seed)
    device = get_device(args.device)
    
    print("\nLoading data...")
    data_dict = prepare_data(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    print("\nLoading preprocessor...")
    preprocessor = EnhancedTextPreprocessor()
    
    if args.vocab_path:
        preprocessor.load_vocabulary(args.vocab_path)
    else:
        print("Building vocabulary from training data...")
        preprocessor.build_vocabulary(data_dict['train']['texts'])
    
    print("\nCreating data loaders...")
    loaders = create_data_loaders(
        data_dict,
        preprocessor,
        batch_size=args.batch_size,
        add_special_tokens=False
    )
    
    print("\nLoading model...")
    model, use_transformer = load_model_for_analysis(
        args.model_path,
        args.vocab_path,
        device
    )
    
    print("\nEvaluating model and analyzing errors...")
    
    # Use ModelEvaluator to get comprehensive results
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate_with_error_analysis(loaders['test'], verbose=True)
    
    # Extract error analysis
    error_analysis = results.get('error_analysis')
    
    if error_analysis:
        print("\n" + "="*80)
        print("DETAILED ERROR ANALYSIS")
        print("="*80)
        
        summary = error_analysis['summary']
        print(f"\nTotal errors: {summary['total_errors']} / {summary['total_samples']}")
        print(f"Error rate: {summary['error_rate']:.2%}")
        
        print("\nErrors by true class:")
        for class_name, stats in error_analysis['errors_by_class'].items():
            print(f"  {class_name}: {stats['errors']}/{stats['total']} ({stats['error_rate']:.2%})")
        
        print("\nTop confusion pairs:")
        sorted_pairs = sorted(error_analysis['confusion_pairs'].items(), 
                            key=lambda x: x[1], reverse=True)
        for pair, count in sorted_pairs[:5]:
            print(f"  {pair}: {count}")
        
        print(f"\nShowing {min(args.n_examples, len(error_analysis['error_examples']))} high-confidence error examples:")
        for i, example in enumerate(error_analysis['error_examples'][:args.n_examples]):
            print(f"\n{i+1}. True: {example['true_label']}, Predicted: {example['predicted_label']}")
            print(f"   Confidence: {example['confidence']:.3f}")
            print(f"   Text: {example['text'][:150]}...")
    
    # Now use ErrorAnalyzer for additional pattern analysis
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    analyzer = ErrorAnalyzer()
    
    # Get predictions and texts
    texts = data_dict['test']['texts']
    true_labels = results['labels']
    pred_labels = results['predictions']
    probabilities = results['probabilities']
    
    # Detailed analysis
    detailed_analysis = analyzer.analyze_errors(
        texts, true_labels, pred_labels, probabilities, verbose=True
    )
    
    # Save results
    output_dir = args.output_dir if args.output_dir else Path(args.model_path).parent
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed analysis
    analysis_path = os.path.join(output_dir, 'detailed_error_analysis.pkl')
    with open(analysis_path, 'wb') as f:
        pickle.dump(detailed_analysis, f)
    print(f"\n✅ Detailed analysis saved to: {analysis_path}")
    
    # Save evaluator results
    evaluator.save_results(results, output_dir, model_name='model')
    
    print("\n✅ Error analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model errors')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file (for baseline models)')
    parser.add_argument('--data_path', type=str, default='data/raw/sentiment_dataset.csv')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for error analysis')
    parser.add_argument('--n_examples', type=int, default=5,
                       help='Number of examples to show per error type')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.1)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    analyze_model_errors(args)