"""
QUICK FIX: Generate missing results.pkl files
Run this from your Sentiment_analysis directory:
    python generate_results_pkl.py

This script will:
1. Find all best_model.pt files
2. Evaluate them on the test set
3. Create results.pkl files for 04_evaluate_all.py
"""

import torch
import pickle
import os
from pathlib import Path
import sys

# Add src to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

def check_environment():
    """Check if we're in the right directory"""
    if not Path('src').exists():
        print("❌ Error: 'src' directory not found!")
        print("Please run this script from the Sentiment_analysis directory")
        sys.exit(1)
    
    if not Path('data/raw/sentiment_dataset.csv').exists():
        print("❌ Error: Dataset not found!")
        print("Expected: data/raw/sentiment_dataset.csv")
        sys.exit(1)
    
    print("✅ Environment check passed")


def generate_results_for_model(model_path, model_type, device='cuda'):
    """
    Generate results.pkl for a single model
    
    Args:
        model_path: Path to best_model.pt
        model_type: 'baseline' or 'pretrained'
        device: Device to use
    """
    from src.data.preprocessing import EnhancedTextPreprocessor, prepare_data
    from src.data.dataset import create_data_loaders, create_transformer_data_loaders
    from src.evaluation.evaluator import ModelEvaluator
    from src.utils.config import set_seed, get_device
    
    model_path = Path(model_path)
    model_name = model_path.parent.name
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    # Set seed
    set_seed(42)
    device = get_device(device)
    
    # Load data
    print("Loading data...")
    data_dict = prepare_data(
        'data/raw/sentiment_dataset.csv',
        test_size=0.1,
        val_size=0.1,
        random_state=42
    )
    
    # Load model and create data loader
    if model_type == 'baseline':
        # Load vocabulary
        vocab_path = Path('results/baseline/vocabulary.pkl')
        if not vocab_path.exists():
            print(f"❌ Vocabulary not found: {vocab_path}")
            return None
        
        print("Loading vocabulary...")
        preprocessor = EnhancedTextPreprocessor()
        preprocessor.load_vocabulary(vocab_path)
        
        print("Creating data loaders...")
        loaders = create_data_loaders(
            data_dict,
            preprocessor,
            batch_size=64,
            add_special_tokens=False
        )
        
        # Load model (try new method first, fallback to legacy)
        print("Loading model...")
        try:
            from src.utils.model_loader import ModelLoader
            model = ModelLoader.load_model(model_path, device=device)
            print("✅ Loaded using ModelLoader")
        except:
            print("⚠️  Using legacy loading...")
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            vocab_size = preprocessor.get_vocab_size()
            padding_idx = preprocessor.get_padding_idx()
            
            if 'fasttext' in model_name:
                from src.models.baseline.fasttext import create_fasttext_model
                model, _ = create_fasttext_model(
                    vocab_size=vocab_size,
                    embedding_dim=100,
                    num_classes=3,
                    dropout=0.3,
                    padding_idx=padding_idx,
                    use_char_ngrams=True
                )
            elif 'bilstm' in model_name:
                from src.models.baseline.bilstm_attention import create_bilstm_model
                model, _ = create_bilstm_model(
                    vocab_size=vocab_size,
                    embedding_dim=128,
                    hidden_dim=256,
                    num_layers=2,
                    num_classes=3,
                    dropout=0.5,
                    num_attention_heads=4,
                    padding_idx=padding_idx,
                    use_sentiment_features=True
                )
            elif 'transformer' in model_name:
                from src.models.baseline.custom_transformer import create_custom_transformer
                model, _ = create_custom_transformer(
                    vocab_size=vocab_size,
                    d_model=256,
                    num_heads=4,
                    num_layers=4,
                    d_ff=1024,
                    num_classes=3,
                    max_len=100,
                    dropout=0.1,
                    padding_idx=padding_idx
                )
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
    
    elif model_type == 'pretrained':
        # Get tokenizer and create loaders
        if 'roberta' in model_name:
            from src.models.pretrained.roberta import get_roberta_tokenizer
            tokenizer = get_roberta_tokenizer('roberta-base')
        elif 'bertweet' in model_name:
            from src.models.pretrained.bertweet import get_bertweet_tokenizer
            tokenizer = get_bertweet_tokenizer('vinai/bertweet-base')
        
        print("Creating data loaders...")
        loaders = create_transformer_data_loaders(
            data_dict,
            tokenizer,
            max_length=128,
            batch_size=32
        )
        
        # Load model
        print("Loading model...")
        try:
            from src.utils.model_loader import ModelLoader
            model = ModelLoader.load_model(model_path, device=device)
            print("✅ Loaded using ModelLoader")
        except:
            print("⚠️  Using legacy loading...")
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            if 'roberta' in model_name:
                from src.models.pretrained.roberta import create_roberta_model
                model = create_roberta_model(
                    model_name='roberta-base',
                    num_classes=3,
                    dropout=0.5,
                    freeze_bert=False,
                    freeze_layers=0
                )
            elif 'bertweet' in model_name:
                from src.models.pretrained.bertweet import create_bertweet_model
                model = create_bertweet_model(
                    model_name='vinai/bertweet-base',
                    num_classes=3,
                    dropout=0.5,
                    freeze_bert=False,
                    freeze_layers=0
                )
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
    
    # Evaluate
    print("Evaluating model...")
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(loaders['test'], verbose=True)
    
    # Add metadata
    results['model_name'] = model_name
    results['model_type'] = model_type
    
    # Save results
    results_file = model_path.parent / 'results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✅ Saved: {results_file}")
    print(f"   Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"   F1-Score: {results['metrics']['f1_macro']:.4f}")
    
    return results


def main():
    print("="*80)
    print("GENERATING MISSING RESULTS.PKL FILES")
    print("="*80)
    print()
    
    # Check environment
    check_environment()
    
    # Find all best_model.pt files
    baseline_models = list(Path('results/baseline').rglob('best_model.pt'))
    pretrained_models = list(Path('results/pretrained').rglob('best_model.pt'))
    
    print(f"\nFound {len(baseline_models)} baseline models:")
    for m in baseline_models:
        print(f"  - {m}")
    
    print(f"\nFound {len(pretrained_models)} pretrained models:")
    for m in pretrained_models:
        print(f"  - {m}")
    
    # Process baseline models
    if baseline_models:
        print("\n" + "="*80)
        print("PROCESSING BASELINE MODELS")
        print("="*80)
        
        for model_path in baseline_models:
            try:
                generate_results_for_model(model_path, 'baseline')
            except Exception as e:
                print(f"❌ Error processing {model_path.parent.name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Process pretrained models
    if pretrained_models:
        print("\n" + "="*80)
        print("PROCESSING PRETRAINED MODELS")
        print("="*80)
        
        for model_path in pretrained_models:
            try:
                generate_results_for_model(model_path, 'pretrained')
            except Exception as e:
                print(f"❌ Error processing {model_path.parent.name}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print("\nGenerated files in:")
    for model_path in baseline_models + pretrained_models:
        results_file = model_path.parent / 'results.pkl'
        if results_file.exists():
            print(f"  ✅ {results_file}")
    
    print("\nYou can now run:")
    print("  python scripts/04_evaluate_all.py")


if __name__ == "__main__":
    main()