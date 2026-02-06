"""
Inference Script for Sentiment Analysis
Perform predictions on new text using trained models
"""

import torch
import argparse
import pickle
from pathlib import Path
import sys

# Fix path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.model_loader import ModelLoader
from src.data.preprocessing import EnhancedTextPreprocessor
from src.utils.config import set_seed, get_device


def load_model_and_preprocessor(model_path, vocab_path=None, device='cpu'):
    """
    Load trained model and preprocessor using ModelLoader
    
    Args:
        model_path: Path to model checkpoint
        vocab_path: Path to vocabulary file (for baseline models)
        device: Device to run on
    
    Returns:
        tuple: (model, preprocessor, is_transformer)
    """
    from src.utils.model_loader import ModelLoader
    
    print(f"Loading model from {model_path}...")
    
    try:
        # Use ModelLoader for automatic detection
        model = ModelLoader.load_model(model_path, device=device)
        info = ModelLoader.get_model_info(model_path)
        
        model_type = info['model_type']
        is_transformer = model_type in ['roberta', 'bertweet']
        
        print(f"Model type: {model_type}")
        
        # Load appropriate preprocessor/tokenizer
        if is_transformer:
            if model_type == 'bertweet':
                from src.models.pretrained.bertweet import get_bertweet_tokenizer
                preprocessor = get_bertweet_tokenizer()
                print("Loaded BERTweet tokenizer")
            else:  # roberta
                from src.models.pretrained.roberta import get_roberta_tokenizer
                preprocessor = get_roberta_tokenizer()
                print("Loaded RoBERTa tokenizer")
        else:
            # Baseline model - need vocabulary
            if vocab_path is None:
                raise ValueError(
                    f"vocab_path required for baseline model (type: {model_type})\n"
                    f"Please specify --vocab_path argument"
                )
            
            from src.data.preprocessing import EnhancedTextPreprocessor
            preprocessor = EnhancedTextPreprocessor()
            preprocessor.load_vocabulary(vocab_path)
            print(f"Loaded vocabulary from {vocab_path}")
        
        print(f"✅ Successfully loaded {model_type} model!")
        
        return model, preprocessor, is_transformer
        
    except ValueError as e:
        print(f"⚠️  Warning: {e}")
        
        # Check if it's a metadata error or vocab path error
        if "vocab_path required" in str(e):
            raise  # Re-raise vocab path errors
        
        print("Attempting legacy loading method...")
        
        # Legacy path-based detection
        checkpoint = torch.load(model_path, map_location=device)
        model_path_lower = str(model_path).lower()
        
        if 'bertweet' in model_path_lower:
            from src.models.pretrained.bertweet import create_bertweet_model, get_bertweet_tokenizer
            model = create_bertweet_model(num_classes=3)
            preprocessor = get_bertweet_tokenizer()
            is_transformer = True
            
        elif 'roberta' in model_path_lower:
            from src.models.pretrained.roberta import create_roberta_model, get_roberta_tokenizer
            model = create_roberta_model(num_classes=3)
            preprocessor = get_roberta_tokenizer()
            is_transformer = True
            
        else:
            # Baseline model
            if vocab_path is None:
                raise ValueError("vocab_path required for baseline models")
            
            from src.data.preprocessing import EnhancedTextPreprocessor
            import pickle
            
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
            vocab_size = len(vocab_data['word2idx'])
            
            if 'bilstm' in model_path_lower:
                from src.models.baseline.bilstm_attention import create_bilstm_model
                model, _ = create_bilstm_model(vocab_size=vocab_size, num_classes=3)
            elif 'transformer' in model_path_lower:
                from src.models.baseline.custom_transformer import create_custom_transformer
                model, _ = create_custom_transformer(vocab_size=vocab_size, num_classes=3)
            else:  # fasttext
                from src.models.baseline.fasttext import create_fasttext_model
                model, _ = create_fasttext_model(vocab_size=vocab_size, num_classes=3)
            
            preprocessor = EnhancedTextPreprocessor()
            preprocessor.load_vocabulary(vocab_path)
            is_transformer = False
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print("✅ Model loaded using legacy method")
        
        return model, preprocessor, is_transformer


def predict_text(text, model, preprocessor, is_transformer, device='cpu', max_length=128):
    """
    Predict sentiment for a single text
    
    Args:
        text: Input text
        model: Loaded model
        preprocessor: Preprocessor or tokenizer
        is_transformer: Whether model is transformer
        device: Device to run on
        max_length: Maximum sequence length
    
    Returns:
        prediction: Predicted class (0, 1, 2)
        probabilities: Class probabilities
        class_name: Predicted class name
    """
    class_names = ['Negative', 'Neutral', 'Positive']
    
    with torch.no_grad():
        if is_transformer:
            # Transformer model
            encoded = preprocessor(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            
        else:
            # Baseline model
            sequence = preprocessor.text_to_sequence(text)
            padded = preprocessor.pad_sequence(sequence)
            
            inputs = torch.LongTensor([padded]).to(device)
            sentiment_scores = torch.FloatTensor([preprocessor.compute_vader_features(text)]).to(device)
            
            logits = model(inputs, sentiment_scores=sentiment_scores)
        
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        probabilities = probs[0].cpu().numpy()
    
    return prediction, probabilities, class_names[prediction]


def batch_predict(texts, model, preprocessor, is_transformer, device='cpu', max_length=128, batch_size=32):
    """
    Predict sentiments for multiple texts
    
    Args:
        texts: List of texts
        model: Loaded model
        preprocessor: Preprocessor or tokenizer
        is_transformer: Whether model is transformer
        device: Device to run on
        max_length: Maximum sequence length
        batch_size: Batch size for processing
    
    Returns:
        predictions: List of predicted classes
        probabilities: Array of class probabilities
        class_names_list: List of predicted class names
    """
    class_names = ['Negative', 'Neutral', 'Positive']
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            if is_transformer:
                # Transformer model
                encoded = preprocessor(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask=attention_mask)
                
            else:
                # Baseline model
                batch_sequences = []
                batch_vader = []
                
                for text in batch_texts:
                    sequence = preprocessor.text_to_sequence(text)
                    padded = preprocessor.pad_sequence(sequence)
                    batch_sequences.append(padded)
                    batch_vader.append(preprocessor.compute_vader_features(text))
                
                inputs = torch.LongTensor(batch_sequences).to(device)
                sentiment_scores = torch.FloatTensor(batch_vader).to(device)
                
                logits = model(inputs, sentiment_scores=sentiment_scores)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    import numpy as np
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    class_names_list = [class_names[pred] for pred in all_predictions]
    
    return all_predictions, all_probabilities, class_names_list


def interactive_mode(model, preprocessor, is_transformer, device):
    """
    Interactive inference mode
    
    Args:
        model: Loaded model
        preprocessor: Preprocessor or tokenizer
        is_transformer: Whether model is transformer
        device: Device to run on
    """
    print("\n" + "="*80)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("="*80)
    print("Enter text to analyze (or 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not text:
            print("⚠️  Please enter some text")
            continue
        
        # Predict
        prediction, probabilities, class_name = predict_text(
            text, model, preprocessor, is_transformer, device
        )
        
        # Display results
        print(f"\n{'='*80}")
        print(f"Text: {text}")
        print(f"{'='*80}")
        print(f"Prediction: {class_name}")
        print(f"Confidence: {probabilities[prediction]:.2%}")
        print(f"\nProbabilities:")
        print(f"  Negative: {probabilities[0]:.2%}")
        print(f"  Neutral:  {probabilities[1]:.2%}")
        print(f"  Positive: {probabilities[2]:.2%}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Inference')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file (for baseline models)')
    
    # Inference arguments
    parser.add_argument('--text', type=str, default=None,
                       help='Single text to analyze')
    parser.add_argument('--file', type=str, default=None,
                       help='File with texts to analyze (one per line)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for batch predictions (CSV)')
    
    # Model arguments
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for batch prediction')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run on (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    print("="*80)
    print("SENTIMENT ANALYSIS INFERENCE")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print("="*80)
    
    # Load model
    model, preprocessor, is_transformer = load_model_and_preprocessor(
        args.model_path,
        args.vocab_path,
        device
    )
    
    # Inference mode
    if args.interactive:
        # Interactive mode
        interactive_mode(model, preprocessor, is_transformer, device)
        
    elif args.text:
        # Single text
        prediction, probabilities, class_name = predict_text(
            args.text, model, preprocessor, is_transformer, device, args.max_length
        )
        
        print(f"\n{'='*80}")
        print(f"Text: {args.text}")
        print(f"{'='*80}")
        print(f"Prediction: {class_name}")
        print(f"Confidence: {probabilities[prediction]:.2%}")
        print(f"\nProbabilities:")
        print(f"  Negative: {probabilities[0]:.2%}")
        print(f"  Neutral:  {probabilities[1]:.2%}")
        print(f"  Positive: {probabilities[2]:.2%}")
        print(f"{'='*80}\n")
        
    elif args.file:
        # Batch from file
        print(f"\nReading texts from {args.file}...")
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(texts)} texts")
        print("Predicting...")
        
        predictions, probabilities, class_names = batch_predict(
            texts, model, preprocessor, is_transformer, device,
            args.max_length, args.batch_size
        )
        
        # Save results if output specified
        if args.output:
            import pandas as pd
            
            df = pd.DataFrame({
                'text': texts,
                'prediction': class_names,
                'negative_prob': probabilities[:, 0],
                'neutral_prob': probabilities[:, 1],
                'positive_prob': probabilities[:, 2]
            })
            
            df.to_csv(args.output, index=False)
            print(f"\n✅ Results saved to {args.output}")
        
        # Display sample results
        print(f"\n{'='*80}")
        print("SAMPLE RESULTS (first 5)")
        print(f"{'='*80}")
        for i in range(min(5, len(texts))):
            print(f"\n{i+1}. {texts[i]}")
            print(f"   → {class_names[i]} ({probabilities[i][predictions[i]]:.2%})")
        print(f"{'='*80}\n")
        
    else:
        print("\n⚠️  Please specify --text, --file, or --interactive")
        print("Examples:")
        print("  python scripts/06_inference.py --model_path models/roberta/best_model.pt --text 'This is amazing!'")
        print("  python scripts/06_inference.py --model_path models/bilstm/best_model.pt --vocab_path results/baseline/vocabulary.pkl --file texts.txt --output results.csv")
        print("  python scripts/06_inference.py --model_path models/roberta/best_model.pt --interactive")


if __name__ == "__main__":
    main()