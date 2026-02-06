import torch
import os
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import EnhancedTextPreprocessor, prepare_data
from src.data.augmentation import augment_training_data
from src.data.dataset import create_data_loaders, get_class_weights
from src.models.baseline.bilstm_attention import create_bilstm_model
from src.models.baseline.fasttext import create_fasttext_model
from src.models.baseline.custom_transformer import create_custom_transformer
from src.training.trainer import Trainer
from src.training.losses import get_loss_function, compute_class_weights
from src.utils.config import set_seed, get_device
from src.utils.logger import setup_logger
from src.utils.helpers import count_parameters


def train_baseline_models(data_path='data/raw/sentiment_dataset.csv',
                          output_dir='results/baseline',
                          use_augmentation=True,
                          use_class_weights=True,
                          device='auto',
                          seed=42):
    
    set_seed(seed)
    device = get_device(device)
    logger = setup_logger(name='baseline_training', log_dir='logs')
    
    logger.info("="*80)
    logger.info("BASELINE MODELS TRAINING")
    logger.info("="*80)
    
    data_dict = prepare_data(data_path, test_size=0.1, val_size=0.1, random_state=seed)
    
    if use_augmentation:
        logger.info("\nApplying data augmentation...")
        data_dict = augment_training_data(
            data_dict,
            n_aug=1,
            balance_classes=True,
            aug_methods=['synonym', 'swap', 'delete']
        )
    
    preprocessor = EnhancedTextPreprocessor(
        vocab_size=10000,
        max_length=100,
        min_freq=2,
        use_spell_check=False,
        use_lemmatization=False
    )
    
    logger.info("\nBuilding vocabulary...")
    preprocessor.build_vocabulary(data_dict['train']['texts'])
    
    vocab_path = Path(output_dir) / 'vocabulary.pkl'
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save_vocabulary(vocab_path)
    
    logger.info("\nCreating data loaders...")
    loaders = create_data_loaders(
        data_dict,
        preprocessor,
        batch_size=64,
        add_special_tokens=False
    )
    
    class_weights = None
    if use_class_weights:
        class_weights = get_class_weights(data_dict['train']['labels'])
        logger.info(f"\nClass weights: {class_weights}")
    
    models_to_train = {
      'fasttext': {
         'model_fn': lambda: create_fasttext_model(
            vocab_size=preprocessor.get_vocab_size(),
            embedding_dim=100,
            num_classes=3,
            dropout=0.3,
            padding_idx=preprocessor.get_padding_idx(),
            use_char_ngrams=True
        ),
        
        'lr': 0.001,
        'epochs': 20
    },
    'bilstm': {
        'model_fn': lambda: create_bilstm_model(
            vocab_size=preprocessor.get_vocab_size(),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=3,
            dropout=0.5,
            num_attention_heads=4,
            padding_idx=preprocessor.get_padding_idx(),
            use_sentiment_features=True
        ),
        
        'lr': 0.003,
        'epochs': 20
    },
    'transformer': {
        'model_fn': lambda: create_custom_transformer(
            vocab_size=preprocessor.get_vocab_size(),
            d_model=256,
            num_heads=4,
            num_layers=4,
            d_ff=1024,
            num_classes=3,
            max_len=100,
            dropout=0.1,
            padding_idx=preprocessor.get_padding_idx()
        ),
        
        'lr': 0.0001,
        'epochs': 25
    }
}
    
    results = {}
    
    for model_name, config in models_to_train.items():
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING {model_name.upper()} MODEL")
        logger.info("="*80)
        
        model, model_config = config['model_fn']()
        model_config['model_type'] = model_name
        params = count_parameters(model)
        logger.info(f"\nModel parameters: {params['trainable']:,}")
        
        loss_fn = get_loss_function(
            loss_type='focal' if use_class_weights else 'cross_entropy',
            class_weights=class_weights,
            num_classes=3,
            gamma=2.0
        )
        
        trainer = Trainer(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            loss_fn=loss_fn,
            device=device,
            logger=logger
        )
        
        model_dir = Path(output_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        trained_model, history = trainer.train(
            num_epochs=config['epochs'],
            learning_rate=config['lr'],
            early_stopping_patience=5,
            save_path=model_dir / 'best_model.pt'
        )
        
        test_metrics = trainer.evaluate(loaders['test'])
        
        results[model_name] = {
            'history': history,
            'test_metrics': test_metrics,
            'model_path': str(model_dir / 'best_model.pt')
        }
        
        logger.info(f"\n{model_name.upper()} Test Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("="*80)
    
    logger.info("\nModel Comparison:")
    for model_name, result in results.items():
        metrics = result['test_metrics']
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--data_path', type=str, default='data/raw/sentiment_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='results/baseline')
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--class_weights', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    results = train_baseline_models(
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_augmentation=args.augmentation,
        use_class_weights=args.class_weights,
        device=args.device,
        seed=args.seed
    )