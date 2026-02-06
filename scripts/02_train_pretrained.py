import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from transformers import AutoTokenizer
from src.data.preprocessing import prepare_data
from src.data.augmentation import augment_training_data
from src.data.dataset import TransformerDataset, create_transformer_data_loaders, get_class_weights
from src.models.pretrained.roberta import create_roberta_model, get_roberta_tokenizer
from src.models.pretrained.bertweet import create_bertweet_model, get_bertweet_tokenizer
from src.training.trainer import Trainer as TransformerTrainer
from src.training.losses import get_loss_function
from src.utils.config import set_seed, get_device
from src.utils.logger import setup_logger
from src.utils.helpers import count_parameters


def train_pretrained_models(data_path='data/raw/sentiment_dataset.csv',
                            output_dir='results/pretrained',
                            use_augmentation=True,
                            use_class_weights=True,
                            device='auto',
                            seed=42):
    
    set_seed(seed)
    device = get_device(device)
    logger = setup_logger(name='pretrained_training', log_dir='logs')
    
    logger.info("="*80)
    logger.info("PRETRAINED TRANSFORMERS TRAINING")
    logger.info("="*80)
    
    data_dict = prepare_data(data_path, test_size=0.1, val_size=0.1, random_state=seed)
    
    if use_augmentation:
        logger.info("\nApplying data augmentation...")
        data_dict = augment_training_data(
            data_dict,
            n_aug=1,
            balance_classes=True,
            aug_methods=['synonym', 'swap']
        )
    
    class_weights = None
    if use_class_weights:
        class_weights = get_class_weights(data_dict['train']['labels'])
        logger.info(f"\nClass weights: {class_weights}")
    
    models_config = {
        'roberta': {
            'model_fn': lambda: create_roberta_model(
                model_name='roberta-base',
                num_classes=3,
                dropout=0.5,
                freeze_bert=False,
                freeze_layers=0
            ),
            'tokenizer_fn': lambda: get_roberta_tokenizer('roberta-base'),
            'lr': 1e-5,
            'epochs': 5,
            'batch_size': 32,
            'max_length': 128
        },
        'bertweet': {
            'model_fn': lambda: create_bertweet_model(
                model_name='vinai/bertweet-base',
                num_classes=3,
                dropout=0.5,
                freeze_bert=False,
                freeze_layers=0
            ),
            'tokenizer_fn': lambda: get_bertweet_tokenizer('vinai/bertweet-base'),
            'lr': 2e-5,
            'epochs': 4,
            'batch_size': 32,
            'max_length': 128
        }
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING {model_name.upper()} MODEL")
        logger.info("="*80)
        
        try:
            tokenizer = config['tokenizer_fn']()
            logger.info(f"\nTokenizer loaded: {model_name}")
            
            loaders = create_transformer_data_loaders(
                data_dict,
                tokenizer,
                max_length=config['max_length'],
                batch_size=config['batch_size']
            )
            
            model = config['model_fn']()
            params = count_parameters(model)
            logger.info(f"\nModel parameters: {params['trainable']:,}")
            
            loss_fn = get_loss_function(
                loss_type='focal' if use_class_weights else 'cross_entropy',
                class_weights=class_weights,
                num_classes=3,
                gamma=2.0
            )
            
            trainer = TransformerTrainer(
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
                warmup_ratio=0.1,
                early_stopping_patience=2,
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
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    logger.info("\n" + "="*80)
    logger.info("PRETRAINED MODELS TRAINING COMPLETE")
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
    
    parser = argparse.ArgumentParser(description='Train pretrained transformer models')
    parser.add_argument('--data_path', type=str, default='data/raw/sentiment_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='results/pretrained')
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--class_weights', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    results = train_pretrained_models(
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_augmentation=args.augmentation,
        use_class_weights=args.class_weights,
        device=args.device,
        seed=args.seed
    )