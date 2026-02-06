"""
ModelLoader - Intelligent model loading with automatic type detection
Replaces fragile path-based model detection with metadata-based approach
"""

import torch
from pathlib import Path
import json
import warnings


class ModelLoader:
    """
    Utility class for loading models with automatic type detection
    
    Features:
    - Loads models using metadata from checkpoint
    - No more fragile path-based detection
    - Complete model reproducibility
    - Quick metadata inspection without loading weights
    
    Example:
        >>> # Quick info without loading weights
        >>> info = ModelLoader.get_model_info('model.pt')
        >>> print(f"Model type: {info['model_type']}")
        
        >>> # Load model automatically
        >>> model = ModelLoader.load_model('model.pt', device='cuda')
        
        >>> # Load with optimizer state
        >>> data = ModelLoader.load_checkpoint_full('model.pt')
        >>> model = data['model']
        >>> optimizer.load_state_dict(data['optimizer_state'])
    """
    
    @staticmethod
    def get_model_info(checkpoint_path):
        """
        Get model information without loading weights (fast)
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            dict: Model metadata (type, config, metrics, timestamp)
        
        Example:
            >>> info = ModelLoader.get_model_info('results/bilstm/best_model.pt')
            >>> print(f"Model: {info['model_type']}")
            >>> print(f"Accuracy: {info['metrics']['accuracy']:.2%}")
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Try to load metadata file first (much faster)
        metadata_path = str(checkpoint_path).replace('.pt', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        # Fall back to loading checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        return {
            'model_type': checkpoint.get('model_type'),
            'model_config': checkpoint.get('model_config'),
            'metrics': checkpoint.get('metrics'),
            'timestamp': checkpoint.get('timestamp'),
            'pytorch_version': checkpoint.get('pytorch_version')
        }
    
    @staticmethod
    def _create_model(model_type, model_config):
        """
        Create model instance based on type and config
        
        Args:
            model_type: Type of model ('bilstm', 'transformer', 'fasttext', 'roberta', 'bertweet')
            model_config: Configuration dictionary
        
        Returns:
            Model instance
        
        Raises:
            ValueError: If model_type is unknown
        """
        if model_type == 'bilstm':
            from src.models.baseline.bilstm_attention import create_bilstm_model
            model, _ = create_bilstm_model(**model_config)
            return model
            
        elif model_type == 'transformer':
            from src.models.baseline.custom_transformer import create_custom_transformer
            model, _ = create_custom_transformer(**model_config)
            return model
            
        elif model_type == 'fasttext':
            from src.models.baseline.fasttext import create_fasttext_model
            model, _ = create_fasttext_model(**model_config)
            return model
            
        elif model_type == 'roberta':
            from src.models.pretrained.roberta import create_roberta_model
            # RoBERTa models don't return config tuple in old version
            model_name = model_config.get('model_name', 'roberta-base')
            num_classes = model_config.get('num_classes', 3)
            dropout = model_config.get('dropout', 0.5)
            freeze_bert = model_config.get('freeze_bert', False)
            freeze_layers = model_config.get('freeze_layers', 0)
            
            model = create_roberta_model(
                model_name=model_name,
                num_classes=num_classes,
                dropout=dropout,
                freeze_bert=freeze_bert,
                freeze_layers=freeze_layers
            )
            return model
            
        elif model_type == 'bertweet':
            from src.models.pretrained.bertweet import create_bertweet_model
            model_name = model_config.get('model_name', 'vinai/bertweet-base')
            num_classes = model_config.get('num_classes', 3)
            dropout = model_config.get('dropout', 0.5)
            freeze_bert = model_config.get('freeze_bert', False)
            freeze_layers = model_config.get('freeze_layers', 0)
            
            model = create_bertweet_model(
                model_name=model_name,
                num_classes=num_classes,
                dropout=dropout,
                freeze_bert=freeze_bert,
                freeze_layers=freeze_layers
            )
            return model
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def load_model(checkpoint_path, device='cpu', strict=True):
        """
        Load model with automatic type detection from metadata
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on ('cpu', 'cuda', etc.)
            strict: Whether to strictly enforce state_dict matching
        
        Returns:
            Loaded model (in eval mode, on specified device)
        
        Raises:
            ValueError: If checkpoint missing metadata or has unknown model type
        
        Example:
            >>> model = ModelLoader.load_model('results/bilstm/best_model.pt', device='cuda')
            >>> predictions = model(inputs)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model type and config from metadata
        model_type = checkpoint.get('model_type')
        model_config = checkpoint.get('model_config')
        
        if model_type is None:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} does not contain 'model_type' metadata.\n"
                f"This checkpoint was saved with the old save_model() function.\n"
                f"Please either:\n"
                f"  1. Re-train the model with the updated save_model() function, or\n"
                f"  2. Use the legacy loading method (not recommended)"
            )
        
        if model_config is None:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} does not contain 'model_config' metadata.\n"
                f"Cannot recreate model without configuration."
            )
        
        # Create model from config
        print(f"Loading {model_type} model from checkpoint...")
        model = ModelLoader._create_model(model_type, model_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        
        return model
    
    @staticmethod
    def load_checkpoint_full(checkpoint_path, device='cpu'):
        """
        Load full checkpoint including optimizer state and metadata
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load on
        
        Returns:
            dict with keys:
                - 'model': Loaded model
                - 'optimizer_state': Optimizer state dict (or None)
                - 'epoch': Last epoch number
                - 'metrics': Saved metrics
                - 'model_type': Model type
                - 'model_config': Model configuration
        
        Example:
            >>> data = ModelLoader.load_checkpoint_full('model.pt')
            >>> model = data['model']
            >>> 
            >>> # Resume training
            >>> optimizer = torch.optim.Adam(model.parameters())
            >>> if data['optimizer_state']:
            ...     optimizer.load_state_dict(data['optimizer_state'])
            >>> 
            >>> start_epoch = data['epoch'] + 1
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model_type = checkpoint.get('model_type')
        model_config = checkpoint.get('model_config')
        
        if model_type is None or model_config is None:
            raise ValueError("Checkpoint missing required metadata (model_type or model_config)")
        
        # Create and load model
        model = ModelLoader._create_model(model_type, model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return {
            'model': model,
            'optimizer_state': checkpoint.get('optimizer_state_dict'),
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'model_type': model_type,
            'model_config': model_config
        }
    
    @staticmethod
    def load_model_legacy(checkpoint_path, model_type, model_config, device='cpu'):
        """
        Load model the old way (for checkpoints without metadata)
        
        Args:
            checkpoint_path: Path to checkpoint
            model_type: Model type string (must specify manually)
            model_config: Model configuration dict (must specify manually)
            device: Device to load on
        
        Returns:
            Loaded model
        
        Example:
            >>> # For old checkpoints without metadata
            >>> config = {'vocab_size': 10000, 'num_classes': 3}
            >>> model = ModelLoader.load_model_legacy(
            ...     'old_model.pt',
            ...     model_type='bilstm',
            ...     model_config=config
            ... )
        """
        warnings.warn(
            "Using legacy loading method. Consider re-training model with new save format.",
            DeprecationWarning
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model
        model = ModelLoader._create_model(model_type, model_config)
        
        # Load weights (handle both old and new checkpoint formats)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model


# Convenience function for backward compatibility
def load_model(checkpoint_path, device='cpu'):
    """
    Convenience function to load model (uses ModelLoader internally)
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
    
    Returns:
        Loaded model
    
    Example:
        >>> from src.utils.model_loader import load_model
        >>> model = load_model('results/bilstm/best_model.pt', device='cuda')
    """
    return ModelLoader.load_model(checkpoint_path, device)


if __name__ == "__main__":
    print("="*80)
    print("MODELLOADER UTILITY")
    print("="*80)
    print("\nThis module provides intelligent model loading with automatic type detection.")
    print("\nUsage:")
    print("  1. Quick info: ModelLoader.get_model_info('model.pt')")
    print("  2. Load model: ModelLoader.load_model('model.pt', device='cuda')")
    print("  3. Full load:  ModelLoader.load_checkpoint_full('model.pt')")
    print("\nFeatures:")
    print("  ✅ Automatic model type detection from metadata")
    print("  ✅ No more fragile path-based detection")
    print("  ✅ Complete reproducibility")
    print("  ✅ Quick metadata inspection")
    print("  ✅ Legacy checkpoint support")
    print("\n" + "="*80)