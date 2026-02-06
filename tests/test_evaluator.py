"""
Unit tests for model evaluator
"""
import pytest
import torch
import tempfile
from pathlib import Path
from src.evaluation.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    def test_evaluator_initialization(self, small_bilstm_model, device):
        """Evaluator should initialize with model."""
        evaluator = ModelEvaluator(small_bilstm_model, device=device)
        
        assert evaluator.model is not None
        assert evaluator.device is not None
    
    @pytest.mark.slow
    def test_evaluate_basic(self, small_bilstm_model, preprocessor_with_vocab, sample_dataset_dict, device):
        """Evaluator should evaluate model."""
        from src.data.dataset import create_data_loaders
        
        # Create data loaders
        loaders = create_data_loaders(
            sample_dataset_dict,
            preprocessor_with_vocab,
            batch_size=4
        )
        
        evaluator = ModelEvaluator(small_bilstm_model, device=device)
        results = evaluator.evaluate(loaders['test'], verbose=False)
        
        assert 'metrics' in results
        assert 'labels' in results
        assert 'predictions' in results
        assert 'probabilities' in results
    
    def test_save_results(self, small_bilstm_model, device):
        """Should save evaluation results."""
        evaluator = ModelEvaluator(small_bilstm_model, device=device)
        
        # Create dummy results
        results = {
            'metrics': {'accuracy': 0.85, 'f1_macro': 0.83},
            'labels': [0, 1, 2],
            'predictions': [0, 1, 2],
            'probabilities': [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator.save_results(results, tmpdir, 'test_model')
            
            # Check files were created
            output_dir = Path(tmpdir)
            assert (output_dir / 'metrics.json').exists()