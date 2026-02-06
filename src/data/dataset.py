"""
PyTorch Dataset Classes for Sentiment Analysis
Type-safe implementation with explicit per-key typing
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Union, Protocol, runtime_checkable, cast, Any
import warnings
import sys


@runtime_checkable
class AugmenterProtocol(Protocol):
    """Protocol defining the required interface for text augmenters"""
    def augment_text(self, text: str) -> str: ...
    @property
    def aug_methods(self) -> List[str]: ...


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis"""
    
    def __init__(
        self,
        texts: Union[List[str], np.ndarray],
        labels: Union[List[int], np.ndarray, torch.Tensor],
        preprocessor,
        add_special_tokens: bool = False,
        augmenter: Optional[AugmenterProtocol] = None,
        augment_prob: float = 0.0,
        num_classes: int = 3
    ):
        if len(texts) == 0:
            raise ValueError("texts cannot be empty")
        if len(labels) == 0:
            raise ValueError("labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(f"Length mismatch: texts ({len(texts)}) != labels ({len(labels)})")
        
        self.texts = np.asarray(texts)
        self.labels = np.asarray(labels).astype(np.int64)
        
        if np.min(self.labels) < 0 or np.max(self.labels) >= num_classes:
            invalid_labels = np.unique(self.labels[(self.labels < 0) | (self.labels >= num_classes)])
            raise ValueError(
                f"Labels must be in range [0, {num_classes-1}]. "
                f"Found invalid labels: {invalid_labels.tolist()}"
            )
        
        if not hasattr(preprocessor, 'word2idx') or len(preprocessor.word2idx) == 0:
            raise RuntimeError(
                "Preprocessor vocabulary not built. Call preprocessor.build_vocabulary() first."
            )
        
        self.preprocessor = preprocessor
        self.add_special_tokens = add_special_tokens
        self.augmenter = augmenter
        self.augment_prob = max(0.0, min(1.0, augment_prob))
        self.num_classes = num_classes
        
        self.precomputed = (augmenter is None or self.augment_prob == 0.0)
        if self.precomputed:
            self._precompute_features()
        else:
            self.raw_texts = self.texts.copy()
            self.raw_labels = self.labels.copy()
    
    def _precompute_features(self):
        print("Precomputing features for faster training...")
        
        self.sequences = []
        self.padded_sequences = []
        self.lengths = []
        self.vader_features = []
        
        invalid_count = 0
        for i, text in enumerate(self.texts):
            if not isinstance(text, str) or not text.strip():
                text = "unknown"
                invalid_count += 1
            
            try:
                sequence = self.preprocessor.text_to_sequence(text)
                padded = self.preprocessor.pad_sequence(sequence, self.add_special_tokens)
                length = min(len(sequence), self.preprocessor.max_length)
                
                try:
                    vader = self.preprocessor.compute_vader_features(text)
                except Exception as e:
                    warnings.warn(f"VADER failed for text '{text[:30]}...': {e}. Using zeros.")
                    vader = np.zeros(4, dtype=np.float32)
                
                self.sequences.append(sequence)
                self.padded_sequences.append(padded)
                self.lengths.append(length)
                self.vader_features.append(vader)
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to process text at index {i} ('{text[:50]}...'): {e}"
                )
        
        if invalid_count > 0:
            warnings.warn(f"Found {invalid_count} invalid/empty texts. Replaced with 'unknown'.")
        
        print(f"Precomputed {len(self.sequences)} samples successfully")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.precomputed:
            text = self.texts[idx]
            label = self.labels[idx]
            padded = self.padded_sequences[idx]
            length = self.lengths[idx]
            vader = self.vader_features[idx]
            
            if self.augmenter is not None and np.random.random() < self.augment_prob:
                try:
                    if isinstance(self.augmenter, AugmenterProtocol):
                        augmented = self.augmenter.augment_text(text)
                        if augmented and isinstance(augmented, str) and augmented.strip():
                            text = augmented
                            sequence = self.preprocessor.text_to_sequence(text)
                            padded = self.preprocessor.pad_sequence(sequence, self.add_special_tokens)
                            length = min(len(sequence), self.preprocessor.max_length)
                            vader = self.preprocessor.compute_vader_features(text)
                except Exception as e:
                    warnings.warn(f"Augmentation failed for text '{text[:30]}...': {e}. Using original.")
        else:
            text = self.raw_texts[idx]
            label = self.raw_labels[idx]
            
            if not isinstance(text, str) or not text.strip():
                text = "unknown"
            
            if self.augmenter is not None and np.random.random() < self.augment_prob:
                try:
                    if isinstance(self.augmenter, AugmenterProtocol):
                        augmented = self.augmenter.augment_text(text)
                        if augmented and isinstance(augmented, str) and augmented.strip():
                            text = augmented
                except Exception as e:
                    warnings.warn(f"Augmentation failed: {e}. Using original text.")
            
            try:
                sequence = self.preprocessor.text_to_sequence(text)
                padded = self.preprocessor.pad_sequence(sequence, self.add_special_tokens)
                length = min(len(sequence), self.preprocessor.max_length)
                vader = self.preprocessor.compute_vader_features(text)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to process text at index {idx} ('{text[:50]}...'): {e}"
                )
        
        if not isinstance(padded, (list, np.ndarray)):
            raise RuntimeError(f"Invalid padded sequence type: {type(padded)}")
        if len(padded) != self.preprocessor.max_length:
            raise RuntimeError(
                f"Sequence length mismatch: expected {self.preprocessor.max_length}, got {len(padded)}"
            )
        
        return {
            'text': torch.LongTensor(padded),
            'label': torch.LongTensor([int(label)]),
            'length': torch.LongTensor([int(length)]),
            'sentiment_score': torch.FloatTensor(vader),
            'raw_text': str(text)
        }


class TransformerDataset(Dataset):
    """Dataset for transformer models"""
    
    def __init__(
        self,
        texts: Union[List[str], np.ndarray],
        labels: Union[List[int], np.ndarray, torch.Tensor],
        tokenizer,
        max_length: int = 128,
        augmenter: Optional[AugmenterProtocol] = None,
        augment_prob: float = 0.0,
        num_classes: int = 3
    ):
        if len(texts) == 0:
            raise ValueError("texts cannot be empty")
        if len(labels) == 0:
            raise ValueError("labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(f"Length mismatch: texts ({len(texts)}) != labels ({len(labels)})")
        
        self.texts = np.asarray(texts)
        self.labels = np.asarray(labels).astype(np.int64)
        
        if np.min(self.labels) < 0 or np.max(self.labels) >= num_classes:
            invalid_labels = np.unique(self.labels[(self.labels < 0) | (self.labels >= num_classes)])
            raise ValueError(
                f"Labels must be in range [0, {num_classes-1}]. "
                f"Found invalid labels: {invalid_labels.tolist()}"
            )
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmenter = augmenter
        self.augment_prob = max(0.0, min(1.0, augment_prob))
        self.num_classes = num_classes
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        if not isinstance(text, str) or not text.strip():
            text = "unknown"
        
        if self.augmenter is not None and np.random.random() < self.augment_prob:
            try:
                if isinstance(self.augmenter, AugmenterProtocol):
                    augmented = self.augmenter.augment_text(text)
                    if augmented and isinstance(augmented, str) and augmented.strip():
                        text = augmented
            except Exception as e:
                warnings.warn(f"Augmentation failed: {e}. Using original text.")
        
        try:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        except Exception as e:
            warnings.warn(f"Tokenization failed for text '{text[:30]}...': {e}. Using fallback.")
            encoded = self.tokenizer(
                "unknown",
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.LongTensor([int(label)]),
            'raw_text': str(text)
        }


# ✅ CRITICAL FIX: Return precise per-key types WITHOUT union
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a dictionary with precise per-key types:
      - 'text', 'label', 'length', 'sentiment_score': torch.Tensor
      - 'raw_text': List[str]
    """
    text_list = [item['text'] for item in batch]  # List[Tensor]
    label_list = [item['label'] for item in batch]  # List[Tensor]
    length_list = [item['length'] for item in batch]  # List[Tensor]
    sentiment_list = [item['sentiment_score'] for item in batch]  # List[Tensor]
    raw_texts = [item['raw_text'] for item in batch]  # List[str]
    
    return {
        'text': torch.stack(text_list),
        'label': torch.cat(label_list),
        'length': torch.cat(length_list),
        'sentiment_score': torch.stack(sentiment_list),
        'raw_text': raw_texts  # Pure List[str]
    }


def transformer_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a dictionary with precise per-key types for transformers
    """
    input_ids_list = [item['input_ids'] for item in batch]
    attention_list = [item['attention_mask'] for item in batch]
    label_list = [item['label'] for item in batch]
    raw_texts = [item['raw_text'] for item in batch]
    
    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_list),
        'label': torch.cat(label_list),
        'raw_text': raw_texts
    }


def create_data_loaders(
    data_dict: Dict[str, Dict[str, np.ndarray]],
    preprocessor,
    batch_size: int = 64,
    add_special_tokens: bool = False,
    augmenter: Optional[AugmenterProtocol] = None,
    augment_train_only: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None
) -> Dict[str, DataLoader]:
    required_splits = ['train', 'val', 'test']
    required_keys = ['texts', 'labels']
    
    for split in required_splits:
        if split not in data_dict:
            raise ValueError(f"data_dict missing required split: '{split}'")
        for key in required_keys:
            if key not in data_dict[split]:
                raise ValueError(f"data_dict['{split}'] missing required key: '{key}'")
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    print("\n" + "="*80)
    print("CREATING DATA LOADERS")
    print("="*80)
    
    train_dataset = SentimentDataset(
        texts=data_dict['train']['texts'],
        labels=data_dict['train']['labels'],
        preprocessor=preprocessor,
        add_special_tokens=add_special_tokens,
        augmenter=augmenter if augment_train_only else None,
        augment_prob=0.5 if (augmenter and augment_train_only) else 0.0
    )
    
    val_dataset = SentimentDataset(
        texts=data_dict['val']['texts'],
        labels=data_dict['val']['labels'],
        preprocessor=preprocessor,
        add_special_tokens=add_special_tokens,
        augmenter=None,
        augment_prob=0.0
    )
    
    test_dataset = SentimentDataset(
        texts=data_dict['test']['texts'],
        labels=data_dict['test']['labels'],
        preprocessor=preprocessor,
        add_special_tokens=add_special_tokens,
        augmenter=None,
        augment_prob=0.0
    )
    
    if num_workers > 0 and sys.platform == 'win32':
        warnings.warn(
            "Windows multiprocessing with DataLoader may cause issues. "
            "Consider setting num_workers=0 if you encounter errors."
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,  # Returns Dict[str, Any] with precise runtime types
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")
    
    print(f"\nBatch configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Num workers:   {num_workers}")
    print(f"  Pin memory:    {pin_memory}")
    
    if augmenter and augment_train_only:
        print(f"\nData augmentation: ENABLED (train only)")
        print(f"  Augmentation probability: 50%")
        print(f"  Methods: {getattr(augmenter, 'aug_methods', 'N/A')}")
    else:
        print(f"\nData augmentation: DISABLED")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_transformer_data_loaders(
    data_dict: Dict[str, Dict[str, np.ndarray]],
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    augmenter: Optional[AugmenterProtocol] = None,
    augment_train_only: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None
) -> Dict[str, DataLoader]:
    required_splits = ['train', 'val', 'test']
    required_keys = ['texts', 'labels']
    
    for split in required_splits:
        if split not in data_dict:
            raise ValueError(f"data_dict missing required split: '{split}'")
        for key in required_keys:
            if key not in data_dict[split]:
                raise ValueError(f"data_dict['{split}'] missing required key: '{key}'")
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    print("\n" + "="*80)
    print("CREATING TRANSFORMER DATA LOADERS")
    print("="*80)
    
    train_dataset = TransformerDataset(
        texts=data_dict['train']['texts'],
        labels=data_dict['train']['labels'],
        tokenizer=tokenizer,
        max_length=max_length,
        augmenter=augmenter if augment_train_only else None,
        augment_prob=0.5 if (augmenter and augment_train_only) else 0.0
    )
    
    val_dataset = TransformerDataset(
        texts=data_dict['val']['texts'],
        labels=data_dict['val']['labels'],
        tokenizer=tokenizer,
        max_length=max_length,
        augmenter=None,
        augment_prob=0.0
    )
    
    test_dataset = TransformerDataset(
        texts=data_dict['test']['texts'],
        labels=data_dict['test']['labels'],
        tokenizer=tokenizer,
        max_length=max_length,
        augmenter=None,
        augment_prob=0.0
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=transformer_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=transformer_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=transformer_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")
    
    print(f"\nBatch configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max length: {max_length}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Num workers:   {num_workers}")
    print(f"  Pin memory:    {pin_memory}")
    
    if augmenter and augment_train_only:
        print(f"\nData augmentation: ENABLED (train only)")
        print(f"  Augmentation probability: 50%")
    else:
        print(f"\nData augmentation: DISABLED")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_class_weights(
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    method: str = 'inverse_freq',
    num_classes: int = 3,
    epsilon: float = 1e-6
) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    if labels.size == 0:
        raise ValueError("labels cannot be empty")
    
    class_counts = np.bincount(labels, minlength=num_classes)
    
    if np.any(class_counts == 0):
        warnings.warn(
            f"Class(es) with zero samples detected: {np.where(class_counts == 0)[0].tolist()}. "
            "Using epsilon to avoid division by zero."
        )
        class_counts = np.maximum(class_counts, epsilon)
    
    if method == 'inverse_freq':
        total_samples = len(labels)
        weights = total_samples / (num_classes * (class_counts + epsilon))
        
    elif method == 'effective_num':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts + epsilon)
        weights = (1.0 - beta) / (effective_num + epsilon)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'inverse_freq' or 'effective_num'")
    
    weights = weights / (np.sum(weights) + epsilon) * num_classes
    
    return torch.FloatTensor(weights)


def print_batch_info(batch: Dict[str, Any], dataset_type: str = 'standard'):
    """
    ✅ FIXED: Explicit variable assignment with runtime type safety
    """
    print("\n" + "="*80)
    print("BATCH INFORMATION")
    print("="*80)
    
    if dataset_type == 'standard':
        # Runtime type safety: we KNOW these keys have these types based on collate_fn
        text_tensor = batch['text']          # torch.Tensor
        label_tensor = batch['label']        # torch.Tensor
        length_tensor = batch['length']      # torch.Tensor
        sentiment_tensor = batch['sentiment_score']  # torch.Tensor
        raw_texts = batch['raw_text']        # List[str]
        
        print(f"Text shape: {text_tensor.shape} (dtype: {text_tensor.dtype})")
        print(f"Label shape: {label_tensor.shape} (dtype: {label_tensor.dtype})")
        print(f"Length shape: {length_tensor.shape} (dtype: {length_tensor.dtype})")
        print(f"Sentiment score shape: {sentiment_tensor.shape} (dtype: {sentiment_tensor.dtype})")
        
        print(f"\nSample text (first in batch):")
        print(f"  Tokens (first 20): {text_tensor[0][:20].tolist()}...")
        print(f"  Label: {label_tensor[0].item()}")
        print(f"  Length: {length_tensor[0].item()}")
        print(f"  VADER scores: {sentiment_tensor[0].tolist()}")
        print(f"  Raw text: {raw_texts[0][:100]}...")
        
    elif dataset_type == 'transformer':
        input_ids = batch['input_ids']           # torch.Tensor
        attention_mask = batch['attention_mask'] # torch.Tensor
        label_tensor = batch['label']            # torch.Tensor
        raw_texts = batch['raw_text']            # List[str]
        
        print(f"Input IDs shape: {input_ids.shape} (dtype: {input_ids.dtype})")
        print(f"Attention mask shape: {attention_mask.shape} (dtype: {attention_mask.dtype})")
        print(f"Label shape: {label_tensor.shape} (dtype: {label_tensor.dtype})")
        
        print(f"\nSample (first in batch):")
        print(f"  Input IDs (first 20): {input_ids[0][:20].tolist()}...")
        print(f"  Attention mask sum: {attention_mask[0].sum().item()}")
        print(f"  Label: {label_tensor[0].item()}")
        print(f"  Raw text: {raw_texts[0][:100]}...")
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'standard' or 'transformer'")


if sys.platform == 'win32':
    warnings.warn(
        "Windows detected: Setting default num_workers=0 for DataLoader to avoid multiprocessing issues. "
        "You can override this in create_data_loaders() if needed."
    )


if __name__ == "__main__":
    print("="*80)
    print("TESTING DATASET MODULE")
    print("="*80)
    
    labels = np.array([0]*100 + [1]*500 + [2]*200)
    
    weights_inv = get_class_weights(labels, method='inverse_freq')
    print(f"\nInverse frequency weights: {weights_inv.numpy()}")
    
    weights_eff = get_class_weights(labels, method='effective_num')
    print(f"Effective number weights: {weights_eff.numpy()}")
    
    try:
        get_class_weights(np.array([]))
    except ValueError as e:
        print(f"✓ Correctly rejected empty labels: {e}")
    
    try:
        get_class_weights(labels, method='unknown')
    except ValueError as e:
        print(f"✓ Correctly rejected invalid method: {e}")
    
    print("\n✅ Dataset module tested successfully!")