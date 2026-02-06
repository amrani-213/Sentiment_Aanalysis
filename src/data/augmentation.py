"""
Text augmentation module for sentiment analysis
Handles synonym replacement, word swapping, deletion, insertion, and back-translation
"""
import random
import numpy as np
import torch
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from tqdm import tqdm


class TextAugmenter:
    
    def __init__(self, aug_methods=['synonym', 'swap', 'delete'], aug_p=0.1):
        self.aug_methods = aug_methods
        self.aug_p = aug_p
        
        if 'synonym' in aug_methods:
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=aug_p)
        
        if 'backtranslation' in aug_methods:
            try:
                self.back_trans_aug = naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de',
                    to_model_name='facebook/wmt19-de-en',
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            except Exception as e:
                print(f"Warning: Back-translation not available: {e}")
                if 'backtranslation' in self.aug_methods:
                    self.aug_methods.remove('backtranslation')
    
    def random_swap(self, text, n=2):
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text, p=0.1):
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_insertion(self, text, n=1):
        words = text.split()
        if len(words) == 0:
            return text
        
        for _ in range(n):
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def synonym_replacement(self, text):
        if 'synonym' in self.aug_methods and hasattr(self, 'synonym_aug'):
            try:
                augmented = self.synonym_aug.augment(text)
                
                # ✅ FIX: Handle nlpaug's variable return types safely
                if augmented is None:
                    return text
                elif isinstance(augmented, list):
                    # nlpaug sometimes returns list even with n=1
                    if len(augmented) > 0 and isinstance(augmented[0], str) and augmented[0].strip():
                        return augmented[0]
                    else:
                        return text
                elif isinstance(augmented, str) and augmented.strip():
                    return augmented
                else:
                    return text
            except Exception as e:
                print(f"Synonym augmentation failed for text '{text[:50]}...': {e}")
                return text
        return text
    
    def back_translation(self, text):
        if 'backtranslation' in self.aug_methods and hasattr(self, 'back_trans_aug'):
            try:
                augmented = self.back_trans_aug.augment(text)
                
                # ✅ FIX: Same safe handling as above
                if augmented is None:
                    return text
                elif isinstance(augmented, list):
                    if len(augmented) > 0 and isinstance(augmented[0], str) and augmented[0].strip():
                        return augmented[0]
                    else:
                        return text
                elif isinstance(augmented, str) and augmented.strip():
                    return augmented
                else:
                    return text
            except Exception as e:
                print(f"Back-translation failed for text '{text[:50]}...': {e}")
                return text
        return text
    
    def augment_text(self, text, method=None):
        if not text or not isinstance(text, str) or not text.strip():
            return text
        
        if method is None:
            method = random.choice(self.aug_methods)
        
        if method == 'swap':
            return self.random_swap(text, n=max(1, len(text.split()) // 10))
        elif method == 'delete':
            return self.random_deletion(text, p=self.aug_p)
        elif method == 'insert':
            return self.random_insertion(text, n=max(1, len(text.split()) // 10))
        elif method == 'synonym':
            return self.synonym_replacement(text)
        elif method == 'backtranslation':
            return self.back_translation(text)
        else:
            return text
    
    def augment_dataset(self, texts, labels, n_aug=1, keep_original=True):
        print(f"Augmenting dataset (n_aug={n_aug}, methods={self.aug_methods})...")
        
        augmented_texts = []
        augmented_labels = []
        
        if keep_original:
            augmented_texts.extend(texts)
            augmented_labels.extend(labels)
        
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Augmenting"):
            for _ in range(n_aug):
                aug_text = self.augment_text(text)
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
        
        print(f"Original size: {len(texts)}")
        print(f"Augmented size: {len(augmented_texts)}")
        
        return np.array(augmented_texts), np.array(augmented_labels)
    
    def augment_minority_classes(self, texts, labels, target_ratio=1.0):
        from collections import Counter
        
        class_counts = Counter(labels)
        max_count = max(class_counts.values())
        
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        for class_label, count in class_counts.items():
            target_count = int(max_count * target_ratio)
            n_to_generate = max(0, target_count - count)
            
            if n_to_generate > 0:
                class_texts = [text for text, label in zip(texts, labels) if label == class_label]
                
                for _ in tqdm(range(n_to_generate), desc=f"Augmenting class {class_label}"):
                    text = random.choice(class_texts)
                    aug_text = self.augment_text(text)
                    augmented_texts.append(aug_text)
                    augmented_labels.append(class_label)
        
        print(f"\nOriginal size: {len(texts)}")
        print(f"Augmented size: {len(augmented_texts)}")
        print(f"Original distribution: {Counter(labels)}")
        print(f"Augmented distribution: {Counter(augmented_labels)}")
        
        return np.array(augmented_texts), np.array(augmented_labels)


def augment_training_data(data_dict, n_aug=1, balance_classes=True, aug_methods=['synonym', 'swap', 'delete']):
    """
    Augment training data to handle class imbalance
    
    Args:
        data_dict: Dictionary with 'train' key containing 'texts' and 'labels'
        n_aug: Number of augmentations per sample (if not balancing)
        balance_classes: If True, augment minority classes to match majority
        aug_methods: List of augmentation methods to use
    
    Returns:
        Updated data_dict with augmented training data
    """
    augmenter = TextAugmenter(aug_methods=aug_methods, aug_p=0.1)
    
    train_texts = data_dict['train']['texts']
    train_labels = data_dict['train']['labels']
    
    if balance_classes:
        aug_texts, aug_labels = augmenter.augment_minority_classes(
            train_texts, train_labels, target_ratio=1.0
        )
    else:
        aug_texts, aug_labels = augmenter.augment_dataset(
            train_texts, train_labels, n_aug=n_aug, keep_original=True
        )
    
    data_dict['train']['texts'] = aug_texts
    data_dict['train']['labels'] = aug_labels
    
    return data_dict