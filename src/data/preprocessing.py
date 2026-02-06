"""
Text preprocessing and dataset utilities for sentiment analysis
Handles cleaning, vocabulary building, VADER features, and PyTorch dataset creation
"""
import re
import string
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from autocorrect import Speller
import torch
from torch.utils.data import Dataset, DataLoader


class EnhancedTextPreprocessor:
    
    def __init__(self, vocab_size=10000, max_length=100, min_freq=2, 
                 use_spell_check=False, use_lemmatization=False):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_freq = min_freq
        self.use_spell_check = use_spell_check
        self.use_lemmatization = use_lemmatization
        
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        
        self.special_tokens = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        
        self.vader = SentimentIntensityAnalyzer()
        
        if self.use_spell_check:
            self.spell = Speller(lang='en')
        
        self.negations = [
            "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
            "none", "hardly", "scarcely", "barely", "doesn't", "isn't", "wasn't",
            "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't"
        ]
        
        self.slang_dict = {
            "u": "you", "ur": "your", "r": "are", "y": "why",
            "btw": "by the way", "idk": "i do not know", "imo": "in my opinion",
            "tbh": "to be honest", "lol": "laughing out loud", "omg": "oh my god",
            "wtf": "what the fuck", "smh": "shaking my head", "fyi": "for your information",
            "asap": "as soon as possible", "brb": "be right back", "irl": "in real life",
            "jk": "just kidding", "nvm": "never mind", "pls": "please", "thx": "thanks",
            "gonna": "going to", "wanna": "want to", "gotta": "got to",
            "kinda": "kind of", "sorta": "sort of", "dunno": "do not know",
            "yeah": "yes", "yep": "yes", "nope": "no", "yup": "yes"
        }
        
        self.contractions = {
            "can't": "can not", "won't": "will not", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "i'm": "i am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
            "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "i'd": "i would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "doesn't": "does not", "don't": "do not", "didn't": "did not",
            "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not"
        }
    
    def clean_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower()
        
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        words = text.split()
        words = [self.slang_dict.get(word, word) for word in words]
        text = ' '.join(words)
        
        text = self.handle_negations_v2(text)
        
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        text = re.sub(r'[^a-zA-Z0-9\s_]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.use_spell_check:
            words = text.split()
            words = [self.spell(word) if len(word) > 2 else word for word in words]
            text = ' '.join(words)
        
        return text
    
    def handle_negations_v2(self, text, window=3):
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            current_word = words[i].lower()
            is_negation = any(neg in current_word for neg in self.negations)
            
            if is_negation:
                result.append(current_word)
                marked = 0
                for j in range(1, min(window + 1, len(words) - i)):
                    next_word = words[i + j].lower()
                    if any(neg in next_word for neg in self.negations):
                        break
                    result.append(f"NEG_{words[i + j]}")
                    marked = j
                i += marked + 1
            else:
                result.append(words[i])
                i += 1
        
        return ' '.join(result)
    
    def compute_vader_features(self, text):
        try:
            scores = self.vader.polarity_scores(text)
            return np.array([
                scores['compound'],
                scores['pos'],
                scores['neu'],
                scores['neg']
            ], dtype=np.float32)
        except Exception as e:
            print(f"VADER error for text '{text[:50]}...': {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def extract_text_features(self, text):
        features = {}
        features['length'] = len(text.split())
        features['char_length'] = len(text)
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['punct_ratio'] = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        return features
    
    def build_vocabulary(self, texts):
        print("Building vocabulary...")
        
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            self.word_counts.update(words)
        
        for token in self.special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
        
        filtered_words = [word for word, count in self.word_counts.most_common() 
                         if count >= self.min_freq]
        
        for word in filtered_words[:self.vocab_size - len(self.special_tokens)]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Vocabulary built: {len(self.word2idx)} words")
        print(f"Total word occurrences: {sum(self.word_counts.values())}")
        print(f"Unique words before filtering: {len(self.word_counts)}")
    
    def text_to_sequence(self, text):
        cleaned = self.clean_text(text)
        words = cleaned.split()
        unk_idx = self.word2idx[self.unk_token]
        sequence = [self.word2idx.get(word, unk_idx) for word in words]
        return sequence
    
    def pad_sequence(self, sequence, add_special_tokens=False):
        if add_special_tokens:
            sos_idx = self.word2idx[self.sos_token]
            eos_idx = self.word2idx[self.eos_token]
            sequence = [sos_idx] + sequence + [eos_idx]
        
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        pad_idx = self.word2idx[self.pad_token]
        padded = sequence + [pad_idx] * (self.max_length - len(sequence))
        
        return padded
    
    def get_vocab_size(self):
        return len(self.word2idx)
    
    def get_padding_idx(self):
        return self.word2idx[self.pad_token]
    
    def save_vocabulary(self, path):
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_counts': self.word_counts,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'min_freq': self.min_freq
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {path}")
    
    def load_vocabulary(self, path):
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.word_counts = vocab_data['word_counts']
        self.vocab_size = vocab_data['vocab_size']
        self.max_length = vocab_data['max_length']
        self.min_freq = vocab_data['min_freq']
        
        print(f"Vocabulary loaded from {path}")
        print(f"Vocabulary size: {len(self.word2idx)}")


def prepare_data(data_path, test_size=0.1, val_size=0.1, random_state=42):
    """
    Load and split dataset with proper stratification
    
    Args:
        data_path: Path to CSV file
        test_size: Proportion for test set
        val_size: Proportion for validation set (of remaining after test)
        random_state: Seed for reproducibility
    
    Returns:
        Dictionary with train/val/test splits
    """
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    df = pd.read_csv(data_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nClass distribution:")
    print(df['sentiment'].value_counts())
    print(f"\nClass distribution (%):")
    print(df['sentiment'].value_counts(normalize=True) * 100)
    
    # ✅ CRITICAL FIX: Convert to numpy arrays explicitly for sklearn compatibility
    # This resolves the type checker error and ensures stratify works correctly
    texts = np.asarray(df['text'].values)  # Explicit conversion to ndarray
    labels = np.asarray(df['label'].values)  # Explicit conversion to ndarray
    
    # Verify stratification is possible
    label_counts = np.bincount(labels)
    min_class_count = np.min(label_counts)
    
    if min_class_count < 2:
        raise ValueError(
            f"Stratification impossible: class with label {np.argmin(label_counts)} "
            f"has only {min_class_count} samples (need at least 2 per class for train/test split)"
        )
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels  # ✅ Now safe with explicit ndarray conversion
    )
    
    # Second split: train vs val (adjust val_size proportionally)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        random_state=random_state, 
        stratify=y_temp  # ✅ Safe with ndarray
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(texts)*100:.1f}%)")
    
    # Verify stratification worked
    print("\nStratification check (label distribution %):")
    print(f"  Train: {np.bincount(y_train) / len(y_train) * 100}")
    print(f"  Val:   {np.bincount(y_val) / len(y_val) * 100}")
    print(f"  Test:  {np.bincount(y_test) / len(y_test) * 100}")
    
    data_dict = {
        'train': {'texts': X_train, 'labels': y_train},
        'val': {'texts': X_val, 'labels': y_val},
        'test': {'texts': X_test, 'labels': y_test}
    }
    
    return data_dict


class SentimentDataset(Dataset):
    """
    PyTorch dataset for sentiment analysis with VADER features
    """
    
    def __init__(self, texts, labels, preprocessor, add_special_tokens=False):
        # ✅ Ensure texts and labels are numpy arrays for consistency
        self.texts = np.asarray(texts)
        self.labels = np.asarray(labels)
        self.preprocessor = preprocessor
        self.add_special_tokens = add_special_tokens
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Handle non-string inputs gracefully
        if not isinstance(text, str):
            text = str(text)
        
        sequence = self.preprocessor.text_to_sequence(text)
        padded = self.preprocessor.pad_sequence(sequence, self.add_special_tokens)
        length = min(len(sequence), self.preprocessor.max_length)
        
        vader_features = self.preprocessor.compute_vader_features(text)
        
        return {
            'text': torch.LongTensor(padded),
            'label': torch.LongTensor([label]),
            'length': torch.LongTensor([length]),
            'sentiment_score': torch.FloatTensor(vader_features)
        }


def create_data_loaders(data_dict, preprocessor, batch_size=64, add_special_tokens=False):
    """
    Create PyTorch DataLoaders for train/val/test splits
    
    Args:
        data_dict: Dictionary from prepare_data()
        preprocessor: EnhancedTextPreprocessor instance
        batch_size: Batch size for DataLoaders
        add_special_tokens: Whether to add SOS/EOS tokens
    
    Returns:
        Dictionary of DataLoaders
    """
    print("\nCreating data loaders...")
    
    train_dataset = SentimentDataset(
        data_dict['train']['texts'],
        data_dict['train']['labels'],
        preprocessor,
        add_special_tokens
    )
    
    val_dataset = SentimentDataset(
        data_dict['val']['texts'],
        data_dict['val']['labels'],
        preprocessor,
        add_special_tokens
    )
    
    test_dataset = SentimentDataset(
        data_dict['test']['texts'],
        data_dict['test']['labels'],
        preprocessor,
        add_special_tokens
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }