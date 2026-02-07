<<<<<<< HEAD
"""
Utility Script: Save Preprocessed and Augmented Data
Run this to generate CSV files in data/processed and data/augmented
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, 'src')

from src.data.preprocessing import EnhancedTextPreprocessor, prepare_data
from src.data.augmentation import TextAugmenter

# Paths
DATA_PATH = "data/raw/sentiment_dataset.csv"
VOCAB_PATH = "results/baseline/vocabulary.pkl"
PROCESSED_DIR = Path("data/processed")
AUGMENTED_DIR = Path("data/augmented")

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)


def save_preprocessed_data():
    """Load raw data, preprocess it, and save to CSV"""
    print("="*80)
    print("SAVING PREPROCESSED DATA")
    print("="*80)
    
    # Load raw data
    print(f"\n1. Loading raw data from: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df_raw)} samples")
    
    # Load preprocessor with vocabulary
    print(f"\n2. Loading preprocessor and vocabulary from: {VOCAB_PATH}")
    preprocessor = EnhancedTextPreprocessor(
        vocab_size=10000,
        max_length=100,
        min_freq=2,
        use_spell_check=False,
        use_lemmatization=False
    )
    preprocessor.load_vocabulary(VOCAB_PATH)
    print(f"   Vocabulary size: {len(preprocessor.word2idx)}")
    
    # Preprocess all texts
    print(f"\n3. Preprocessing texts...")
    df_raw['cleaned_text'] = df_raw['text'].apply(preprocessor.clean_text)
    
    # Add VADER features
    print(f"\n4. Computing VADER sentiment scores...")
    vader_scores = df_raw['text'].apply(preprocessor.compute_vader_features)
    df_raw['vader_compound'] = [scores[0] for scores in vader_scores]
    df_raw['vader_pos'] = [scores[1] for scores in vader_scores]
    df_raw['vader_neu'] = [scores[2] for scores in vader_scores]
    df_raw['vader_neg'] = [scores[3] for scores in vader_scores]
    
    # Add text features
    print(f"\n5. Extracting text features...")
    text_features = df_raw['text'].apply(preprocessor.extract_text_features)
    df_raw['text_length'] = [f['length'] for f in text_features]
    df_raw['char_length'] = [f['char_length'] for f in text_features]
    df_raw['caps_ratio'] = [f['caps_ratio'] for f in text_features]
    df_raw['punct_ratio'] = [f['punct_ratio'] for f in text_features]
    df_raw['avg_word_length'] = [f['avg_word_length'] for f in text_features]
    
    # Save full preprocessed data
    output_path = PROCESSED_DIR / "full_preprocessed.csv"
    print(f"\n6. Saving preprocessed data to: {output_path}")
    df_raw.to_csv(output_path, index=False)
    print(f"   Saved {len(df_raw)} samples")
    
    # Also save train/val/test splits
    print(f"\n7. Creating and saving train/val/test splits...")
    data_dict = prepare_data(DATA_PATH, test_size=0.1, val_size=0.1, random_state=42)
    
    for split_name in ['train', 'val', 'test']:
        split_df = pd.DataFrame({
            'text': data_dict[split_name]['texts'],
            'label': data_dict[split_name]['labels']
        })
        
        # Add cleaned text
        split_df['cleaned_text'] = split_df['text'].apply(preprocessor.clean_text)
        
        # Get sentiment from label map
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        split_df['sentiment'] = split_df['label'].map(label_map)
        
        # Save
        split_path = PROCESSED_DIR / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        print(f"   - {split_name}: {len(split_df)} samples → {split_path}")
    
    print("\n✅ Preprocessed data saved successfully!")
    return df_raw


def save_augmented_data():
    """Load preprocessed data, apply augmentation, and save"""
    print("\n" + "="*80)
    print("SAVING AUGMENTED DATA")
    print("="*80)
    
    # Load preprocessed train split
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        print(f"❌ Error: {train_path} not found. Run save_preprocessed_data() first.")
        return
    
    print(f"\n1. Loading preprocessed train data from: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"   Loaded {len(df_train)} samples")
    
    # Show class distribution
    print(f"\n   Original class distribution:")
    print(df_train['label'].value_counts().sort_index())
    
    # Create augmenter
    print(f"\n2. Creating text augmenter...")
    augmenter = TextAugmenter(
        aug_methods=['synonym', 'swap', 'delete'],
        aug_p=0.1
    )
    
    # Augment minority classes
    print(f"\n3. Augmenting data to balance classes...")
    aug_texts, aug_labels = augmenter.augment_minority_classes(
        df_train['text'].values,
        df_train['label'].values,
        target_ratio=1.0  # Balance all classes
    )
    
    # Create augmented dataframe
    df_augmented = pd.DataFrame({
        'text': aug_texts,
        'label': aug_labels
    })
    
    # Add sentiment column
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df_augmented['sentiment'] = df_augmented['label'].map(label_map)
    
    # Mark original vs augmented samples
    df_augmented['is_augmented'] = False
    df_augmented.loc[len(df_train):, 'is_augmented'] = True
    
    # Save augmented data
    output_path = AUGMENTED_DIR / "train_augmented.csv"
    print(f"\n4. Saving augmented data to: {output_path}")
    df_augmented.to_csv(output_path, index=False)
    print(f"   Saved {len(df_augmented)} samples")
    print(f"   - Original: {len(df_train)}")
    print(f"   - Augmented: {len(df_augmented) - len(df_train)}")
    
    print(f"\n   Augmented class distribution:")
    print(df_augmented['label'].value_counts().sort_index())
    
    # Save just the newly augmented samples
    df_new_only = df_augmented[df_augmented['is_augmented']]
    new_only_path = AUGMENTED_DIR / "train_new_augmented_only.csv"
    df_new_only.to_csv(new_only_path, index=False)
    print(f"\n5. Saved new augmented samples only to: {new_only_path}")
    print(f"   {len(df_new_only)} new samples")
    
    print("\n✅ Augmented data saved successfully!")
    return df_augmented


def main():
    """Main function to save all processed data"""
    print("="*80)
    print("DATA PROCESSING PIPELINE")
    print("="*80)
    print("\nThis script will:")
    print("1. Load raw data and preprocess it")
    print("2. Save preprocessed data to data/processed/")
    print("3. Create train/val/test splits")
    print("4. Apply data augmentation to training data")
    print("5. Save augmented data to data/augmented/")
    print("\n" + "="*80)
    
    try:
        # Step 1: Save preprocessed data
        df_preprocessed = save_preprocessed_data()
        
        # Step 2: Save augmented data
        df_augmented = save_augmented_data()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("\n✅ All data saved successfully!")
        print(f"\nFiles created:")
        print(f"  data/processed/full_preprocessed.csv")
        print(f"  data/processed/train.csv")
        print(f"  data/processed/val.csv")
        print(f"  data/processed/test.csv")
        print(f"  data/augmented/train_augmented.csv")
        print(f"  data/augmented/train_new_augmented_only.csv")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure these files exist:")
        print(f"  - {DATA_PATH}")
        print(f"  - {VOCAB_PATH}")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
=======
"""
Utility Script: Save Preprocessed and Augmented Data
Run this to generate CSV files in data/processed and data/augmented
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, 'src')

from src.data.preprocessing import EnhancedTextPreprocessor, prepare_data
from src.data.augmentation import TextAugmenter

# Paths
DATA_PATH = "data/raw/sentiment_dataset.csv"
VOCAB_PATH = "results/baseline/vocabulary.pkl"
PROCESSED_DIR = Path("data/processed")
AUGMENTED_DIR = Path("data/augmented")

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)


def save_preprocessed_data():
    """Load raw data, preprocess it, and save to CSV"""
    print("="*80)
    print("SAVING PREPROCESSED DATA")
    print("="*80)
    
    # Load raw data
    print(f"\n1. Loading raw data from: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df_raw)} samples")
    
    # Load preprocessor with vocabulary
    print(f"\n2. Loading preprocessor and vocabulary from: {VOCAB_PATH}")
    preprocessor = EnhancedTextPreprocessor(
        vocab_size=10000,
        max_length=100,
        min_freq=2,
        use_spell_check=False,
        use_lemmatization=False
    )
    preprocessor.load_vocabulary(VOCAB_PATH)
    print(f"   Vocabulary size: {len(preprocessor.word2idx)}")
    
    # Preprocess all texts
    print(f"\n3. Preprocessing texts...")
    df_raw['cleaned_text'] = df_raw['text'].apply(preprocessor.clean_text)
    
    # Add VADER features
    print(f"\n4. Computing VADER sentiment scores...")
    vader_scores = df_raw['text'].apply(preprocessor.compute_vader_features)
    df_raw['vader_compound'] = [scores[0] for scores in vader_scores]
    df_raw['vader_pos'] = [scores[1] for scores in vader_scores]
    df_raw['vader_neu'] = [scores[2] for scores in vader_scores]
    df_raw['vader_neg'] = [scores[3] for scores in vader_scores]
    
    # Add text features
    print(f"\n5. Extracting text features...")
    text_features = df_raw['text'].apply(preprocessor.extract_text_features)
    df_raw['text_length'] = [f['length'] for f in text_features]
    df_raw['char_length'] = [f['char_length'] for f in text_features]
    df_raw['caps_ratio'] = [f['caps_ratio'] for f in text_features]
    df_raw['punct_ratio'] = [f['punct_ratio'] for f in text_features]
    df_raw['avg_word_length'] = [f['avg_word_length'] for f in text_features]
    
    # Save full preprocessed data
    output_path = PROCESSED_DIR / "full_preprocessed.csv"
    print(f"\n6. Saving preprocessed data to: {output_path}")
    df_raw.to_csv(output_path, index=False)
    print(f"   Saved {len(df_raw)} samples")
    
    # Also save train/val/test splits
    print(f"\n7. Creating and saving train/val/test splits...")
    data_dict = prepare_data(DATA_PATH, test_size=0.1, val_size=0.1, random_state=42)
    
    for split_name in ['train', 'val', 'test']:
        split_df = pd.DataFrame({
            'text': data_dict[split_name]['texts'],
            'label': data_dict[split_name]['labels']
        })
        
        # Add cleaned text
        split_df['cleaned_text'] = split_df['text'].apply(preprocessor.clean_text)
        
        # Get sentiment from label map
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        split_df['sentiment'] = split_df['label'].map(label_map)
        
        # Save
        split_path = PROCESSED_DIR / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        print(f"   - {split_name}: {len(split_df)} samples → {split_path}")
    
    print("\n✅ Preprocessed data saved successfully!")
    return df_raw


def save_augmented_data():
    """Load preprocessed data, apply augmentation, and save"""
    print("\n" + "="*80)
    print("SAVING AUGMENTED DATA")
    print("="*80)
    
    # Load preprocessed train split
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        print(f"❌ Error: {train_path} not found. Run save_preprocessed_data() first.")
        return
    
    print(f"\n1. Loading preprocessed train data from: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"   Loaded {len(df_train)} samples")
    
    # Show class distribution
    print(f"\n   Original class distribution:")
    print(df_train['label'].value_counts().sort_index())
    
    # Create augmenter
    print(f"\n2. Creating text augmenter...")
    augmenter = TextAugmenter(
        aug_methods=['synonym', 'swap', 'delete'],
        aug_p=0.1
    )
    
    # Augment minority classes
    print(f"\n3. Augmenting data to balance classes...")
    aug_texts, aug_labels = augmenter.augment_minority_classes(
        df_train['text'].values,
        df_train['label'].values,
        target_ratio=1.0  # Balance all classes
    )
    
    # Create augmented dataframe
    df_augmented = pd.DataFrame({
        'text': aug_texts,
        'label': aug_labels
    })
    
    # Add sentiment column
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df_augmented['sentiment'] = df_augmented['label'].map(label_map)
    
    # Mark original vs augmented samples
    df_augmented['is_augmented'] = False
    df_augmented.loc[len(df_train):, 'is_augmented'] = True
    
    # Save augmented data
    output_path = AUGMENTED_DIR / "train_augmented.csv"
    print(f"\n4. Saving augmented data to: {output_path}")
    df_augmented.to_csv(output_path, index=False)
    print(f"   Saved {len(df_augmented)} samples")
    print(f"   - Original: {len(df_train)}")
    print(f"   - Augmented: {len(df_augmented) - len(df_train)}")
    
    print(f"\n   Augmented class distribution:")
    print(df_augmented['label'].value_counts().sort_index())
    
    # Save just the newly augmented samples
    df_new_only = df_augmented[df_augmented['is_augmented']]
    new_only_path = AUGMENTED_DIR / "train_new_augmented_only.csv"
    df_new_only.to_csv(new_only_path, index=False)
    print(f"\n5. Saved new augmented samples only to: {new_only_path}")
    print(f"   {len(df_new_only)} new samples")
    
    print("\n✅ Augmented data saved successfully!")
    return df_augmented


def main():
    """Main function to save all processed data"""
    print("="*80)
    print("DATA PROCESSING PIPELINE")
    print("="*80)
    print("\nThis script will:")
    print("1. Load raw data and preprocess it")
    print("2. Save preprocessed data to data/processed/")
    print("3. Create train/val/test splits")
    print("4. Apply data augmentation to training data")
    print("5. Save augmented data to data/augmented/")
    print("\n" + "="*80)
    
    try:
        # Step 1: Save preprocessed data
        df_preprocessed = save_preprocessed_data()
        
        # Step 2: Save augmented data
        df_augmented = save_augmented_data()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("\n✅ All data saved successfully!")
        print(f"\nFiles created:")
        print(f"  data/processed/full_preprocessed.csv")
        print(f"  data/processed/train.csv")
        print(f"  data/processed/val.csv")
        print(f"  data/processed/test.csv")
        print(f"  data/augmented/train_augmented.csv")
        print(f"  data/augmented/train_new_augmented_only.csv")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure these files exist:")
        print(f"  - {DATA_PATH}")
        print(f"  - {VOCAB_PATH}")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
>>>>>>> ffa323fbcc88d44d2b99039bb7823d62917b21ad
    main()