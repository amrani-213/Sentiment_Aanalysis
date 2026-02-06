import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from wordcloud import WordCloud
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocessing import EnhancedTextPreprocessor

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(data_path='data/raw/sentiment_dataset.csv'):
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def basic_statistics(df):
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nClass Distribution:")
    print(df['sentiment'].value_counts())
    print("\nClass Distribution (%):")
    print(df['sentiment'].value_counts(normalize=True) * 100)
    
    print("\nLabel Distribution:")
    print(df['label'].value_counts().sort_index())


def text_length_analysis(df, output_dir='results/eda'):
    print("\n" + "="*80)
    print("TEXT LENGTH ANALYSIS")
    print("="*80)
    
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    df['char_length'] = df['text'].apply(lambda x: len(str(x)))
    
    print("\nWord Count Statistics:")
    print(df['text_length'].describe())
    
    print("\nCharacter Count Statistics:")
    print(df['char_length'].describe())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].hist(df['text_length'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of Words', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Text Length (Words)', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(df['text_length'].mean(), color='r', linestyle='--', label=f'Mean: {df["text_length"].mean():.1f}')
    axes[0, 0].legend()
    
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]['text_length']
        axes[0, 1].hist(subset, bins=30, alpha=0.5, label=sentiment, edgecolor='black')
    axes[0, 1].set_xlabel('Number of Words', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Text Length by Sentiment', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    
    sentiment_order = ['negative', 'neutral', 'positive']
    df_ordered = df.copy()
    df_ordered['sentiment'] = pd.Categorical(df_ordered['sentiment'], categories=sentiment_order, ordered=True)
    sns.boxplot(data=df_ordered, x='sentiment', y='text_length', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Sentiment', fontsize=12)
    axes[1, 0].set_ylabel('Number of Words', fontsize=12)
    axes[1, 0].set_title('Text Length Distribution by Sentiment', fontsize=14, fontweight='bold')
    
    axes[1, 1].scatter(df['text_length'], df['char_length'], alpha=0.3)
    axes[1, 1].set_xlabel('Word Count', fontsize=12)
    axes[1, 1].set_ylabel('Character Count', fontsize=12)
    axes[1, 1].set_title('Word Count vs Character Count', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/text_length_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/text_length_analysis.png")
    plt.close()
    
    return df


def class_distribution_analysis(df, output_dir='results/eda'):
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=colors, edgecolor='black')
    axes[0].set_xlabel('Sentiment', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Class Distribution (Counts)', fontsize=14, fontweight='bold')
    for i, v in enumerate(sentiment_counts.values):
        axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=(0.05, 0.05, 0.05))
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    label_counts = df['label'].value_counts().sort_index()
    axes[2].bar(label_counts.index, label_counts.values, color=colors, edgecolor='black')
    axes[2].set_xlabel('Label', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_title('Label Distribution (0=Neg, 1=Neu, 2=Pos)', fontsize=14, fontweight='bold')
    axes[2].set_xticks([0, 1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/class_distribution.png")
    plt.close()


def word_frequency_analysis(df, output_dir='results/eda', top_n=20):
    print("\n" + "="*80)
    print("WORD FREQUENCY ANALYSIS")
    print("="*80)
    
    preprocessor = EnhancedTextPreprocessor()
    
    # Define sentiment order and colors upfront (FIXED: moved to top)
    sentiment_order = ['negative', 'neutral', 'positive']
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']  # red, gray, green
    
    all_words = []
    sentiment_words = {sentiment: [] for sentiment in sentiment_order}
    
    for idx, row in df.iterrows():
        cleaned = preprocessor.clean_text(row['text'])
        words = cleaned.split()
        all_words.extend(words)
        if row['sentiment'] in sentiment_words:
            sentiment_words[row['sentiment']].extend(words)
    
    overall_freq = Counter(all_words)
    
    print(f"\nTotal words (after cleaning): {len(all_words)}")
    print(f"Unique words: {len(overall_freq)}")
    print(f"\nTop {top_n} most common words:")
    for word, count in overall_freq.most_common(top_n):
        print(f"  {word}: {count}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall top words
    top_words = overall_freq.most_common(top_n)
    words, counts = zip(*top_words)
    axes[0, 0].barh(range(len(words)), counts, color='steelblue', edgecolor='black')
    axes[0, 0].set_yticks(range(len(words)))
    axes[0, 0].set_yticklabels(words)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'Top {top_n} Most Common Words', fontsize=14, fontweight='bold')
    
    # Per-sentiment top words (FIXED: safe iteration with explicit order)
    for sentiment_idx, sentiment in enumerate(sentiment_order):
        if sentiment not in sentiment_words:
            continue
            
        words_list = sentiment_words[sentiment]
        freq = Counter(words_list)
        top = freq.most_common(15)
        if not top:
            continue
            
        words, counts = zip(*top)
        
        # Map to subplot positions: negative → [0,1], neutral → [1,0], positive → [1,1]
        if sentiment_idx == 0:
            ax = axes[0, 1]
        elif sentiment_idx == 1:
            ax = axes[1, 0]
        else:
            ax = axes[1, 1]
            
        ax.barh(range(len(words)), counts, color=colors[sentiment_idx], edgecolor='black')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(f'Top 15 Words: {sentiment.capitalize()}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/word_frequency.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/word_frequency.png")
    plt.close()


def generate_wordclouds(df, output_dir='results/eda'):
    print("\n" + "="*80)
    print("GENERATING WORD CLOUDS")
    print("="*80)
    
    preprocessor = EnhancedTextPreprocessor()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sentiments = ['negative', 'neutral', 'positive']
    colors_map = ['Reds', 'Greys', 'Greens']
    
    for idx, sentiment in enumerate(sentiments):
        texts = df[df['sentiment'] == sentiment]['text'].values
        cleaned_texts = [preprocessor.clean_text(text) for text in texts]
        all_text = ' '.join(cleaned_texts)
        
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap=colors_map[idx],
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(f'{sentiment.capitalize()} Sentiment', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wordclouds.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/wordclouds.png")
    plt.close()


def vader_sentiment_analysis(df, output_dir='results/eda'):
    print("\n" + "="*80)
    print("VADER SENTIMENT ANALYSIS")
    print("="*80)
    
    preprocessor = EnhancedTextPreprocessor()
    
    vader_scores = []
    for text in df['text']:
        scores = preprocessor.compute_vader_features(text)
        vader_scores.append(scores)
    
    vader_df = pd.DataFrame(vader_scores, columns=['compound', 'pos', 'neu', 'neg'])
    df_with_vader = pd.concat([df.reset_index(drop=True), vader_df], axis=1)
    
    print("\nVADER Score Statistics:")
    print(vader_df.describe())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for sentiment in df['sentiment'].unique():
        subset = df_with_vader[df_with_vader['sentiment'] == sentiment]['compound']
        axes[0, 0].hist(subset, bins=30, alpha=0.5, label=sentiment, edgecolor='black')
    axes[0, 0].set_xlabel('VADER Compound Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('VADER Compound Score by Sentiment', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    
    sentiment_order = ['negative', 'neutral', 'positive']
    df_ordered = df_with_vader.copy()
    df_ordered['sentiment'] = pd.Categorical(df_ordered['sentiment'], categories=sentiment_order, ordered=True)
    sns.boxplot(data=df_ordered, x='sentiment', y='compound', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Sentiment', fontsize=12)
    axes[0, 1].set_ylabel('VADER Compound Score', fontsize=12)
    axes[0, 1].set_title('VADER Score Distribution by Class', fontsize=14, fontweight='bold')
    
    sns.violinplot(data=df_ordered, x='sentiment', y='compound', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Sentiment', fontsize=12)
    axes[1, 0].set_ylabel('VADER Compound Score', fontsize=12)
    axes[1, 0].set_title('VADER Score Violin Plot', fontsize=14, fontweight='bold')
    
    for idx, sentiment in enumerate(sentiment_order):
        subset = df_with_vader[df_with_vader['sentiment'] == sentiment]
        axes[1, 1].scatter(subset['pos'], subset['neg'], alpha=0.5, label=sentiment, s=20)
    axes[1, 1].set_xlabel('Positive Score', fontsize=12)
    axes[1, 1].set_ylabel('Negative Score', fontsize=12)
    axes[1, 1].set_title('VADER Positive vs Negative Scores', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vader_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/vader_analysis.png")
    plt.close()
    
    return df_with_vader


def sample_texts_display(df, n_samples=3):
    print("\n" + "="*80)
    print("SAMPLE TEXTS")
    print("="*80)
    
    for sentiment in df['sentiment'].unique():
        print(f"\n{sentiment.upper()} Samples:")
        print("-" * 80)
        samples = df[df['sentiment'] == sentiment].sample(n=n_samples, random_state=42)
        for idx, row in samples.iterrows():
            print(f"\nText: {row['text']}")
            print(f"Label: {row['label']}")


def correlation_analysis(df_with_vader, output_dir='results/eda'):
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    corr_data = df_with_vader[['label', 'text_length', 'char_length', 'compound', 'pos', 'neu', 'neg']].copy()
    
    correlation_matrix = corr_data.corr()
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir}/correlation_matrix.png")
    plt.close()


def main():
    output_dir = 'results/eda'
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data('data/raw/sentiment_dataset.csv')
    
    basic_statistics(df)
    
    df = text_length_analysis(df, output_dir)
    
    class_distribution_analysis(df, output_dir)
    
    word_frequency_analysis(df, output_dir, top_n=20)
    
    generate_wordclouds(df, output_dir)
    
    df_with_vader = vader_sentiment_analysis(df, output_dir)
    
    correlation_analysis(df_with_vader, output_dir)
    
    sample_texts_display(df, n_samples=3)
    
    print("\n" + "="*80)
    print("EDA COMPLETED!")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  1. text_length_analysis.png")
    print("  2. class_distribution.png")
    print("  3. word_frequency.png")
    print("  4. wordclouds.png")
    print("  5. vader_analysis.png")
    print("  6. correlation_matrix.png")


if __name__ == "__main__":
    main()