# Sentiment Analysis with Deep Learning

**Author:** Amrani Bouabdellah  
**Institution:** Higher National School of Statistics and Applied Economy (ENSSEA)  
**Course:** Introduction to Deep Learning - 5th Year Statistics and Data Science  
**Date:** February 2026

---

## Project Overview

This project implements a comprehensive sentiment analysis system using deep learning architectures to classify text samples into three sentiment categories: negative, neutral, and positive. The system analyzes 31,475 text samples from various online platforms including social media posts, product reviews, and customer feedback.

The implementation addresses the challenges of natural language processing including handling informal language, emojis, abbreviations, sarcasm, and mixed sentiments through multiple neural network architectures and advanced preprocessing techniques.

---

## Dataset

**Source:** [Hugging Face - Multiclass Sentiment Analysis Dataset](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction)

**Size:** 31,475 samples

**Structure:**
- `id`: Unique identifier
- `text`: Raw text content (variable length)
- `label`: Numerical sentiment (0=negative, 1=neutral, 2=positive)
- `sentiment`: Text representation of label

**Class Distribution:**
- Neutral: 37.3% (11,649 samples)
- Positive: 33.5% (10,478 samples)
- Negative: 29.2% (9,105 samples)

---

## Project Structure

```
Sentiment_analysis/
├── data/
│   └── raw/
│       └── sentiment_dataset.csv
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # Text preprocessing pipeline
│   │   ├── augmentation.py       # Data augmentation
│   │   └── dataset.py            # PyTorch dataset classes
│   │
│   ├── models/
│   │   ├── baseline/
│   │   │   ├── fasttext.py       # FastText model
│   │   │   ├── bilstm_attention.py  # BiLSTM with multi-head attention
│   │   │   └── custom_transformer.py  # Transformer from scratch
│   │   │
│   │   ├── pretrained/
│   │   │   ├── roberta.py        # RoBERTa classifier
│   │   │   └── bertweet.py       # BERTweet classifier
│   │   │
│   │   └── ensemble/
│   │       ├── voting_ensemble.py    # Voting strategies
│   │       └── stacking_ensemble.py  # Meta-learner ensemble
│   │
│   ├── training/
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Custom loss functions
│   │   └── metrics.py            # Evaluation metrics
│   │
│   ├── evaluation/
│   │   ├── evaluator.py          # Model evaluation
│   │   ├── error_analysis.py     # Error pattern analysis
│   │   └── visualizer.py         # Result visualization
│   │
│   └── utils/
│       ├── config.py             # Configuration utilities
│       ├── logger.py             # Logging setup
│       ├── helpers.py            # Helper functions
│       └── model_loader.py       # Model loading utilities
│
├── scripts/
│   ├── 00_eda.py                 # Exploratory data analysis
│   ├── 01_train_baseline.py      # Train baseline models
│   ├── 02_train_pretrained.py    # Train pretrained models
│   ├── 03_train_ensemble.py      # Train ensemble models
│   ├── 04_evaluate_all.py        # Comprehensive evaluation
│   └── 05_error_analysis.py      # Error analysis
│
├── results/
│   ├── eda/                      # EDA visualizations
│   ├── baseline/                 # Baseline model results
│   ├── pretrained/               # Pretrained model results
│   ├── ensemble/                 # Ensemble results
│   └── final_comparison/         # Comparative analysis
│
├── notebooks/
│   ├── EDA_Analysis.ipynb        # Interactive EDA
│   ├── Model_baseline_Comparison.ipynb   # Baseline model comparison
│   └── Model_ensemble_Comparison.ipynb   # Ensemble model comparison
│
├── debugging/                    # Debugging scripts and experiments
│   ├── FIX_ensemble_results.py
│   ├── patch_models.py
│   ├── results_save.py
│   ├── save_processed_data.py
│   └── placeholder.py            # Optional placeholder
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── baseline_training_20260206_122307.log
├── pretrained_training_20260206_123139.log
├── .gitattributes
└── .gitignore

```

---

## Methodology

### 1. Exploratory Data Analysis

**Key Findings:**
- Average text length: 18 words
- Character count range: 5-2,000+ characters
- Moderate class imbalance favoring neutral sentiment
- VADER sentiment scores show strong correlation with labels (ρ = 0.60)
- Word clouds reveal distinct vocabulary patterns per class

**Visualizations:**
- Text length distributions
- Class balance analysis
- Word frequency analysis
- VADER sentiment score distributions
- Feature correlation matrices

### 2. Text Preprocessing Pipeline

**Components:**

1. **Text Cleaning:**
   - URL removal/normalization
   - HTML entity decoding
   - Special character handling
   - Whitespace normalization

2. **Tokenization:**
   - Word-level tokenization
   - Vocabulary building (10,000 tokens)
   - Padding/truncation to 100 tokens
   - Special token handling

3. **Feature Engineering:**
   - VADER sentiment scores (compound, positive, negative, neutral)
   - Text length features
   - Character n-grams (for FastText)

4. **Data Augmentation:**
   - Synonym replacement (WordNet)
   - Random word swapping
   - Random deletion
   - Class balancing through targeted augmentation

### 3. Model Architectures

#### Baseline Models (from scratch)

**A. FastText**
- Architecture: Averaged word embeddings + character n-grams
- Embedding dimension: 100
- Parameters: ~1M
- Training: 20 epochs, Adam optimizer (lr=0.001)

**B. BiLSTM with Multi-Head Attention**
- Architecture: Bidirectional LSTM + 4-head self-attention
- Hidden dimension: 256
- LSTM layers: 2
- Attention heads: 4
- Parameters: ~2.5M
- Features: Sentiment score integration, layer normalization
- Training: 20 epochs, Adam optimizer (lr=0.003)

**C. Custom Transformer**
- Architecture: Transformer encoder from scratch
- Model dimension: 256
- Encoder layers: 4
- Attention heads: 4
- Feed-forward dimension: 1024
- Parameters: ~3M
- Components: Positional encoding, multi-head attention, residual connections
- Training: 25 epochs, Adam optimizer (lr=0.0001)

#### Pretrained Models

**D. RoBERTa-base**
- Base model: `roberta-base` (125M parameters)
- Classification head: 3-layer MLP with layer normalization
- Dropout: 0.5
- Training: 5 epochs, AdamW (lr=1e-5), linear warmup

**E. BERTweet-base**
- Base model: `vinai/bertweet-base` (135M parameters)
- Twitter-specific pretraining
- Classification head: 2-layer MLP
- Dropout: 0.5
- Training: 4 epochs, AdamW (lr=2e-5), linear warmup

#### Ensemble Methods

**F. Voting Ensemble**
- Strategy: Soft voting (probability averaging)
- Models: All baseline + pretrained models
- Weighting: Performance-based automatic optimization

**G. Weighted Voting Ensemble**
- Weights optimized on validation set
- Proportional to individual model accuracy

**H. Stacking Ensemble**
- Base models: All trained models
- Meta-learner: 2-layer MLP (64 hidden units)
- Meta-training: 20 epochs on validation set

### 4. Training Configuration

**Loss Functions:**
- Focal Loss (γ=2.0) for class imbalance handling
- Label Smoothing (ε=0.1) for calibration
- Cross-Entropy baseline

**Optimization:**
- Baseline models: Adam optimizer
- Pretrained models: AdamW with weight decay
- Learning rate schedules: Linear warmup + decay
- Early stopping: Patience of 5 epochs (baseline), 2 epochs (pretrained)

**Regularization:**
- Dropout (0.3-0.5)
- Layer normalization
- Weight decay (pretrained models)
- Data augmentation

**Data Split:**
- Training: 80% (with augmentation)
- Validation: 10%
- Test: 10%

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | MCC |
|-------|----------|-----------|--------|----------|-----|
| **BERTweet** | **77.56%** | **77.73%** | **77.92%** | **77.73%** | **0.663** |
| **RoBERTa** | **76.25%** | **76.27%** | **76.47%** | **76.27%** | **0.643** |
| Weighted Voting | 71.29% | 71.60% | 71.52% | 71.56% | 0.568 |
| Voting | 70.87% | 71.19% | 71.10% | 71.14% | 0.562 |
| Stacking | 69.14% | 69.30% | 69.40% | 69.35% | 0.536 |
| BiLSTM-Attention | 68.98% | 69.09% | 69.30% | 69.19% | 0.533 |
| Custom Transformer | 65.24% | 65.75% | 65.63% | 65.64% | 0.477 |
| FastText | 64.72% | 64.42% | 65.30% | 64.59% | 0.473 |

### Per-Class Performance (BERTweet - Best Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 76.25% | 73.21% | 74.70% | 911 |
| Neutral | 72.45% | 69.35% | 70.87% | 1,165 |
| Positive | 84.54% | 89.79% | 87.09% | 1,048 |

**Key Observations:**
- BERTweet achieves best overall performance due to Twitter-specific pretraining
- Positive sentiment shows highest classification accuracy
- Neutral class presents greatest challenge (48.9% error rate for stacking model)
- Pretrained transformers significantly outperform baseline models
- Ensemble methods show moderate improvement over individual baseline models

### Error Analysis

**Common Misclassification Patterns:**

1. **Neutral → Positive** (most frequent confusion)
   - Short affirmative phrases: "Amazing and helpful"
   - Professional language with subtle positivity

2. **Neutral → Negative**
   - Complaints framed as questions
   - Sarcasm and irony

3. **Challenges:**
   - Short texts (< 5 words): Lack of context
   - Mixed sentiment: "Good app but expensive"
   - Sarcasm: "Yeah, that's just great"
   - Domain-specific terminology

**High-Confidence Errors:**
- "Expensive" → Predicted: Negative (True: Positive)
- "Best app" → Predicted: Positive (True: Neutral)
- "-sigh" → Predicted: Negative (True: Neutral)

---

## Technical Implementation

### Requirements

```
Python >= 3.8
PyTorch >= 2.0.0
transformers >= 4.30.0
scikit-learn >= 1.3.0
pandas >= 2.0.0
numpy >= 1.24.0
nltk >= 3.8.1
vaderSentiment >= 3.3.2
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Sentiment_analysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### Usage

**1. Exploratory Data Analysis:**
```bash
python scripts/00_eda.py
```

**2. Train Baseline Models:**
```bash
python scripts/01_train_baseline.py \
  --data_path data/raw/sentiment_dataset.csv \
  --output_dir results/baseline \
  --augmentation \
  --device cuda
```

**3. Train Pretrained Models:**
```bash
python scripts/02_train_pretrained.py \
  --data_path data/raw/sentiment_dataset.csv \
  --output_dir results/pretrained \
  --device cuda
```

**4. Train Ensemble Models:**
```bash
python scripts/03_train_ensemble.py \
  --baseline_dir results/baseline \
  --pretrained_dir results/pretrained \
  --output_dir results/ensemble \
  --device cuda
```

**5. Evaluate All Models:**
```bash
python scripts/04_evaluate_all.py \
  --results_dir results \
  --output_dir results/final_comparison
```

**6. Error Analysis:**
```bash
python scripts/05_error_analysis.py \
  --model_path results/pretrained/bertweet/best_model.pt \
  --output_dir results/error_analysis \
  --device cuda
```

---

## Key Innovations

1. **Enhanced Preprocessing Pipeline:**
   - VADER sentiment features integration
   - Character n-gram support for FastText
   - Robust handling of social media text

2. **Custom Architectures:**
   - BiLSTM with multi-head self-attention mechanism
   - Transformer implementation from scratch with proper positional encoding
   - Advanced regularization techniques (focal loss, label smoothing)

3. **Comprehensive Ensemble Framework:**
   - Multiple voting strategies (hard, soft, weighted)
   - Meta-learning through stacking
   - Automatic weight optimization

4. **Robust Evaluation:**
   - Multiple metrics (Accuracy, Precision, Recall, F1, MCC, ROC-AUC)
   - Per-class performance analysis
   - Confidence-based error analysis
   - Confusion pattern identification

---

## Deployment Considerations

### Production Recommendations

1. **Model Selection:**
   - **Real-time systems:** FastText or BiLSTM (low latency, < 50ms)
   - **Batch processing:** BERTweet (highest accuracy)
   - **Resource-constrained:** FastText (smallest footprint)

2. **Infrastructure:**
   - GPU: NVIDIA T4 or better for transformer models
   - CPU: Sufficient for baseline models
   - Memory: 4GB minimum, 16GB recommended for transformers

3. **Optimization Strategies:**
   - Model quantization (INT8) for 4x speedup
   - ONNX export for cross-platform deployment
   - Batch inference for throughput optimization
   - Model distillation for compact deployment

4. **Monitoring:**
   - Prediction confidence tracking
   - Class distribution drift detection
   - Latency and throughput metrics
   - A/B testing framework

5. **Data Pipeline:**
   - Real-time preprocessing queue
   - Caching for frequent queries
   - Asynchronous processing for high throughput

### Practical Applications

- **Customer Experience Management:** Automated review analysis
- **Brand Monitoring:** Social media sentiment tracking
- **Market Research:** Consumer attitude analysis
- **Content Moderation:** Negative content detection

---

## Limitations and Future Work

### Current Limitations

1. **Context Understanding:**
   - Struggles with sarcasm and irony
   - Limited understanding of domain-specific language
   - Difficulty with very short texts

2. **Class Imbalance:**
   - Neutral class remains challenging
   - Mixed sentiments not explicitly handled

3. **Computational Requirements:**
   - Transformer models require significant GPU memory
   - Training time: 30-50 minutes per epoch for BERTweet

### Future Improvements

1. **Model Enhancements:**
   - Multi-task learning with emotion detection
   - Attention visualization for interpretability
   - Cross-lingual sentiment analysis

2. **Data Augmentation:**
   - Back-translation for diversity
   - Contextual word embeddings augmentation
   - Adversarial examples for robustness

3. **Advanced Architectures:**
   - GPT-based generative models
   - Multi-modal sentiment (text + images)
   - Hierarchical attention for long documents

---

## Reproducibility

All experiments are reproducible with:
- Fixed random seeds (seed=42)
- Deterministic algorithms enabled
- Complete hyperparameter documentation
- Version-controlled dependencies

Training logs available in `logs/` directory.

---

## References

1. Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
2. Liu et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach."
3. Nguyen et al. (2020). "BERTweet: A pre-trained language model for English Tweets."
4. Lin et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
5. Hutto & Gilbert (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis."

---

## License

This project is submitted as academic work for the Introduction to Deep Learning course at ENSSEA.

---

## Contact

**Amrani Bouabdellah**  
Higher National School of Statistics and Applied Economy (ENSSEA)  
Email: [Contact via ENSSEA]

---

**Submission Date:** February 7, 2026  
**Course:** Introduction to Deep Learning  
**Instructor:** Ayoub Asri
