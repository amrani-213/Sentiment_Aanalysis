"""
Sentiment Analysis App - Custom Transformer vs BERTweet
Clean UI with single text prediction, model comparison, and CSV batch processing
"""

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import sys
import os

# Add project root and src to path
# This handles both running from root and from app directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'app' in current_dir else current_dir
src_path = os.path.join(project_root, 'src')

# Add both paths to ensure imports work
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Try different import methods to be robust
try:
    from src.models.baseline.custom_transformer import CustomTransformer
    from src.models.pretrained.bertweet import BERTweetClassifier, get_bertweet_tokenizer
    from src.data.preprocessing import EnhancedTextPreprocessor
except ModuleNotFoundError:
    # Fallback: try direct imports if src is in path
    from models.baseline.custom_transformer import CustomTransformer
    from models.pretrained.bertweet import BERTweetClassifier, get_bertweet_tokenizer
    from data.preprocessing import EnhancedTextPreprocessor


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """App configuration"""
    # Paths (adjust these to match your actual paths)
    TRANSFORMER_MODEL_PATH = r"C:\Sentiment_analysis\results\baseline\transformer\best_model.pt"
    BERTWEET_MODEL_PATH = r"C:\Sentiment_analysis\results\pretrained\bertweet\best_model.pt"
    VOCABULARY_PATH = r"C:\Sentiment_analysis\results\baseline\vocabulary.pkl"
    
    # Model parameters
    TRANSFORMER_CONFIG = {
        'vocab_size': 10000,  # Must match training vocab_size
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 4,
        'd_ff': 1024,
        'num_classes': 3,
        'max_len': 100,  # Must match training max_len (not 128!)
        'dropout': 0.1,
        'padding_idx': 0
    }
    
    BERTWEET_CONFIG = {
        'model_name': 'vinai/bertweet-base',
        'num_classes': 3,
        'dropout': 0.5
    }
    
    # Labels
    LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    LABEL_COLORS = {
        'Negative': '#FF4B4B',
        'Neutral': '#FFA500', 
        'Positive': '#00D66C'
    }
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_custom_transformer(model_path: str) -> CustomTransformer:
    """Load Custom Transformer model"""
    try:
        # Use config directly (vocab_size already set to 10000)
        config = Config.TRANSFORMER_CONFIG.copy()
        
        # Create model
        model = CustomTransformer(**config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(Config.DEVICE)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"❌ Failed to load Custom Transformer: {e}")
        return None


@st.cache_resource
def load_bertweet_model(model_path: str) -> BERTweetClassifier:
    """Load BERTweet model"""
    try:
        # Create model
        model = BERTweetClassifier(**Config.BERTWEET_CONFIG)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(Config.DEVICE)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"❌ Failed to load BERTweet: {e}")
        return None


@st.cache_resource
def load_bertweet_tokenizer():
    """Load BERTweet tokenizer"""
    try:
        return get_bertweet_tokenizer()
    except Exception as e:
        st.error(f"❌ Failed to load BERTweet tokenizer: {e}")
        return None


@st.cache_resource
def load_preprocessor():
    """Load text preprocessor"""
    try:
        # Create preprocessor with correct parameters
        preprocessor = EnhancedTextPreprocessor(
            vocab_size=10000,
            max_length=100,
            min_freq=2,
            use_spell_check=False,
            use_lemmatization=False
        )
        
        # Load vocabulary
        preprocessor.load_vocabulary(Config.VOCABULARY_PATH)
        
        return preprocessor
    except Exception as e:
        st.error(f"❌ Failed to load preprocessor: {e}")
        return None


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def text_to_indices(text: str, preprocessor: EnhancedTextPreprocessor, max_len: int = 100) -> torch.Tensor:
    """Convert text to token indices for Custom Transformer"""
    # Tokenize (simple whitespace tokenization)
    tokens = text.lower().split()
    
    # Convert to indices using preprocessor's vocabulary
    indices = [preprocessor.word2idx.get(token, preprocessor.word2idx.get('<UNK>', 1)) for token in tokens]
    
    # Pad or truncate
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    return torch.tensor([indices], dtype=torch.long)


def predict_custom_transformer(
    text: str, 
    model: CustomTransformer, 
    preprocessor: EnhancedTextPreprocessor
) -> Tuple[str, Dict[str, float]]:
    """Predict sentiment using Custom Transformer"""
    try:
        # Preprocess text using the preprocessor's clean_text method
        processed_text = preprocessor.clean_text(text)
        
        # Convert to indices
        input_ids = text_to_indices(processed_text, preprocessor, max_len=100).to(Config.DEVICE)
        
        # Create attention mask
        mask = (input_ids != 0).float()
        
        # Predict (NO sentiment_scores - this was the bug!)
        with torch.no_grad():
            logits = model(input_ids, mask=mask)
            probs = F.softmax(logits, dim=1)[0]
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        pred_label = Config.LABEL_MAP[pred_idx]
        
        # Get probabilities for all classes
        confidences = {
            Config.LABEL_MAP[i]: float(probs[i]) 
            for i in range(len(Config.LABEL_MAP))
        }
        
        return pred_label, confidences
    
    except Exception as e:
        st.error(f"❌ Custom Transformer prediction failed: {e}")
        return "Error", {}


def predict_bertweet(
    text: str, 
    model: BERTweetClassifier, 
    tokenizer,
    preprocessor: EnhancedTextPreprocessor
) -> Tuple[str, Dict[str, float]]:
    """Predict sentiment using BERTweet"""
    try:
        # Preprocess text (lighter preprocessing for BERT models)
        processed_text = preprocessor.clean_text(text)
        
        # Tokenize
        encoded = tokenizer(
            processed_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(Config.DEVICE)
        attention_mask = encoded['attention_mask'].to(Config.DEVICE)
        
        # Predict
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=1)[0]
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        pred_label = Config.LABEL_MAP[pred_idx]
        
        # Get probabilities for all classes
        confidences = {
            Config.LABEL_MAP[i]: float(probs[i]) 
            for i in range(len(Config.LABEL_MAP))
        }
        
        return pred_label, confidences
    
    except Exception as e:
        st.error(f"❌ BERTweet prediction failed: {e}")
        return "Error", {}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_confidence_chart(confidences: Dict[str, float], model_name: str) -> go.Figure:
    """Create a beautiful confidence bar chart"""
    labels = list(confidences.keys())
    values = [confidences[label] * 100 for label in labels]
    colors = [Config.LABEL_COLORS[label] for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f'{model_name} Confidence Scores',
            font=dict(size=16, family='Arial, sans-serif')
        ),
        xaxis=dict(
            title='Confidence (%)',
            range=[0, 100],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='',
            categoryorder='array',
            categoryarray=['Positive', 'Neutral', 'Negative']
        ),
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif')
    )
    
    return fig


def create_comparison_chart(transformer_conf: Dict, bertweet_conf: Dict) -> go.Figure:
    """Create side-by-side comparison chart"""
    labels = list(Config.LABEL_MAP.values())
    
    transformer_values = [transformer_conf[label] * 100 for label in labels]
    bertweet_values = [bertweet_conf[label] * 100 for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            name='Custom Transformer',
            x=labels,
            y=transformer_values,
            marker_color='#636EFA',
            text=[f'{v:.1f}%' for v in transformer_values],
            textposition='auto',
        ),
        go.Bar(
            name='BERTweet',
            x=labels,
            y=bertweet_values,
            marker_color='#EF553B',
            text=[f'{v:.1f}%' for v in bertweet_values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Sentiment',
        yaxis_title='Confidence (%)',
        barmode='group',
        height=350,
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif')
    )
    
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon="💭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            height: 3em;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .negative-box {
            background-color: #ffe6e6;
            border: 2px solid #FF4B4B;
            color: #c41e3a;
        }
        .neutral-box {
            background-color: #fff5e6;
            border: 2px solid #FFA500;
            color: #d97706;
        }
        .positive-box {
            background-color: #e6f7ed;
            border: 2px solid #00D66C;
            color: #059669;
        }
        .stTextArea textarea {
            border-radius: 8px;
        }
        h1 {
            color: #1e3a8a;
            font-weight: 700;
        }
        h2, h3 {
            color: #1e40af;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("💭 Sentiment Analysis App")
    st.markdown("### Compare Custom Transformer vs BERTweet Models")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        st.markdown("**Custom Transformer**")
        st.info("""
        - Architecture: From-scratch Transformer
        - Layers: 4 encoder layers
        - Attention Heads: 4
        - Parameters: ~2M
        """)
        
        st.markdown("**BERTweet**")
        st.info("""
        - Architecture: Twitter-specific BERT
        - Pretrained: vinai/bertweet-base
        - Parameters: ~135M
        - Fine-tuned for sentiment
        """)
        
        st.markdown("---")
        st.markdown("**Labels**")
        st.markdown("🔴 **Negative** - Negative sentiment")
        st.markdown("🟠 **Neutral** - Neutral sentiment")
        st.markdown("🟢 **Positive** - Positive sentiment")
    
    # Load models
    with st.spinner("🔄 Loading models..."):
        # Load preprocessor (includes vocabulary)
        preprocessor = load_preprocessor()
        
        # Load models
        transformer_model = load_custom_transformer(Config.TRANSFORMER_MODEL_PATH)
        bertweet_model = load_bertweet_model(Config.BERTWEET_MODEL_PATH)
        bertweet_tokenizer = load_bertweet_tokenizer()
    
    # Check if models loaded successfully
    models_loaded = all([
        preprocessor is not None,
        transformer_model is not None,
        bertweet_model is not None,
        bertweet_tokenizer is not None
    ])
    
    if not models_loaded:
        st.error("❌ Failed to load one or more models. Please check the paths and try again.")
        st.stop()
    
    st.success(f"✅ Models loaded successfully! Running on: **{Config.DEVICE}**")
    
    # Tabs
    tab1, tab2 = st.tabs(["📝 Single Text Prediction", "📁 Batch CSV Processing"])
    
    # ========================================================================
    # TAB 1: SINGLE TEXT PREDICTION
    # ========================================================================
    with tab1:
        st.markdown("### Enter text to analyze sentiment")
        
        # Text input
        user_text = st.text_area(
            "Text Input",
            placeholder="Type or paste your text here... (e.g., 'This movie was absolutely amazing!')",
            height=120,
            label_visibility="collapsed"
        )
        
        # Example texts
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("😊 Example: Positive"):
                user_text = "This is absolutely amazing! I love it so much! 🎉"
                st.rerun()
        with col2:
            if st.button("😐 Example: Neutral"):
                user_text = "It was okay. Nothing special, but not bad either."
                st.rerun()
        with col3:
            if st.button("😞 Example: Negative"):
                user_text = "This is terrible. Worst experience ever. Very disappointed."
                st.rerun()
        
        # Predict button
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if not user_text.strip():
                st.warning("⚠️ Please enter some text to analyze!")
            else:
                with st.spinner("🤖 Analyzing sentiment..."):
                    # Get predictions from both models
                    transformer_pred, transformer_conf = predict_custom_transformer(
                        user_text, transformer_model, preprocessor
                    )
                    bertweet_pred, bertweet_conf = predict_bertweet(
                        user_text, bertweet_model, bertweet_tokenizer, preprocessor
                    )
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                # Side-by-side predictions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Custom Transformer")
                    sentiment_class = transformer_pred.lower()
                    st.markdown(
                        f'<div class="prediction-box {sentiment_class}-box">'
                        f'Sentiment: {transformer_pred}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    fig1 = create_confidence_chart(transformer_conf, "Custom Transformer")
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.markdown("#### BERTweet")
                    sentiment_class = bertweet_pred.lower()
                    st.markdown(
                        f'<div class="prediction-box {sentiment_class}-box">'
                        f'Sentiment: {bertweet_pred}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    fig2 = create_confidence_chart(bertweet_conf, "BERTweet")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Comparison chart
                st.markdown("---")
                st.markdown("### 📊 Model Comparison")
                fig_comparison = create_comparison_chart(transformer_conf, bertweet_conf)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Agreement/Disagreement indicator
                if transformer_pred == bertweet_pred:
                    st.success(f"✅ **Both models agree:** {transformer_pred}")
                else:
                    st.warning(f"⚠️ **Models disagree:** Transformer={transformer_pred}, BERTweet={bertweet_pred}")
    
    # ========================================================================
    # TAB 2: BATCH CSV PROCESSING
    # ========================================================================
    with tab2:
        st.markdown("### Upload CSV file for batch prediction")
        st.markdown("**Required:** Your CSV must have a column named `text`")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with a 'text' column"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Check for 'text' column
                if 'text' not in df.columns:
                    st.error("❌ CSV must contain a 'text' column!")
                    st.stop()
                
                st.success(f"✅ Loaded {len(df)} texts from CSV")
                
                # Show preview
                with st.expander("📋 Preview Data", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Process button
                if st.button("🚀 Process All Texts", type="primary"):
                    with st.spinner(f"🤖 Processing {len(df)} texts..."):
                        # Initialize result lists
                        transformer_predictions = []
                        transformer_confidences = []
                        bertweet_predictions = []
                        bertweet_confidences = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        
                        # Process each text
                        for idx, text in enumerate(df['text']):
                            # Skip empty texts
                            if pd.isna(text) or str(text).strip() == '':
                                transformer_predictions.append('N/A')
                                transformer_confidences.append(0.0)
                                bertweet_predictions.append('N/A')
                                bertweet_confidences.append(0.0)
                            else:
                                # Transformer prediction
                                t_pred, t_conf = predict_custom_transformer(
                                    str(text), transformer_model, preprocessor
                                )
                                transformer_predictions.append(t_pred)
                                transformer_confidences.append(max(t_conf.values()))
                                
                                # BERTweet prediction
                                b_pred, b_conf = predict_bertweet(
                                    str(text), bertweet_model, bertweet_tokenizer, preprocessor
                                )
                                bertweet_predictions.append(b_pred)
                                bertweet_confidences.append(max(b_conf.values()))
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / len(df))
                        
                        progress_bar.empty()
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['Transformer_Prediction'] = transformer_predictions
                    results_df['Transformer_Confidence'] = [f"{c:.2%}" for c in transformer_confidences]
                    results_df['BERTweet_Prediction'] = bertweet_predictions
                    results_df['BERTweet_Confidence'] = [f"{c:.2%}" for c in bertweet_confidences]
                    results_df['Agreement'] = [
                        '✅' if t == b else '❌' 
                        for t, b in zip(transformer_predictions, bertweet_predictions)
                    ]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 🎯 Batch Prediction Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Texts", len(results_df))
                    
                    with col2:
                        agreement_rate = (results_df['Agreement'] == '✅').sum() / len(results_df) * 100
                        st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
                    
                    with col3:
                        avg_trans_conf = np.mean(transformer_confidences) * 100
                        st.metric("Avg Transformer Conf.", f"{avg_trans_conf:.1f}%")
                    
                    with col4:
                        avg_bert_conf = np.mean(bertweet_confidences) * 100
                        st.metric("Avg BERTweet Conf.", f"{avg_bert_conf:.1f}%")
                    
                    # Results table
                    st.markdown("#### 📊 Detailed Results")
                    
                    # Color-code predictions
                    def highlight_sentiment(row):
                        colors = []
                        for col in row.index:
                            if 'Prediction' in col:
                                if 'Positive' in str(row[col]):
                                    colors.append('background-color: #d1fae5')
                                elif 'Negative' in str(row[col]):
                                    colors.append('background-color: #fee2e2')
                                elif 'Neutral' in str(row[col]):
                                    colors.append('background-color: #fef3c7')
                                else:
                                    colors.append('')
                            else:
                                colors.append('')
                        return colors
                    
                    styled_df = results_df.style.apply(highlight_sentiment, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Distribution charts
                    st.markdown("---")
                    st.markdown("### 📈 Distribution Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Transformer distribution
                        trans_counts = results_df['Transformer_Prediction'].value_counts()
                        fig_trans_dist = go.Figure(data=[
                            go.Pie(
                                labels=trans_counts.index,
                                values=trans_counts.values,
                                marker=dict(colors=[
                                    Config.LABEL_COLORS.get(label, '#gray') 
                                    for label in trans_counts.index
                                ]),
                                hole=0.4
                            )
                        ])
                        fig_trans_dist.update_layout(
                            title="Custom Transformer Distribution",
                            height=300
                        )
                        st.plotly_chart(fig_trans_dist, use_container_width=True)
                    
                    with col2:
                        # BERTweet distribution
                        bert_counts = results_df['BERTweet_Prediction'].value_counts()
                        fig_bert_dist = go.Figure(data=[
                            go.Pie(
                                labels=bert_counts.index,
                                values=bert_counts.values,
                                marker=dict(colors=[
                                    Config.LABEL_COLORS.get(label, '#gray') 
                                    for label in bert_counts.index
                                ]),
                                hole=0.4
                            )
                        ])
                        fig_bert_dist.update_layout(
                            title="BERTweet Distribution",
                            height=300
                        )
                        st.plotly_chart(fig_bert_dist, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error processing CSV: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit | Custom Transformer vs BERTweet Comparison"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()