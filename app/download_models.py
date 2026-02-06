# download_models.py
import os
import gdown
from pathlib import Path
import streamlit as st

# Google Drive file IDs
MODEL_FILES = {
    'transformer': {
        'url': 'https://drive.google.com/uc?id=124EHm4lHWWWzJfVdlJF-9l8QRqTjI0BK',
        'path': 'results/baseline/transformer/best_model.pt'
    },
    'bertweet': {
        'url': 'https://drive.google.com/uc?id=1DlGRe4qHypaWby6MU1ab0ZcpJp2FhQtL',
        'path': 'results/pretrained/bertweet/best_model.pt'
    },
    'vocabulary': {
        'url': 'https://drive.google.com/uc?id=1DkbnnYe1_dVFGuOwsDaZ_9UuH1zCXCwE',  # Fixed format
        'path': 'results/baseline/vocabulary.pkl'
    }
}

@st.cache_resource
def download_model_if_needed(model_name):
    """Download model or vocabulary file from Google Drive if not present."""
    model_info = MODEL_FILES.get(model_name)
    if not model_info:
        st.error(f"Unknown model: {model_name}")
        return None
    
    file_path = model_info['path']
    
    # Check if already downloaded
    if os.path.exists(file_path):
        st.info(f"✓ {model_name} already downloaded")
        return file_path
    
    # Create directory (using Path for cross-platform compatibility)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    st.info(f"⬇️ Downloading {model_name}... (this may take a minute)")
    try:
        gdown.download(model_info['url'], file_path, quiet=False, fuzzy=True)
        
        # Verify download
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            st.success(f"✅ Successfully downloaded {model_name}!")
            return file_path
        else:
            st.error(f"❌ Download failed - file is empty or missing")
            return None
            
    except Exception as e:
        st.error(f"❌ Failed to download {model_name}: {str(e)}")
        return None

def ensure_all_models_downloaded():
    """Ensure all models and the vocabulary file are downloaded."""
    success = True
    for model_name in MODEL_FILES.keys():
        result = download_model_if_needed(model_name)  # Only call once!
        if result is None:
            success = False
    
    return success