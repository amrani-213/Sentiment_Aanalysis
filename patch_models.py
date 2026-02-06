"""
Quick Patch: Add **kwargs to FastText and Transformer models
This allows them to accept but ignore sentiment_scores parameter

Run this once to patch your model files
"""

import os
from pathlib import Path


def patch_file(filepath, old_signature, new_signature, model_name):
    """Patch a single file"""
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if new_signature in content:
        print(f"✅ {model_name} already patched, skipping")
        return True
    
    # Check if old signature exists
    if old_signature not in content:
        print(f"⚠️  {model_name}: Old signature not found, may already be modified")
        return False
    
    # Create backup
    backup_path = filepath + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  💾 Created backup: {backup_path}")
    
    # Apply patch
    content = content.replace(old_signature, new_signature)
    
    # Write patched file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ Patched {model_name}")
    return True


def main():
    print("="*80)
    print("PATCHING MODEL FILES FOR EVALUATOR COMPATIBILITY")
    print("="*80)
    print()
    
    # Patch FastText
    fasttext_path = 'src/models/baseline/fasttext.py'
    fasttext_old = "def forward(self, x, ngram_indices=None, mask=None):"
    fasttext_new = "def forward(self, x, ngram_indices=None, mask=None, **kwargs):"
    
    print("1. Patching FastText...")
    patch_file(fasttext_path, fasttext_old, fasttext_new, "FastText")
    
    # Patch Transformer
    transformer_path = 'src/models/baseline/custom_transformer.py'
    transformer_old = "def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:"
    transformer_new = "def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:"
    
    print("\n2. Patching Transformer...")
    patch_file(transformer_path, transformer_old, transformer_new, "Transformer")
    
    print("\n" + "="*80)
    print("✅ PATCHING COMPLETE!")
    print("="*80)
    print()
    print("Changes made:")
    print("  - FastText.forward() now accepts **kwargs")
    print("  - CustomTransformer.forward() now accepts **kwargs")
    print()
    print("These models will now ignore sentiment_scores when passed by the evaluator.")
    print()
    print("You can now run:")
    print("  python -m scripts.05_error_analysis --model_path results/baseline/transformer/best_model.pt --vocab_path results/baseline/vocabulary.pkl --output_dir results/error_analysis/transformer")


if __name__ == "__main__":
    main()