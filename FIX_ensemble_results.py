"""
Fix existing ensemble results by adding missing metrics
Run this in your Sentiment_analysis directory
"""

import pickle
import os
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np


def fix_ensemble_results():
    """Add missing metrics to existing ensemble results"""
    
    results_dir = Path('results/ensemble')
    
    if not results_dir.exists():
        print(f"❌ Directory {results_dir} does not exist!")
        return
    
    # Find all results.pkl files
    result_files = list(results_dir.glob('*/results.pkl'))
    
    if not result_files:
        print(f"❌ No results.pkl files found in {results_dir}")
        return
    
    print(f"Found {len(result_files)} results files")
    print("="*80)
    
    for result_file in result_files:
        print(f"\n📁 Processing: {result_file.parent.name}")
        
        try:
            # Load existing results
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            # Check if already has required metrics
            required_metrics = ['precision_macro', 'recall_macro', 'mcc']
            has_all = all(m in results for m in required_metrics)
            
            if has_all:
                print("  ✅ Already has all required metrics, skipping")
                continue
            
            # Extract predictions and labels
            predictions = results['predictions']
            labels = results['true_labels']
            
            # Calculate ALL metrics
            accuracy = results.get('accuracy', 0)
            
            # Macro metrics
            precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
            recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
            f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
            
            # Matthews Correlation Coefficient
            mcc = matthews_corrcoef(labels, predictions)
            
            # Per-class metrics (0=negative, 1=neutral, 2=positive)
            f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
            precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
            recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
            
            # Update results with ALL metrics
            results['accuracy'] = accuracy
            results['precision_macro'] = precision_macro
            results['recall_macro'] = recall_macro
            results['f1_macro'] = f1_macro
            results['mcc'] = mcc
            
            results['f1_negative'] = f1_per_class[0]
            results['f1_neutral'] = f1_per_class[1]
            results['f1_positive'] = f1_per_class[2]
            
            results['precision_negative'] = precision_per_class[0]
            results['precision_neutral'] = precision_per_class[1]
            results['precision_positive'] = precision_per_class[2]
            
            results['recall_negative'] = recall_per_class[0]
            results['recall_neutral'] = recall_per_class[1]
            results['recall_positive'] = recall_per_class[2]
            
            # Add ensemble_type if missing
            if 'ensemble_type' not in results:
                dir_name = result_file.parent.name.lower()
                if 'voting' in dir_name:
                    results['ensemble_type'] = 'voting'
                elif 'weighted' in dir_name:
                    results['ensemble_type'] = 'weighted'
                elif 'stacking' in dir_name:
                    results['ensemble_type'] = 'stacking'
                else:
                    results['ensemble_type'] = 'ensemble'
            
            # Save updated results (backup first)
            backup_file = result_file.with_suffix('.pkl.backup')
            if not backup_file.exists():
                import shutil
                shutil.copy2(result_file, backup_file)
                print(f"  💾 Created backup: {backup_file.name}")
            
            with open(result_file, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"  ✅ Updated with all metrics:")
            print(f"     Type: {results['ensemble_type']}")
            print(f"     Accuracy: {accuracy:.4f}")
            print(f"     Precision: {precision_macro:.4f}")
            print(f"     Recall: {recall_macro:.4f}")
            print(f"     F1: {f1_macro:.4f}")
            print(f"     MCC: {mcc:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error processing {result_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ All results files updated!")
    print("\nYou can now run:")
    print("  python -m scripts.04_evaluate_all")


if __name__ == "__main__":
    print("="*80)
    print("FIXING ENSEMBLE RESULTS FILES")
    print("="*80)
    print()
    
    fix_ensemble_results()