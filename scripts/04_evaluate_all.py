import os
import pickle
import argparse
from pathlib import Path
import sys

# Fix path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.visualizer import SentimentVisualizer as Visualizer


def evaluate_all_models(args):
    print("="*80)
    print("EVALUATING ALL MODELS")
    print("="*80)
    
    results_base_dir = Path(args.results_dir)
    
    all_results = {}
    
    print("\nSearching for model results...")
    for result_dir in results_base_dir.rglob('results.pkl'):
        model_name = result_dir.parent.name
        
        print(f"Found: {model_name}")
        
        with open(result_dir, 'rb') as f:
            results = pickle.load(f)
        
        all_results[model_name] = results
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(all_results)} model results")
    
    output_dir = args.output_dir if args.output_dir else 'results/final_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    viz = Visualizer(output_dir)
    
    # Prepare models_metrics dict for comparison plots
    models_metrics = {}
    for model_name, results in all_results.items():
        if 'metrics' in results:
            models_metrics[model_name] = results['metrics']
        else:
            models_metrics[model_name] = results
    
    print("1. Model comparison radar...")
    viz.plot_model_comparison_radar(models_metrics, save_name='model_comparison_radar.png')
    
    print("2. Model comparison bars...")
    viz.plot_model_comparison_bars(models_metrics, save_name='model_comparison_bars.png')
    
    print("3. Individual confusion matrices...")
    for model_name, results in all_results.items():
        cm = results.get('confusion_matrix') or results.get('metrics', {}).get('confusion_matrix')
        if cm is not None:
            viz.plot_confusion_matrix(cm, normalize=False, save_name=f'{model_name}_confusion_matrix.png')
            viz.plot_confusion_matrix(cm, normalize=True, save_name=f'{model_name}_confusion_matrix_normalized.png')
    
    print("4. ROC curves (for each model)...")
    for model_name, results in all_results.items():
        labels = results.get('labels')
        probs = results.get('probabilities')
        if labels is not None and probs is not None:
            viz.plot_roc_curves(labels, probs, save_name=f'{model_name}_roc_curves.png')
    
    print("5. Summary dashboards (for each model)...")
    for model_name, results in all_results.items():
        metrics = results.get('metrics') or results
        cm = metrics.get('confusion_matrix')
        if cm is not None:
            viz.create_summary_dashboard(metrics, cm, save_name=f'{model_name}_summary_dashboard.png')
    
    # Create comparison table
    import pandas as pd
    comparison_data = []
    for model_name, results in all_results.items():
        metrics = results.get('metrics') or results
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision_macro', 0),
            'Recall': metrics.get('recall_macro', 0),
            'F1-Score': metrics.get('f1_macro', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1-Score', ascending=False)
    
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save markdown table
    md_path = os.path.join(output_dir, 'comparison_table.md')
    with open(md_path, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write(df.to_markdown(index=False))
    
    # Save latex table
    latex_path = os.path.join(output_dir, 'comparison_table.tex')
    latex_table = df.to_latex(index=False, float_format="%.4f")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    print(f"\nAll visualizations and tables saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - model_comparison_radar.png")
    print("  - model_comparison_bars.png")
    print("  - [model]_confusion_matrix.png (for each model)")
    print("  - [model]_confusion_matrix_normalized.png (for each model)")
    print("  - [model]_roc_curves.png (for each model)")
    print("  - [model]_summary_dashboard.png (for each model)")
    print("  - comparison_table.csv")
    print("  - comparison_table.md")
    print("  - comparison_table.tex")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate all models')
    
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Base directory containing model results')
    parser.add_argument('--output_dir', type=str, default='results/final_comparison',
                       help='Output directory for comparison plots')
    
    args = parser.parse_args()
    
    evaluate_all_models(args)