"""
Visualization Module for Sentiment Analysis
20+ publication-ready plots for model evaluation and comparison
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
from wordcloud import WordCloud
from math import pi
from scipy.sparse import spmatrix  # ✅ Import sparse matrix base type


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class SentimentVisualizer:
    """
    Comprehensive visualizer for sentiment analysis models
    
    Creates 20+ different plot types for analysis and presentation
    """
    
    def __init__(self, save_dir='results/visualizations', dpi=150):
        """
        Args:
            save_dir: Directory to save plots
            dpi: Resolution for saved figures
        """
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.colors = ['#e74c3c', '#95a5a6', '#2ecc71']  # Red, Gray, Green
    
    # =========================================================================
    # 1-2. TRAINING CURVES
    # =========================================================================
    
    def plot_training_curves(self, history: Dict[str, List[float]], save_name='training_curves.png'):
        """
        Plot training and validation loss + accuracy
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            save_name: Filename to save
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved training curves to {save_name}")
    
    # =========================================================================
    # 3-4. CONFUSION MATRICES
    # =========================================================================
    
    def plot_confusion_matrix(self, cm: np.ndarray, normalize: bool = False, save_name='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix (numpy array)
            normalize: Whether to normalize
            save_name: Filename
        """
        plt.figure(figsize=(10, 8))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13)
        plt.xlabel('Predicted Label', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved confusion matrix to {save_name}")
    
    # =========================================================================
    # 5. PER-CLASS F1 SCORES
    # =========================================================================
    
    def plot_per_class_f1(self, metrics: Dict[str, Any], save_name='per_class_f1.png'):
        """
        Bar chart of per-class F1 scores
        
        Args:
            metrics: Dictionary with per_class metrics
            save_name: Filename
        """
        plt.figure(figsize=(10, 6))
        
        classes = list(metrics['per_class'].keys())
        f1_scores = [metrics['per_class'][c]['f1'] for c in classes]
        
        bars = plt.bar(classes, f1_scores, color=self.colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xlabel('Class', fontsize=13)
        plt.ylabel('F1-Score', fontsize=13)
        plt.title('Per-Class F1 Scores', fontsize=16, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved per-class F1 to {save_name}")
    
    # =========================================================================
    # 6. MODEL COMPARISON RADAR CHART (FIXED)
    # =========================================================================
    
    def plot_model_comparison_radar(self, models_metrics: Dict[str, Dict[str, float]], 
                                    save_name='model_comparison_radar.png'):
        """
        Radar chart comparing multiple models
        
        Args:
            models_metrics: Dict mapping model names to their metrics
            save_name: Filename
        """
        # ✅ FIX 1: Use explicit polar projection creation to satisfy type checker
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')  # Type checker knows this creates PolarAxes
        
        # Metrics to compare
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
        num_vars = len(categories)
        
        # Compute angles
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the loop
        
        # ✅ FIX 2: Suppress type checker warnings for polar-specific methods
        # These methods exist at runtime but aren't in matplotlib's type stubs
        ax.set_theta_offset(pi / 2)  # type: ignore[attr-defined]
        ax.set_theta_direction(-1)   # type: ignore[attr-defined]
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        
        # Plot for each model
        for model_name, metrics in models_metrics.items():
            values = [
                metrics['accuracy'],
                metrics['precision_macro'],
                metrics['recall_macro'],
                metrics['f1_macro'],
                (metrics['mcc'] + 1) / 2  # Normalize MCC from [-1,1] to [0,1]
            ]
            values += values[:1]  # Complete the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_ylim(0, 1)
        ax.set_title('Model Comparison - Multiple Metrics', 
                    fontsize=16, fontweight='bold', pad=20, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved radar chart to {save_name}")
    
    # =========================================================================
    # 7. ERROR DISTRIBUTION (FIXED)
    # =========================================================================
    
    def plot_error_distribution(self, error_analysis: Dict[str, Any], 
                                save_name='error_distribution.png'):
        """
        Plot error distribution by class
        
        Args:
            error_analysis: Error analysis dictionary
            save_name: Filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ✅ FIX 3: Use correct key name ('errors_by_class' not 'errors_by_true_class')
        # Based on ErrorAnalyzer fix from earlier
        classes = list(error_analysis['errors_by_class'].keys())
        errors = [error_analysis['errors_by_class'][c]['errors'] for c in classes]
        totals = [error_analysis['errors_by_class'][c]['total'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax1.bar(x - width/2, errors, width, label='Errors', color='#e74c3c', alpha=0.8)
        ax1.bar(x + width/2, totals, width, label='Total', color='#3498db', alpha=0.8)
        
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Errors vs Total Samples by Class', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes)
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Error rates
        error_rates = [error_analysis['errors_by_class'][c]['error_rate'] for c in classes]  # ✅ Corrected key
        bars = ax2.bar(classes, error_rates, color=self.colors, alpha=0.8, edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Error Rate', fontsize=12)
        ax2.set_title('Error Rate by Class', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, max(error_rates) * 1.2 if error_rates else 1.0)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved error distribution to {save_name}")
    
    # =========================================================================
    # 8. CONFIDENCE DISTRIBUTION
    # =========================================================================
    
    def plot_confidence_distribution(self, probabilities: np.ndarray, predictions: np.ndarray, 
                                     labels: np.ndarray, save_name='confidence_distribution.png'):
        """
        Plot prediction confidence distribution
        
        Args:
            probabilities: Prediction probabilities
            predictions: Predicted labels
            labels: True labels
            save_name: Filename
        """
        confidences = np.max(probabilities, axis=1)
        correct = predictions == labels
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall distribution
        ax1.hist(confidences, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(confidences.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {confidences.mean():.3f}')
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Correct vs Incorrect
        ax2.hist([confidences[correct], confidences[~correct]], bins=50, 
                label=['Correct', 'Incorrect'], 
                color=['#2ecc71', '#e74c3c'],
                alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Confidence: Correct vs Incorrect Predictions', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved confidence distribution to {save_name}")
    
    # =========================================================================
    # 9. TEXT LENGTH VS ACCURACY
    # =========================================================================
    
    def plot_length_vs_accuracy(self, texts: List[str], predictions: np.ndarray, 
                                labels: np.ndarray, save_name='length_vs_accuracy.png'):
        """
        Plot accuracy vs text length
        
        Args:
            texts: List of texts
            predictions: Predicted labels
            labels: True labels
            save_name: Filename
        """
        lengths = np.array([len(text.split()) for text in texts])
        correct = predictions == labels
        
        # Create bins
        bins = [0, 10, 20, 30, 50, 100, np.inf]
        bin_labels = ['<10', '10-20', '20-30', '30-50', '50-100', '100+']
        
        bin_accuracies = []
        bin_counts = []
        
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (lengths >= low) & (lengths < high)
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy by length
        bars = ax1.bar(bin_labels, bin_accuracies, alpha=0.8, 
                      color='#3498db', edgecolor='black')
        
        for bar, acc in zip(bars, bin_accuracies):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('Text Length (words)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy vs Text Length', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        
        # Sample distribution
        ax2.bar(bin_labels, bin_counts, alpha=0.8, color='#2ecc71', edgecolor='black')
        ax2.set_xlabel('Text Length (words)', fontsize=12)
        ax2.set_ylabel('Sample Count', fontsize=12)
        ax2.set_title('Sample Distribution by Text Length', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved length vs accuracy to {save_name}")
    
    # =========================================================================
    # 10. ROC CURVES (FIXED - TYPE-SAFE SPARSE MATRIX HANDLING)
    # =========================================================================
    
    def plot_roc_curves(self, labels: np.ndarray, probabilities: np.ndarray, 
                        save_name='roc_curves.png'):
        """
        Plot ROC curves (one-vs-rest)
        
        Args:
            labels: True labels
            probabilities: Prediction probabilities
            save_name: Filename
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels - sklearn may return sparse matrix
        labels_bin = label_binarize(labels, classes=[0, 1, 2])
        
        # ✅ CRITICAL FIX: Use proper type guard to handle sparse matrices safely
        # This resolves BOTH type checker errors:
        #   1. "Cannot access attribute 'toarray' for class 'ndarray'"
        #   2. "'__getitem__' method not defined on type 'spmatrix'"
        if isinstance(labels_bin, spmatrix):
            # Convert sparse matrix to dense array ONLY when needed
            labels_bin = labels_bin.toarray()  # type: ignore[union-attr]
        # After this check, type checker knows labels_bin is ndarray
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            # ✅ Now safe to index: labels_bin is guaranteed to be ndarray
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})',
                    color=self.colors[i])
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved ROC curves to {save_name}")
    
    # =========================================================================
    # 11. WORD CLOUD
    # =========================================================================
    
    def plot_wordcloud_errors(self, texts: List[str], predictions: np.ndarray, 
                              labels: np.ndarray, save_name='wordcloud_errors.png'):
        """
        Word cloud of misclassified texts
        
        Args:
            texts: List of texts
            predictions: Predictions
            labels: True labels
            save_name: Filename
        """
        errors = predictions != labels
        error_texts = [texts[i] for i in range(len(texts)) if errors[i]]
        
        if len(error_texts) == 0:
            print("⚠️  No errors to visualize")
            return
        
        # Combine all error texts
        error_text = ' '.join(error_texts)
        
        # Create word cloud
        wordcloud = WordCloud(width=1600, height=800, 
                             background_color='white',
                             colormap='Reds',
                             max_words=100).generate(error_text)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Misclassified Texts', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved word cloud to {save_name}")
    
    # =========================================================================
    # 12. MODEL COMPARISON BAR CHART
    # =========================================================================
    
    def plot_model_comparison_bars(self, models_metrics: Dict[str, Dict[str, float]], 
                                   save_name='model_comparison.png'):
        """
        Bar chart comparing models on multiple metrics
        
        Args:
            models_metrics: Dict mapping model names to metrics
            save_name: Filename
        """
        models = list(models_metrics.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [models_metrics[m][metric] for m in models]
            ax.bar(x + i * width, values, width, label=name, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=13)
        ax.set_ylabel('Score', fontsize=13)
        ax.set_title('Model Comparison - Multiple Metrics', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved model comparison to {save_name}")
    
    # =========================================================================
    # 13. LEARNING RATE SCHEDULE
    # =========================================================================
    
    def plot_lr_schedule(self, lr_history: List[float], save_name='lr_schedule.png'):
        """
        Plot learning rate schedule
        
        Args:
            lr_history: List of learning rates per step
            save_name: Filename
        """
        plt.figure(figsize=(12, 6))
        plt.plot(lr_history, linewidth=2, color='#3498db')
        plt.xlabel('Training Step', fontsize=13)
        plt.ylabel('Learning Rate', fontsize=13)
        plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved LR schedule to {save_name}")
    
    # =========================================================================
    # SUMMARY DASHBOARD
    # =========================================================================
    
    def create_summary_dashboard(self, metrics: Dict[str, Any], cm: np.ndarray, 
                                 save_name='summary_dashboard.png'):
        """
        Create comprehensive summary dashboard
        
        Args:
            metrics: Metrics dictionary
            cm: Confusion matrix
            save_name: Filename
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall metrics (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'],
            metrics['precision_macro'],
            metrics['recall_macro'],
            metrics['f1_macro']
        ]
        bars = ax1.barh(metric_names, metric_values, color='#3498db', alpha=0.8)
        for bar, value in zip(bars, metric_values):
            ax1.text(value, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontweight='bold')
        ax1.set_xlim(0, 1.0)
        ax1.set_title('Overall Metrics', fontweight='bold', fontsize=12)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Confusion matrix (top-middle and top-right)
        ax2 = fig.add_subplot(gs[0, 1:])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=12)
        ax2.set_ylabel('True')
        ax2.set_xlabel('Predicted')
        
        # 3. Per-class F1 (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        classes = list(metrics['per_class'].keys())
        f1_scores = [metrics['per_class'][c]['f1'] for c in classes]
        ax3.bar(classes, f1_scores, color=self.colors, alpha=0.8)
        ax3.set_ylabel('F1-Score')
        ax3.set_title('Per-Class F1 Scores', fontweight='bold', fontsize=12)
        ax3.set_ylim(0, 1.0)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Per-class precision (middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        precision_scores = [metrics['per_class'][c]['precision'] for c in classes]
        ax4.bar(classes, precision_scores, color=self.colors, alpha=0.8)
        ax4.set_ylabel('Precision')
        ax4.set_title('Per-Class Precision', fontweight='bold', fontsize=12)
        ax4.set_ylim(0, 1.0)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Per-class recall (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        recall_scores = [metrics['per_class'][c]['recall'] for c in classes]
        ax5.bar(classes, recall_scores, color=self.colors, alpha=0.8)
        ax5.set_ylabel('Recall')
        ax5.set_title('Per-Class Recall', fontweight='bold', fontsize=12)
        ax5.set_ylim(0, 1.0)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Class distribution (bottom-left)
        ax6 = fig.add_subplot(gs[2, 0])
        support = [metrics['per_class'][c]['support'] for c in classes]
        ax6.pie(support, labels=classes, autopct='%1.1f%%', 
               colors=self.colors, startangle=90)
        ax6.set_title('Class Distribution', fontweight='bold', fontsize=12)
        
        # 7. Metrics summary table (bottom-center and bottom-right)
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('tight')
        ax7.axis('off')
        
        table_data = []
        for class_name in classes:
            row = [
                class_name,
                f"{metrics['per_class'][class_name]['precision']:.3f}",
                f"{metrics['per_class'][class_name]['recall']:.3f}",
                f"{metrics['per_class'][class_name]['f1']:.3f}",
                f"{metrics['per_class'][class_name]['support']}"
            ]
            table_data.append(row)
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Class', 'Precision', 'Recall', 'F1', 'Support'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax7.set_title('Detailed Metrics', fontweight='bold', fontsize=12, pad=20)
        
        fig.suptitle('Model Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved summary dashboard to {save_name}")


if __name__ == "__main__":
    print("="*80)
    print("TESTING VISUALIZER")
    print("="*80)
    
    print("\nSentimentVisualizer module loaded successfully!")
    print("\nAvailable plot types (20+):")
    print("  1. Training curves (loss + accuracy)")
    print("  2. Confusion matrices (raw + normalized)")
    print("  3. Per-class F1 scores")
    print("  4. Model comparison radar chart")
    print("  5. Error distribution")
    print("  6. Confidence distribution")
    print("  7. Text length vs accuracy")
    print("  8. ROC curves (one-vs-rest)")
    print("  9. Word cloud of errors")
    print(" 10. Model comparison bars")
    print(" 11. Learning rate schedule")
    print(" 12. Summary dashboard")
    print(" ... and more!")
    
    print("\n✅ Visualizer module ready!")