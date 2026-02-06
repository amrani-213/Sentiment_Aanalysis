"""
Error Analysis for Sentiment Analysis
Detailed analysis of model errors and patterns with strict type safety
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, TypedDict, Any
from collections import Counter, defaultdict
import re


# ✅ CRITICAL FIX: Define precise nested types to guide type checker
class ErrorByClassStats(TypedDict):
    total: int
    errors: int
    error_rate: float
    accuracy: float

class ConfidenceBinStats(TypedDict):
    total: int
    errors: int
    error_rate: float

class ConfidenceAnalysis(TypedDict):
    avg_confidence_all: float
    avg_confidence_errors: float
    avg_confidence_correct: float
    std_confidence_all: float
    by_confidence_bin: Dict[str, ConfidenceBinStats]

class LengthBinStats(TypedDict):
    total: int
    errors: int
    error_rate: float

class HighConfidenceError(TypedDict):
    index: int
    text: str
    true_label: str
    predicted_label: str
    confidence: float
    probabilities: List[float]


class ErrorAnalyzer:
    """
    Analyze model errors to identify patterns and insights
    
    Features:
    - Error categorization by length, sentiment, etc.
    - Common error patterns
    - Misclassification analysis
    - Text feature correlation with errors
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Args:
            class_names: List of class names
        """
        if class_names is None:
            self.class_names = ['Negative', 'Neutral', 'Positive']
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def analyze_errors(
        self,
        texts: List[str],
        true_labels: Union[np.ndarray, List[int]],
        pred_labels: Union[np.ndarray, List[int]],
        probabilities: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive error analysis with type-safe structure
        
        Returns:
            Dictionary with strictly typed nested results
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"ERROR ANALYSIS")
            print(f"{'='*80}")
        
        # Convert to numpy arrays if needed
        if not isinstance(true_labels, np.ndarray):
            true_labels = np.array(true_labels)
        if not isinstance(pred_labels, np.ndarray):
            pred_labels = np.array(pred_labels)
        
        # Identify errors
        errors = true_labels != pred_labels
        error_indices = np.where(errors)[0]
        correct_indices = np.where(~errors)[0]
        
        # ✅ FIX: Build results with explicit type separation
        results: Dict[str, Any] = {
            'total_samples': int(len(texts)),
            'total_errors': int(errors.sum()),
            'error_rate': float(errors.sum() / len(texts)),
            'error_indices': error_indices.tolist(),
        }
        
        # 1. Error distribution by true class (typed)
        errors_by_class: Dict[str, ErrorByClassStats] = self._analyze_errors_by_class(
            true_labels, errors, verbose=verbose
        )
        results['errors_by_class'] = errors_by_class  # ✅ RENAMED from 'errors_by_true_class' for consistency
        
        # 2. Confusion patterns
        results['confusion_patterns'] = self._analyze_confusion_patterns(
            true_labels, pred_labels, verbose=verbose
        )
        
        # 3. Text length analysis (typed)
        length_analysis: Dict[str, LengthBinStats] = self._analyze_by_text_length(
            texts, errors, verbose=verbose
        )
        results['length_analysis'] = length_analysis
        
        # 4. Confidence analysis (typed)
        confidence_analysis: ConfidenceAnalysis = self._analyze_confidence(
            probabilities, pred_labels, errors, verbose=verbose
        )
        results['confidence_analysis'] = confidence_analysis
        
        # 5. Lexical patterns
        results['lexical_patterns'] = self._analyze_lexical_patterns(
            texts, errors, error_indices, correct_indices, verbose=verbose
        )
        
        # 6. High-confidence errors (typed)
        high_conf_errors: List[HighConfidenceError] = self._find_high_confidence_errors(
            texts, true_labels, pred_labels, probabilities, 
            error_indices, verbose=verbose
        )
        results['high_confidence_errors'] = high_conf_errors
        
        return results
    
    def _analyze_errors_by_class(
        self, 
        true_labels: np.ndarray, 
        errors: np.ndarray, 
        verbose: bool = True
    ) -> Dict[str, ErrorByClassStats]:
        """Analyze error distribution by class with strict typing"""
        analysis: Dict[str, ErrorByClassStats] = {}
        
        for i, class_name in enumerate(self.class_names):
            class_mask = true_labels == i
            class_total = int(class_mask.sum())
            class_errors = int(errors[class_mask].sum())
            
            # ✅ Explicitly construct typed dict
            stats: ErrorByClassStats = {
                'total': class_total,
                'errors': class_errors,
                'error_rate': float(class_errors / max(class_total, 1)),
                'accuracy': float(1 - class_errors / max(class_total, 1))
            }
            analysis[class_name] = stats
        
        if verbose:
            print(f"\nErrors by True Class:")
            for class_name, stats in analysis.items():
                print(f"  {class_name:10s}: {stats['errors']:4d}/{stats['total']:4d} "
                      f"({stats['error_rate']:6.2%}) - Acc: {stats['accuracy']:.2%}")
        
        return analysis
    
    def _analyze_confusion_patterns(
        self, 
        true_labels: np.ndarray, 
        pred_labels: np.ndarray, 
        verbose: bool = True
    ) -> Dict[str, int]:
        """Analyze confusion pairs (true → predicted)"""
        patterns: Dict[Tuple[int, int], int] = defaultdict(int)
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label != pred_label:
                pair = (int(true_label), int(pred_label))
                patterns[pair] += 1
        
        # Convert to named pairs
        named_patterns: Dict[str, int] = {}
        for (true_idx, pred_idx), count in patterns.items():
            pair_name = f"{self.class_names[true_idx]} → {self.class_names[pred_idx]}"
            named_patterns[pair_name] = count
        
        # Sort by frequency
        sorted_patterns = dict(sorted(named_patterns.items(), 
                                     key=lambda x: x[1], reverse=True))
        
        if verbose:
            print(f"\nConfusion Patterns (top 5):")
            for i, (pattern, count) in enumerate(sorted_patterns.items()):
                if i >= 5:
                    break
                print(f"  {pattern:30s}: {count:4d}")
        
        return sorted_patterns
    
    def _analyze_by_text_length(
        self, 
        texts: List[str], 
        errors: np.ndarray, 
        verbose: bool = True
    ) -> Dict[str, LengthBinStats]:
        """Analyze errors by text length with strict typing"""
        text_lengths = np.array([len(text.split()) for text in texts])
        
        # Define length bins
        bins = [0, 10, 20, 30, 50, 100, np.inf]
        bin_labels = ['<10', '10-20', '20-30', '30-50', '50-100', '100+']
        
        analysis: Dict[str, LengthBinStats] = {}
        
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            bin_mask = (text_lengths >= low) & (text_lengths < high)
            bin_total = int(bin_mask.sum())
            bin_errors = int(errors[bin_mask].sum())
            
            if bin_total > 0:
                # ✅ Explicitly construct typed dict
                stats: LengthBinStats = {
                    'total': bin_total,
                    'errors': bin_errors,
                    'error_rate': float(bin_errors / bin_total)
                }
                analysis[bin_labels[i]] = stats
        
        if verbose:
            print(f"\nErrors by Text Length:")
            for bin_label, stats in analysis.items():
                print(f"  {bin_label:10s}: {stats['errors']:4d}/{stats['total']:4d} "
                      f"({stats['error_rate']:6.2%})")
        
        return analysis
    
    def _analyze_confidence(
        self, 
        probabilities: np.ndarray, 
        pred_labels: np.ndarray, 
        errors: np.ndarray, 
        verbose: bool = True
    ) -> ConfidenceAnalysis:
        """Analyze prediction confidence with strict typing"""
        # Get confidence (max probability)
        confidences = np.max(probabilities, axis=1)
        
        # Confidence for errors vs correct
        error_confidences = confidences[errors]
        correct_confidences = confidences[~errors]
        
        # ✅ Build typed structure explicitly
        analysis: ConfidenceAnalysis = {
            'avg_confidence_all': float(confidences.mean()),
            'avg_confidence_errors': float(error_confidences.mean()) if len(error_confidences) > 0 else 0.0,
            'avg_confidence_correct': float(correct_confidences.mean()) if len(correct_confidences) > 0 else 0.0,
            'std_confidence_all': float(confidences.std()),
            'by_confidence_bin': {}
        }
        
        # Confidence bins
        bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['<0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9+']
        
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            bin_mask = (confidences >= low) & (confidences < high)
            bin_total = int(bin_mask.sum())
            bin_errors = int(errors[bin_mask].sum())
            
            if bin_total > 0:
                # ✅ Explicitly construct nested typed dict
                bin_stats: ConfidenceBinStats = {
                    'total': bin_total,
                    'errors': bin_errors,
                    'error_rate': float(bin_errors / bin_total)
                }
                analysis['by_confidence_bin'][bin_labels[i]] = bin_stats
        
        if verbose:
            print(f"\nConfidence Analysis:")
            print(f"  Avg confidence (all):     {analysis['avg_confidence_all']:.4f}")
            print(f"  Avg confidence (errors):  {analysis['avg_confidence_errors']:.4f}")
            print(f"  Avg confidence (correct): {analysis['avg_confidence_correct']:.4f}")
            
            print(f"\n  By Confidence Bin:")
            for bin_label, stats in analysis['by_confidence_bin'].items():
                print(f"    {bin_label:10s}: {stats['errors']:4d}/{stats['total']:4d} "
                      f"({stats['error_rate']:6.2%})")
        
        return analysis
    
    def _analyze_lexical_patterns(
        self, 
        texts: List[str], 
        errors: np.ndarray, 
        error_indices: np.ndarray, 
        correct_indices: np.ndarray, 
        verbose: bool = True
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Analyze lexical patterns in errors"""
        
        # Extract words from error and correct samples
        error_texts = [texts[i] for i in error_indices]
        correct_texts = [texts[i] for i in correct_indices[:len(error_indices)]]  # Sample same size
        
        # Get word frequencies
        error_words = self._extract_words(error_texts)
        correct_words = self._extract_words(correct_texts)
        
        # Find words more common in errors
        error_word_freq = Counter(error_words)
        correct_word_freq = Counter(correct_words)
        
        # Calculate enrichment
        enriched_in_errors: Dict[str, Dict[str, Union[int, float]]] = {}
        for word, error_count in error_word_freq.most_common(100):
            correct_count = correct_word_freq.get(word, 0)
            
            # Skip very common words
            if error_count < 3:
                continue
            
            # Enrichment ratio
            enrichment = (error_count / len(error_texts)) / max((correct_count / len(correct_texts)), 0.001)
            
            if enrichment > 2.0:  # At least 2x more common in errors
                enriched_in_errors[word] = {
                    'error_freq': int(error_count),
                    'correct_freq': int(correct_count),
                    'enrichment': float(enrichment)
                }
        
        # Sort by enrichment
        sorted_words = dict(sorted(enriched_in_errors.items(), 
                                  key=lambda x: x[1]['enrichment'], 
                                  reverse=True)[:20])
        
        if verbose and len(sorted_words) > 0:
            print(f"\nWords Enriched in Errors (top 10):")
            for i, (word, stats) in enumerate(sorted_words.items()):
                if i >= 10:
                    break
                print(f"  {word:15s}: {stats['enrichment']:.2f}x "
                      f"(errors: {stats['error_freq']}, correct: {stats['correct_freq']})")
        
        return sorted_words
    
    def _extract_words(self, texts: List[str]) -> List[str]:
        """Extract words from texts"""
        words = []
        for text in texts:
            # Simple tokenization
            text_words = re.findall(r'\b\w+\b', text.lower())
            words.extend(text_words)
        return words
    
    def _find_high_confidence_errors(
        self, 
        texts: List[str], 
        true_labels: np.ndarray, 
        pred_labels: np.ndarray,
        probabilities: np.ndarray, 
        error_indices: np.ndarray, 
        verbose: bool = True
    ) -> List[HighConfidenceError]:
        """Find high-confidence errors (most concerning)"""
        
        if len(error_indices) == 0:
            return []
        
        # Get confidence for errors
        error_confidences = np.max(probabilities[error_indices], axis=1)
        
        # Find top 20 highest confidence errors
        top_indices = error_indices[np.argsort(error_confidences)[-20:]]
        
        high_conf_errors: List[HighConfidenceError] = []
        for idx in reversed(top_indices):  # Highest confidence first
            error: HighConfidenceError = {
                'index': int(idx),
                'text': str(texts[idx])[:200],  # First 200 chars, ensure string
                'true_label': self.class_names[int(true_labels[idx])],
                'predicted_label': self.class_names[int(pred_labels[idx])],
                'confidence': float(np.max(probabilities[idx])),
                'probabilities': [float(p) for p in probabilities[idx].tolist()]
            }
            high_conf_errors.append(error)
        
        if verbose:
            print(f"\nHigh-Confidence Errors (top 5):")
            for i, error in enumerate(high_conf_errors[:5]):
                print(f"\n  {i+1}. Confidence: {error['confidence']:.3f}")
                print(f"     True: {error['true_label']}, Predicted: {error['predicted_label']}")
                print(f"     Text: {error['text'][:100]}...")
        
        return high_conf_errors
    
    def compare_models(
        self, 
        model_results_dict: Dict[str, Dict[str, Any]], 
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compare error patterns across multiple models
        
        Returns:
            Comparison dictionary with typed structure
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"MODEL ERROR COMPARISON")
            print(f"{'='*80}")
        
        comparison: Dict[str, Any] = {
            'models': list(model_results_dict.keys()),
            'error_rates': {},
            'agreement': {},
            'unique_errors': {}
        }
        
        # Overall error rates
        for model_name, results in model_results_dict.items():
            comparison['error_rates'][model_name] = float(results['error_rate'])
        
        if verbose:
            print(f"\nOverall Error Rates:")
            for model_name, error_rate in comparison['error_rates'].items():
                print(f"  {model_name:20s}: {error_rate:.2%}")
        
        # Pairwise agreement on errors
        model_names = list(model_results_dict.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                errors1 = set(model_results_dict[model1]['error_indices'])
                errors2 = set(model_results_dict[model2]['error_indices'])
                
                if errors1 or errors2:
                    agreement = len(errors1 & errors2) / len(errors1 | errors2)
                    comparison['agreement'][f"{model1} vs {model2}"] = float(agreement)
        
        if verbose and comparison['agreement']:
            print(f"\nError Agreement Between Models:")
            for pair, agreement in comparison['agreement'].items():
                print(f"  {pair:40s}: {agreement:.2%}")
        
        return comparison


if __name__ == "__main__":
    print("="*80)
    print("TESTING ERROR ANALYZER")
    print("="*80)
    
    print("\nErrorAnalyzer module loaded successfully!")
    print("\nFeatures:")
    print("  ✅ Error distribution by class")
    print("  ✅ Confusion pattern analysis")
    print("  ✅ Text length correlation")
    print("  ✅ Confidence analysis")
    print("  ✅ Lexical pattern detection")
    print("  ✅ High-confidence error identification")
    print("  ✅ Multi-model comparison")
    
    print("\nTo use this module:")
    print("  1. Create analyzer: analyzer = ErrorAnalyzer()")
    print("  2. Run analysis: results = analyzer.analyze_errors(texts, true, pred, probs)")
    print("  3. Compare models: analyzer.compare_models(model_results)")
    
    print("\n✅ Error Analyzer module ready!")