import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import os
from collections import defaultdict
import pandas as pd

class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for results
        self.predictions = []
        self.probabilities = []
        self.targets = []
        self.inference_times = []
    
    def evaluate_test_set(self, test_loader, class_names=None):
        """Evaluate model on held-out test set"""
        print("Evaluating on test set...")
        
        self.model.eval()
        correct = 0
        total = 0
        
        # Clear previous results
        self.predictions = []
        self.probabilities = []
        self.targets = []
        self.inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = self.model(data)
                end_time = time.time()
                
                # Store inference time (per sample)
                batch_time = (end_time - start_time) / data.size(0)
                self.inference_times.extend([batch_time] * data.size(0))
                
                # Get predictions and probabilities
                if output.size(1) > 1:  # Multi-class
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                else:  # Binary
                    probs = torch.sigmoid(output)
                    preds = (probs > 0.5).long().squeeze()
                
                # Store results
                self.predictions.extend(preds.cpu().numpy())
                self.probabilities.extend(probs.cpu().numpy())
                self.targets.extend(target.cpu().numpy())
                
                # Calculate accuracy
                correct += (preds == target).sum().item()
                total += target.size(0)
        
        # Convert to numpy arrays
        self.predictions = np.array(self.predictions)
        self.probabilities = np.array(self.probabilities)
        self.targets = np.array(self.targets)
        
        accuracy = 100. * correct / total
        avg_inference_time = np.mean(self.inference_times) * 1000  # Convert to ms
        
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Average Inference Time: {avg_inference_time:.3f} ms per sample")
        
        return {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'avg_inference_time_ms': avg_inference_time
        }
    
    def analyze_per_class_performance(self, class_names=None):
        """Analyze performance for each class/digit"""
        print("\nPer-class Performance Analysis:")
        print("-" * 50)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(self.targets)))]
        
        # Generate classification report
        report = classification_report(
            self.targets, self.predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Create DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        per_class_results = {}
        for i, class_name in enumerate(class_names):
            if class_name in df_report.index:
                precision = df_report.loc[class_name, 'precision']
                recall = df_report.loc[class_name, 'recall']
                f1 = df_report.loc[class_name, 'f1-score']
                support = int(df_report.loc[class_name, 'support'])
                
                print(f"{class_name:<12} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")
                
                per_class_results[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': support
                }
        
        # Overall metrics
        print("-" * 60)
        accuracy = df_report.loc['accuracy', 'precision']
        macro_avg = df_report.loc['macro avg']
        weighted_avg = df_report.loc['weighted avg']
        
        print(f"{'Accuracy':<12} {accuracy:<10.3f}")
        print(f"{'Macro Avg':<12} {macro_avg['precision']:<10.3f} {macro_avg['recall']:<10.3f} {macro_avg['f1-score']:<10.3f}")
        print(f"{'Weighted Avg':<12} {weighted_avg['precision']:<10.3f} {weighted_avg['recall']:<10.3f} {weighted_avg['f1-score']:<10.3f}")
        
        return per_class_results
    
    def plot_confusion_matrix(self, class_names=None, save_path=None):
        """Create and plot confusion matrix"""
        cm = confusion_matrix(self.targets, self.predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        
        return cm
    
    def error_analysis(self, class_names=None, top_errors=5):
        """Analyze model errors and misclassifications"""
        print(f"\nError Analysis (Top {top_errors} Confusion Pairs):")
        print("-" * 50)
        
        cm = confusion_matrix(self.targets, self.predictions)
        
        # Find top error pairs (excluding diagonal)
        error_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:  # Misclassification
                    error_pairs.append((i, j, cm[i, j]))
        
        # Sort by error count
        error_pairs.sort(key=lambda x: x[2], reverse=True)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
        
        print(f"{'True Class':<12} {'Predicted':<12} {'Count':<8} {'% of True':<10}")
        print("-" * 50)
        
        error_analysis = {}
        for true_idx, pred_idx, count in error_pairs[:top_errors]:
            true_class = class_names[true_idx]
            pred_class = class_names[pred_idx]
            percentage = 100 * count / cm[true_idx, :].sum()
            
            print(f"{true_class:<12} {pred_class:<12} {count:<8} {percentage:<10.1f}%")
            
            error_analysis[f"{true_class}_as_{pred_class}"] = {
                'count': count,
                'percentage': percentage
            }
        
        # Find samples with lowest confidence in correct predictions
        self.analyze_confidence()
        
        return error_analysis
    
    def analyze_confidence(self):
        """Analyze model confidence in predictions"""
        print(f"\nConfidence Analysis:")
        print("-" * 30)
        
        # Get confidence scores (max probability)
        if self.probabilities.ndim > 1:
            confidences = np.max(self.probabilities, axis=1)
        else:
            confidences = np.maximum(self.probabilities, 1 - self.probabilities)
        
        # Separate correct and incorrect predictions
        correct_mask = self.predictions == self.targets
        
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        print(f"Average confidence (correct): {np.mean(correct_confidences):.3f}")
        print(f"Average confidence (incorrect): {np.mean(incorrect_confidences):.3f}")
        print(f"Low confidence correct predictions: {np.sum(correct_confidences < 0.8)}")
        print(f"High confidence incorrect predictions: {np.sum(incorrect_confidences > 0.8)}")
        
        # Plot confidence distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if np.sum(mask) > 0:
                acc = np.mean(correct_mask[mask])
                bin_accuracies.append(acc)
            else:
                bin_accuracies.append(0)
        
        plt.plot(confidence_bins[:-1], bin_accuracies, 'bo-')
        plt.xlabel('Confidence Bin')
        plt.ylabel('Accuracy')
        plt.title('Calibration Plot')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def measure_model_performance(self):
        """Measure model size and inference speed"""
        print("\nModel Performance Metrics:")
        print("-" * 30)
        
        # Model size
        param_count = sum(p.numel() for p in self.model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        
        print(f"Parameters: {param_count:,}")
        print(f"Model size: {model_size_mb:.2f} MB")
        
        # Inference speed analysis
        if self.inference_times:
            times_ms = np.array(self.inference_times) * 1000
            print(f"Mean inference time: {np.mean(times_ms):.3f} ms")
            print(f"Median inference time: {np.median(times_ms):.3f} ms")
            print(f"95th percentile: {np.percentile(times_ms, 95):.3f} ms")
            print(f"Throughput: {1000/np.mean(times_ms):.1f} samples/second")
        
        return {
            'parameters': param_count,
            'model_size_mb': model_size_mb,
            'mean_inference_time_ms': np.mean(times_ms) if self.inference_times else None,
            'throughput_samples_per_sec': 1000/np.mean(times_ms) if self.inference_times else None
        }
    
    def generate_evaluation_report(self, class_names=None, save_dir="evaluation_results"):
        """Generate comprehensive evaluation report"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("="*60)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*60)
        
        # 1. Overall performance
        overall_metrics = self.evaluate_test_set(None, class_names)
        
        # 2. Per-class analysis
        per_class_results = self.analyze_per_class_performance(class_names)
        
        # 3. Confusion matrix
        cm = self.plot_confusion_matrix(class_names, f"{save_dir}/confusion_matrix.png")
        
        # 4. Error analysis
        error_results = self.error_analysis(class_names)
        
        # 5. Model performance
        performance_metrics = self.measure_model_performance()
        
        # Save results to JSON
        import json
        results = {
            'overall_metrics': overall_metrics,
            'per_class_results': per_class_results,
            'error_analysis': error_results,
            'performance_metrics': performance_metrics,
            'confusion_matrix': cm.tolist()
        }
        
        with open(f"{save_dir}/evaluation_report.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nEvaluation report saved to: {save_dir}/")
        return results

# Quick evaluation function
def quick_evaluate(model, test_loader, class_names=None):
    """Quick model evaluation with essential metrics"""
    evaluator = ModelEvaluator(model)
    
    # Test set evaluation
    overall_metrics = evaluator.evaluate_test_set(test_loader, class_names)
    
    # Per-class performance
    per_class_results = evaluator.analyze_per_class_performance(class_names)
    
    # Confusion matrix
    cm = evaluator.plot_confusion_matrix(class_names)
    
    # Error analysis
    error_results = evaluator.error_analysis(class_names, top_errors=3)
    
    # Performance metrics
    performance_metrics = evaluator.measure_model_performance()
    
    return {
        'overall': overall_metrics,
        'per_class': per_class_results,
        'errors': error_results,
        'performance': performance_metrics
    }

# Example usage
def main():
    """Example usage with dummy model and data"""
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy model and data
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 10)  # 10 classes (digits 0-9)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    # Generate sample data
    np.random.seed(42)
    X_test = np.random.randn(500, 20)
    y_test = np.random.randint(0, 10, 500)
    
    # Create test loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimpleModel()
    
    # Define class names (for digit recognition)
    digit_names = [f"Digit_{i}" for i in range(10)]
    
    print("Running comprehensive model evaluation...")
    
    # Method 1: Quick evaluation
    results = quick_evaluate(model, test_loader, digit_names)
    
    # Method 2: Detailed evaluation with report
    evaluator = ModelEvaluator(model)
    evaluator.evaluate_test_set(test_loader, digit_names)
    comprehensive_results = evaluator.generate_evaluation_report(digit_names)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()