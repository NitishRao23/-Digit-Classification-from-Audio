import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import pandas as pd

# Simple Neural Network Models
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

# Metrics Calculator
class MetricsCalculator:
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        if self.task_type == 'classification':
            pred_labels = np.argmax(preds, axis=1) if preds.ndim > 1 else (preds > 0.5).astype(int)
            return {
                'accuracy': accuracy_score(targets, pred_labels),
                'f1_score': f1_score(targets, pred_labels, average='weighted'),
                'confusion_matrix': confusion_matrix(targets, pred_labels)
            }
        else:  # regression
            return {
                'mse': np.mean((preds - targets)**2),
                'mae': np.mean(np.abs(preds - targets))
            }
    
    def plot_confusion_matrix(self, title="Confusion Matrix"):
        if self.task_type != 'classification':
            return
        
        metrics = self.compute()
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Training Function
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, optimizer_name='adam'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9)
    }
    optimizer = optimizers[optimizer_name](model.parameters(), lr=lr)
    
    metrics_tracker = MetricsCalculator()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        metrics_tracker.reset()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                pred_probs = torch.softmax(output, dim=1)
                metrics_tracker.update(pred_probs, target)
        
        val_metrics = metrics_tracker.compute()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Acc: {val_metrics["accuracy"]:.4f}')
    
    return val_metrics

# Hyperparameter Tuner
class HyperparameterTuner:
    def __init__(self, model_class, X, y, cv_folds=5):
        self.model_class = model_class
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.results = []
    
    def tune(self, param_grid):
        print(f"Testing {len(list(ParameterGrid(param_grid)))} combinations with {self.cv_folds}-fold CV")
        
        kf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for params in ParameterGrid(param_grid):
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X, self.y)):
                # Split and scale data
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                
                # Create data loaders
                train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
                val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
                
                train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)
                
                # Create and train model
                model = self.model_class(
                    input_size=self.X.shape[1],
                    hidden_size=params['hidden_size'],
                    output_size=len(np.unique(self.y)),
                    dropout=params['dropout']
                )
                
                metrics = train_model(
                    model, train_loader, val_loader,
                    epochs=params['epochs'],
                    lr=params['learning_rate'],
                    optimizer_name=params['optimizer']
                )
                
                fold_scores.append(metrics['accuracy'])
            
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            self.results.append({
                'params': params,
                'mean_accuracy': avg_score,
                'std_accuracy': std_score,
                'all_scores': fold_scores
            })
            
            print(f"Params: {params} -> Accuracy: {avg_score:.4f} ± {std_score:.4f}")
        
        # Sort by performance
        self.results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
        print(f"\nBest accuracy: {self.results[0]['mean_accuracy']:.4f}")
        print(f"Best params: {self.results[0]['params']}")
        
        return self.results
    
    def plot_results(self, top_n=5):
        top_results = self.results[:top_n]
        
        configs = [f"Config {i+1}" for i in range(len(top_results))]
        scores = [r['mean_accuracy'] for r in top_results]
        stds = [r['std_accuracy'] for r in top_results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(configs, scores, yerr=stds, capsize=5)
        plt.title(f'Top {top_n} Hyperparameter Configurations')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print top configurations
        print("\nTop configurations:")
        for i, result in enumerate(top_results):
            print(f"{i+1}. Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
            print(f"   {result['params']}\n")

# Baseline Comparison
def compare_baselines(X, y, cv_folds=3):
    print("Running baseline comparison...")
    
    models = {
        'Simple MLP': SimpleNet,
        'Deep MLP': DeepNet
    }
    
    # Simple parameter set for baselines
    base_params = {
        'batch_size': [32],
        'learning_rate': [0.001],
        'optimizer': ['adam'],
        'epochs': [30],
        'hidden_size': [128],
        'dropout': [0.2]
    }
    
    results = {}
    
    for name, model_class in models.items():
        print(f"\nTesting {name}...")
        
        tuner = HyperparameterTuner(model_class, X, y, cv_folds)
        # Test only one configuration for baseline
        baseline_results = tuner.tune([base_params])
        
        results[name] = baseline_results[0]['mean_accuracy']
        print(f"{name} accuracy: {results[name]:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    models = list(results.keys())
    scores = list(results.values())
    
    bars = plt.bar(models, scores, color=['skyblue', 'lightgreen'])
    plt.title('Baseline Model Comparison')
    plt.ylabel('Accuracy')
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Main execution
def main():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 3, 1000)
    
    print("=== MODEL TRAINING & HYPERPARAMETER TUNING ===\n")
    
    # 1. Baseline comparison
    baseline_results = compare_baselines(X, y, cv_folds=3)
    
    # 2. Hyperparameter tuning
    print("\n=== HYPERPARAMETER TUNING ===")
    
    param_grid = {
        'batch_size': [16, 32, 64],
        'learning_rate': [0.001, 0.01],
        'optimizer': ['adam', 'adamw'],
        'epochs': [20, 30],
        'hidden_size': [64, 128],
        'dropout': [0.1, 0.3]
    }
    
    tuner = HyperparameterTuner(SimpleNet, X, y, cv_folds=3)
    results = tuner.tune(param_grid)
    
    # 3. Plot results
    tuner.plot_results(top_n=5)
    
    # 4. Train final model with best parameters
    print("\n=== FINAL MODEL TRAINING ===")
    best_params = results[0]['params']
    print(f"Training final model with: {best_params}")
    
    # Split data for final training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False)
    
    # Train final model
    final_model = SimpleNet(
        input_size=X.shape[1],
        hidden_size=best_params['hidden_size'],
        output_size=len(np.unique(y)),
        dropout=best_params['dropout']
    )
    
    test_metrics = train_model(
        final_model, train_loader, test_loader,
        epochs=best_params['epochs'],
        lr=best_params['learning_rate'],
        optimizer_name=best_params['optimizer']
    )
    
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Plot confusion matrix
    final_tracker = MetricsCalculator()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = final_model(data)
            pred_probs = torch.softmax(output, dim=1)
            final_tracker.update(pred_probs, target)
    
    final_tracker.plot_confusion_matrix("Final Model Confusion Matrix")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()