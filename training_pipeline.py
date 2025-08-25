import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import pickle

class CustomDataset(Dataset):
    """Custom Dataset class for handling your data"""
    
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = {'features': self.X[idx], 'target': self.y[idx]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample['features'], sample['target']

class DataSplitter:
    """Handle train/validation/test splits"""
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def split_data(self, X, y, scale_features=True):
        """
        Split data into train/validation/test sets
        
        Args:
            X: Features
            y: Target variables
            scale_features: Whether to scale features
        
        Returns:
            Dictionary containing all splits
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
        
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        print(f"Data splits created:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        return splits

class DataLoaderManager:
    """Manage data loaders with proper batching"""
    
    def __init__(self, batch_size=32, num_workers=4, pin_memory=True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def create_data_loaders(self, data_splits):
        """
        Create data loaders for train, validation, and test sets
        
        Args:
            data_splits: Dictionary containing data splits
        
        Returns:
            Dictionary containing data loaders
        """
        # Create datasets
        train_dataset = CustomDataset(data_splits['X_train'], data_splits['y_train'])
        val_dataset = CustomDataset(data_splits['X_val'], data_splits['y_val'])
        test_dataset = CustomDataset(data_splits['X_test'], data_splits['y_test'])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

class EarlyStopping:
    """Early stopping implementation"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

class ModelCheckpoint:
    """Save model checkpoints"""
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def __call__(self, current_score, model, epoch, optimizer=None):
        if self.save_best_only:
            if self.best_score is None or current_score < self.best_score:
                self.best_score = current_score
                self.save_model(model, epoch, current_score, optimizer)
        else:
            self.save_model(model, epoch, current_score, optimizer)
    
    def save_model(self, model, epoch, score, optimizer):
        """Save model state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, self.filepath)
        print(f"Model checkpoint saved at epoch {epoch} with {self.monitor}: {score:.6f}")

class TrainingLoop:
    """Complete training loop implementation"""
    
    def __init__(self, model, criterion, optimizer, device=None, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target.long() if target.dtype==torch.float32 and len(target.shape)==1 else target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Calculate accuracy (for classification)
            if len(target.shape) == 1 or target.shape[1] == 1:  # Classification
                pred = output.argmax(dim=1) if output.shape[1] > 1 else (output > 0.5).float()
                correct += pred.eq(target.view_as(pred).long()).sum().item()
                total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target.long() if target.dtype==torch.float32 and len(target.shape)==1 else target)
                
                # Statistics
                total_loss += loss.item()
                
                # Calculate accuracy
                if len(target.shape) == 1 or target.shape[1] == 1:  # Classification
                    pred = output.argmax(dim=1) if output.shape[1] > 1 else (output > 0.5).float()
                    correct += pred.eq(target.view_as(pred).long()).sum().item()
                    total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def fit(self, train_loader, val_loader, epochs, early_stopping=None, checkpoint=None, verbose=True):
        """
        Complete training loop with early stopping and checkpointing
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping: EarlyStopping instance
            checkpoint: ModelCheckpoint instance
            verbose: Whether to print progress
        """
        print(f"Starting training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(epochs):
            if verbose:
                print(f'\nEpoch {epoch+1}/{epochs}')
                print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch results
            if verbose:
                print(f'Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # no verbose here to avoid issues
                else:
                    self.scheduler.step()
            
            # Model checkpointing
            if checkpoint:
                checkpoint(val_loss, self.model, epoch, self.optimizer)
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_loss, self.model):
                    print(f'\nEarly stopping triggered at epoch {epoch+1}')
                    break
        
        print('\nTraining completed!')
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        print("Evaluating on test set...")
        test_loss, test_acc = self.validate_epoch(test_loader)
        print(f'Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.2f}%')
        return test_loss, test_acc
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()

# Example model for demonstration
class SimpleNet(nn.Module):
    """Example neural network"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(SimpleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    """Example usage of the training pipeline"""
    
    # Generate sample data (replace with your actual data)
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    print("Creating training pipeline...")
    
    # 1. Create train/validation/test splits
    splitter = DataSplitter(test_size=0.2, val_size=0.2, random_state=42)
    data_splits = splitter.split_data(X, y, scale_features=True)
    
    # 2. Create data loaders
    loader_manager = DataLoaderManager(batch_size=32, num_workers=2)
    data_loaders = loader_manager.create_data_loaders(data_splits)
    
    # 3. Initialize model
    model = SimpleNet(
        input_size=n_features,
        hidden_size=64,
        output_size=n_classes,
        dropout_rate=0.3
    )
    
    # 4. Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # CHANGE: Removed verbose=True to avoid errors on some PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # 5. Set up early stopping and checkpointing
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"checkpoints/best_model_{timestamp}.pth"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    
    # 6. Create training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = TrainingLoop(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )
    
    # 7. Train the model
    history = trainer.fit(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        epochs=100,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        verbose=True
    )
    
    # 8. Evaluate on test set
    test_loss, test_acc = trainer.evaluate(data_loaders['test'])
    
    # 9. Plot training history
    trainer.plot_history(save_path=f"training_history_{timestamp}.png")
    
    # 10. Save training history
    with open(f"history_{timestamp}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining pipeline completed successfully!")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Best model saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()
