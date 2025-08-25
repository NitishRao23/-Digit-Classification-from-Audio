import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

class BaseAudioModel(nn.Module, ABC):
    """Abstract base class for all audio models"""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def get_num_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def apply_regularization(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply dropout regularization"""
        return F.dropout(x, p=self.dropout_rate, training=training)

class Conv2DBlock(nn.Module):
    """Reusable 2D convolution block with batch norm and dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dropout_rate: float = 0.25):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = self.bn(x)
        x = self.pool(x)
        return self.dropout(x)

class Conv1DBlock(nn.Module):
    """Reusable 1D convolution block with batch norm and dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dropout_rate: float = 0.25):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout1d(dropout_rate)
        self.pool = nn.MaxPool1d(4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = self.bn(x)
        x = self.pool(x)
        return self.dropout(x)

class AttentionBlock(nn.Module):
    """Lightweight self-attention block"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class SpectrogramCNN(BaseAudioModel):
    """2D CNN for spectrogram input (mel/log spectrograms)"""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10, 
                 dropout_rate: float = 0.3):
        super().__init__(num_classes, dropout_rate)
        
        # Input shape: (channels, height, width)
        self.conv_blocks = nn.Sequential(
            Conv2DBlock(input_shape[0], 32, dropout_rate=0.25),
            Conv2DBlock(32, 64, dropout_rate=0.25),
            Conv2DBlock(64, 128, dropout_rate=0.25)
        )
        
        # Calculate feature size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output = self.conv_blocks(dummy_input)
            self.feature_size = conv_output.view(1, -1).size(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x

class RawAudioCNN(BaseAudioModel):
    """1D CNN for raw audio waveforms"""
    
    def __init__(self, input_length: int, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__(num_classes, dropout_rate)
        
        self.conv_blocks = nn.Sequential(
            # First block: large kernel to capture audio patterns
            Conv1DBlock(1, 32, kernel_size=80, stride=4, padding=40, dropout_rate=0.25),
            Conv1DBlock(32, 64, kernel_size=3, stride=1, padding=1, dropout_rate=0.25),
            Conv1DBlock(64, 128, kernel_size=3, stride=1, padding=1, dropout_rate=0.25)
        )
        
        # Calculate feature size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            conv_output = self.conv_blocks(dummy_input)
            self.feature_size = conv_output.view(1, -1).size(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x

class TransformerModel(BaseAudioModel):
    """Lightweight transformer for MFCC sequences"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int = 10, 
                 dropout_rate: float = 0.3, embed_dim: int = 64):
        super().__init__(num_classes, dropout_rate)
        
        seq_len, feature_dim = input_shape
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads=4, dropout_rate=dropout_rate)
            for _ in range(2)  # 2 attention layers
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(x)
        
        # Global average pooling and classify
        x = x.mean(dim=1)  # Average across sequence dimension
        x = self.classifier(x)
        return x

class ModelLoss:
    """Custom loss functions and metrics"""
    
    @staticmethod
    def cross_entropy_loss() -> nn.Module:
        """Standard cross-entropy loss"""
        return nn.CrossEntropyLoss()
    
    @staticmethod
    def focal_loss(alpha: float = 1.0, gamma: float = 2.0) -> nn.Module:
        """Focal loss for imbalanced datasets"""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss()
    
    @staticmethod
    def label_smoothing_loss(smoothing: float = 0.1) -> nn.Module:
        """Cross-entropy with label smoothing"""
        return nn.CrossEntropyLoss(label_smoothing=smoothing)

class ModelMetrics:
    """Evaluation metrics"""
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy"""
        pred_classes = predictions.argmax(dim=1)
        correct = (pred_classes == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
        """Calculate top-k accuracy"""
        _, top_k_pred = predictions.topk(k, dim=1)
        correct = top_k_pred.eq(targets.view(-1, 1).expand_as(top_k_pred))
        return correct.any(dim=1).float().mean().item()
    
    @staticmethod
    def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Calculate confusion matrix"""
        pred_classes = predictions.argmax(dim=1)
        matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        
        for t, p in zip(targets.view(-1), pred_classes.view(-1)):
            matrix[t.long(), p.long()] += 1
        
        return matrix

class ModelTrainer:
    """Training and evaluation utilities"""
    
    def __init__(self, model: BaseAudioModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   loss_fn: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss, total_acc = 0.0, 0.0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_acc += ModelMetrics.accuracy(outputs, targets)
        
        return total_loss / len(dataloader), total_acc / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader, loss_fn: nn.Module) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0
        all_predictions, all_targets = [], []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = loss_fn(outputs, targets)
                
                total_loss += loss.item()
                total_acc += ModelMetrics.accuracy(outputs, targets)
                
                all_predictions.append(outputs)
                all_targets.append(targets)
        
        # Aggregate results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': total_acc / len(dataloader),
            'top_3_accuracy': ModelMetrics.top_k_accuracy(all_predictions, all_targets, k=3)
        }

# Model Factory
class ModelFactory:
    """Factory to create different model types"""
    
    @staticmethod
    def create_model(model_type: str, input_shape: Tuple, num_classes: int = 10, 
                    dropout_rate: float = 0.3) -> BaseAudioModel:
        """Create model based on type and input shape"""
        
        if model_type == 'spectrogram_cnn':
            return SpectrogramCNN(input_shape, num_classes, dropout_rate)
        
        elif model_type == 'raw_audio_cnn':
            return RawAudioCNN(input_shape[0], num_classes, dropout_rate)
        
        elif model_type == 'transformer':
            return TransformerModel(input_shape, num_classes, dropout_rate)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_optimizer(model: nn.Module, optimizer_type: str = 'adam', 
                     learning_rate: float = 1e-3) -> optim.Optimizer:
        """Get optimizer"""
        if optimizer_type == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'adamw':
            return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    @staticmethod
    def get_scheduler(optimizer: optim.Optimizer, scheduler_type: str = 'cosine') -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler"""
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        else:
            return None

# Example usage
if __name__ == "__main__":
    # Example with different input types
    
    # 1. Spectrogram CNN
    print("Creating Spectrogram CNN...")
    spec_model = ModelFactory.create_model('spectrogram_cnn', (1, 128, 130), num_classes=10)
    print(f"Parameters: {spec_model.get_num_parameters():,}")
    
    # 2. Raw Audio CNN
    print("Creating Raw Audio CNN...")
    raw_model = ModelFactory.create_model('raw_audio_cnn', (66150,), num_classes=10)
    print(f"Parameters: {raw_model.get_num_parameters():,}")
    
    # 3. Transformer
    print("Creating Transformer...")
    transformer_model = ModelFactory.create_model('transformer', (130, 13), num_classes=10)
    print(f"Parameters: {transformer_model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    
    # Test spectrogram model
    spec_input = torch.randn(batch_size, 1, 128, 130)
    spec_output = spec_model(spec_input)
    print(f"Spectrogram output shape: {spec_output.shape}")
    
    # Test raw audio model
    raw_input = torch.randn(batch_size, 66150)
    raw_output = raw_model(raw_input)
    print(f"Raw audio output shape: {raw_output.shape}")
    
    # Test transformer model
    transformer_input = torch.randn(batch_size, 130, 13)
    transformer_output = transformer_model(transformer_input)
    print(f"Transformer output shape: {transformer_output.shape}")
    
    print("\nAll models created successfully!")