import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

class AudioDigitModels:
    """Collection of lightweight models for audio digit classification"""
    
    def __init__(self, num_classes=10, input_shape=None):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.models = {}
        self.scalers = {}
    
    def build_2d_cnn(self, input_shape: Tuple[int, int, int]) -> keras.Model:
        """
        Simple 2D CNN for spectrograms
        Input: (height, width, channels) - e.g., (128, 130, 1) for mel spectrograms
        """
        model = keras.Sequential([
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_1d_cnn(self, input_shape: Tuple[int,]) -> keras.Model:
        """
        1D CNN for raw audio waveforms
        Input: (time_steps,) - e.g., (66150,) for 3-second audio at 22050Hz
        """
        model = keras.Sequential([
            # Reshape for 1D conv
            layers.Reshape((input_shape[0], 1)),
            
            # First conv block
            layers.Conv1D(32, 80, strides=4, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(4),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv1D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(4),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(4),
            layers.Dropout(0.25),
            
            # Classifier
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_lightweight_transformer(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Lightweight transformer for sequence data (MFCC features)
        Input: (time_steps, features) - e.g., (130, 13) for MFCC
        """
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        x = layers.Dense(64)(inputs)
        
        # Simple attention block
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=16
        )(x, x)
        
        # Add & norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(128, activation='relu')(x)
        ff_output = layers.Dense(64)(ff_output)
        
        # Add & norm
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
        
        # Global average pooling and classifier
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def create_traditional_ml_models(self) -> Dict[str, Any]:
        """
        Traditional ML models for MFCC features
        Returns dictionary of scikit-learn models
        """
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            ),
            'svm_linear': SVC(
                kernel='linear',
                C=1.0,
                random_state=42
            )
        }
    
    def prepare_data_for_model(self, X: np.ndarray, model_type: str) -> np.ndarray:
        """
        Prepare data based on model requirements
        """
        if model_type == '2d_cnn':
            # For spectrograms: ensure 3D shape (height, width, channels)
            if len(X.shape) == 3 and X.shape[-1] != 1:
                X = np.expand_dims(X, axis=-1)
            return X
        
        elif model_type == '1d_cnn':
            # For raw audio: flatten if needed
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            return X
        
        elif model_type == 'transformer':
            # For MFCC: transpose if needed to have (batch, time, features)
            if len(X.shape) == 3 and X.shape[1] > X.shape[2]:
                X = np.transpose(X, (0, 2, 1))
            return X
        
        elif model_type == 'traditional':
            # For traditional ML: flatten to 2D
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            return X
        
        return X
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs=50) -> Dict[str, Any]:
        """
        Train a specific model and return training history
        """
        print(f"\nTraining {model_name}...")
        print(f"Training data shape: {X_train.shape}")
        
        if model_name == '2d_cnn':
            # Prepare data for 2D CNN
            X_train = self.prepare_data_for_model(X_train, '2d_cnn')
            X_val = self.prepare_data_for_model(X_val, '2d_cnn')
            
            # Build model
            model = self.build_2d_cnn(X_train.shape[1:])
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            self.models[model_name] = model
            return {'history': history.history}
        
        elif model_name == '1d_cnn':
            # Prepare data for 1D CNN
            X_train = self.prepare_data_for_model(X_train, '1d_cnn')
            X_val = self.prepare_data_for_model(X_val, '1d_cnn')
            
            # Build model
            model = self.build_1d_cnn(X_train.shape[1:])
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            self.models[model_name] = model
            return {'history': history.history}
        
        elif model_name == 'transformer':
            # Prepare data for transformer
            X_train = self.prepare_data_for_model(X_train, 'transformer')
            X_val = self.prepare_data_for_model(X_val, 'transformer')
            
            # Build model
            model = self.build_lightweight_transformer(X_train.shape[1:])
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            self.models[model_name] = model
            return {'history': history.history}
        
        elif model_name in ['random_forest', 'svm', 'svm_linear']:
            # Prepare data for traditional ML
            X_train = self.prepare_data_for_model(X_train, 'traditional')
            X_val = self.prepare_data_for_model(X_val, 'traditional')
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Get model
            ml_models = self.create_traditional_ml_models()
            model = ml_models[model_name]
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
            val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
            
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            return {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            }
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model on test data
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet!")
        
        model = self.models[model_name]
        
        if model_name in ['random_forest', 'svm', 'svm_linear']:
            # Traditional ML evaluation
            X_test = self.prepare_data_for_model(X_test, 'traditional')
            X_test_scaled = self.scalers[model_name].transform(X_test)
            
            predictions = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions)
            
            print(f"\n{model_name} Test Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions))
            
            return {'accuracy': accuracy}
        
        else:
            # Neural network evaluation
            if model_name == '2d_cnn':
                X_test = self.prepare_data_for_model(X_test, '2d_cnn')
            elif model_name == '1d_cnn':
                X_test = self.prepare_data_for_model(X_test, '1d_cnn')
            elif model_name == 'transformer':
                X_test = self.prepare_data_for_model(X_test, 'transformer')
            
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            print(f"\n{model_name} Test Results:")
            print(f"Loss: {test_loss:.4f}")
            print(f"Accuracy: {test_accuracy:.4f}")
            
            return {'loss': test_loss, 'accuracy': test_accuracy}
    
    def compare_models(self, results: Dict[str, Dict]) -> None:
        """
        Compare model performance
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        for model_name, result in results.items():
            if 'accuracy' in result:
                acc = result['accuracy']
                print(f"{model_name:15} | Accuracy: {acc:.4f}")
        
        # Find best model
        best_model = max(results.keys(), 
                        key=lambda x: results[x].get('accuracy', 0))
        best_acc = results[best_model]['accuracy']
        
        print(f"\nBest Model: {best_model} (Accuracy: {best_acc:.4f})")
    
    def plot_training_history(self, histories: Dict[str, Dict]) -> None:
        """
        Plot training histories for neural network models
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for model_name, result in histories.items():
            if 'history' in result:
                history = result['history']
                
                # Plot accuracy
                ax1.plot(history['accuracy'], label=f'{model_name} - Train')
                ax1.plot(history['val_accuracy'], label=f'{model_name} - Val', linestyle='--')
                
                # Plot loss
                ax2.plot(history['loss'], label=f'{model_name} - Train')
                ax2.plot(history['val_loss'], label=f'{model_name} - Val', linestyle='--')
        
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage and experimentation
def run_model_experiments(X_train, y_train, X_val, y_val, X_test, y_test, 
                         feature_type='mfcc'):
    """
    Run experiments with different model architectures
    """
    print("Starting Model Experiments")
    print(f"Feature type: {feature_type}")
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Initialize model collection
    models = AudioDigitModels(num_classes=10)
    
    # Define which models to test based on feature type
    if feature_type == 'spectrogram':
        model_list = ['2d_cnn', 'random_forest']
    elif feature_type == 'raw_audio':
        model_list = ['1d_cnn', 'random_forest']
    elif feature_type == 'mfcc':
        model_list = ['transformer', 'random_forest', 'svm']
    else:
        model_list = ['random_forest', 'svm']
    
    # Train models
    training_results = {}
    
    for model_name in model_list:
        try:
            result = models.train_model(
                model_name, X_train, y_train, X_val, y_val, 
                epochs=30 if model_name in ['2d_cnn', '1d_cnn', 'transformer'] else None
            )
            training_results[model_name] = result
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # Evaluate models
    test_results = {}
    for model_name in model_list:
        if model_name in models.models:
            try:
                result = models.evaluate_model(model_name, X_test, y_test)
                test_results[model_name] = result
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
    
    # Compare results
    models.compare_models(test_results)
    
    # Plot training histories for neural networks
    nn_histories = {name: result for name, result in training_results.items() 
                   if 'history' in result}
    if nn_histories:
        models.plot_training_history(nn_histories)
    
    return models, training_results, test_results

# Example with synthetic data
if __name__ == "__main__":
    # Create synthetic data for demonstration
    n_samples = 1000
    
    # MFCC-like features: (batch, time, features)
    X_mfcc = np.random.randn(n_samples, 130, 13)
    
    # Raw audio-like features: (batch, time_samples)
    X_raw = np.random.randn(n_samples, 66150)
    
    # Spectrogram-like features: (batch, freq, time)
    X_spec = np.random.randn(n_samples, 128, 130)
    
    # Labels (digits 0-9)
    y = np.random.randint(0, 10, n_samples)
    
    # Split data
    split1, split2 = int(0.7 * n_samples), int(0.85 * n_samples)
    
    # Run experiment with MFCC features
    print("Running MFCC experiment...")
    models, train_results, test_results = run_model_experiments(
        X_mfcc[:split1], y[:split1],  # Train
        X_mfcc[split1:split2], y[split1:split2],  # Validation
        X_mfcc[split2:], y[split2:],  # Test
        feature_type='mfcc'
    )