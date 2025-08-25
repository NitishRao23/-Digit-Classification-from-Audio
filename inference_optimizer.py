import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import time
from collections import deque
import onnx
import onnxruntime as ort
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class ModelOptimizer:
    """Model optimization for faster inference"""

    def __init__(self, model, device='cpu'):
        self.original_model = model
        self.device = device
        self.optimized_models = {}

    def optimize_for_inference(self, model):
        """Basic inference optimization"""
        model.eval()

        # Fuse modules for faster inference
        if hasattr(torch.quantization, 'fuse_modules'):
            try:
                modules_to_fuse = []

                # Common fusion patterns for nn.Sequential
                for name, module in model.named_children():
                    if isinstance(module, nn.Sequential):
                        fuse_list = []
                        # Check for patterns Conv/Linear + ReLU to fuse
                        for i in range(len(module) - 1):
                            if (isinstance(module[i], (nn.Linear, nn.Conv1d)) and
                                    isinstance(module[i + 1], (nn.ReLU, nn.ReLU6))):
                                fuse_list.append([f"{name}.{i}", f"{name}.{i + 1}"])
                        modules_to_fuse.extend(fuse_list)

                if modules_to_fuse:
                    torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
            except Exception as e:
                print(f"Warning: fusion failed with error: {e}")

        return model

    def quantize_model(self, model, method='dynamic'):
        """Apply quantization for faster inference"""
        model.eval()

        if method == 'dynamic':
            # Dynamic quantization - fastest to apply
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d},
                dtype=torch.qint8
            )

        elif method == 'static':
            # Static quantization - requires calibration data
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)

            # Calibration with dummy input, ensure shape matches model input
            dummy_input = torch.randn(1, 20)  # Adjust if needed
            with torch.no_grad():
                model(dummy_input)

            quantized_model = torch.quantization.convert(model, inplace=False)

        else:
            quantized_model = model

        return quantized_model

    def export_to_onnx(self, model, input_shape, onnx_path="model.onnx"):
        """Export model to ONNX for optimized inference"""
        model.eval()
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        print(f"Model exported to ONNX: {onnx_path}")
        return onnx_path

    def create_optimized_versions(self, input_shape=(1, 20)):
        """Create multiple optimized versions of the model"""
        print("Creating optimized model versions...")

        # Original model
        self.optimized_models['original'] = self.original_model

        # Inference optimized
        try:
            model_copy = type(self.original_model)()  # Create new instance
            model_copy.load_state_dict(self.original_model.state_dict())
            opt_model = self.optimize_for_inference(model_copy)
            self.optimized_models['optimized'] = opt_model
        except Exception as e:
            print(f"Optimization failed: {e}")
            self.optimized_models['optimized'] = self.original_model

        # Dynamically quantized
        try:
            quant_model = self.quantize_model(self.original_model, method='dynamic')
            self.optimized_models['quantized'] = quant_model
        except Exception as e:
            print(f"Quantization failed: {e}")
            self.optimized_models['quantized'] = self.original_model

        # ONNX version
        try:
            onnx_path = self.export_to_onnx(self.original_model, input_shape)
            ort_session = ort.InferenceSession(onnx_path)
            self.optimized_models['onnx'] = ort_session
        except Exception as e:
            print(f"ONNX export failed: {e}")
            self.optimized_models['onnx'] = None

        return self.optimized_models

    def benchmark_models(self, input_tensor, num_runs=100):
        """Benchmark different model versions"""
        print(f"\nBenchmarking models with {num_runs} runs...")

        results = {}

        for name, model in self.optimized_models.items():
            if model is None:
                continue

            times = []

            for _ in range(num_runs):
                start_time = time.time()

                if name == 'onnx' and isinstance(model, ort.InferenceSession):
                    # ONNX inference
                    input_dict = {'input': input_tensor.numpy()}
                    model.run(None, input_dict)
                else:
                    # PyTorch inference
                    with torch.no_grad():
                        model(input_tensor)

                times.append(time.time() - start_time)

            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000

            results[name] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'throughput_samples_per_sec': 1000 / avg_time if avg_time > 0 else float('inf')
            }

            print(f"{name:>12}: {avg_time:.3f} ± {std_time:.3f} ms")

        return results


class EfficientPredictor:
    """Efficient prediction pipeline with edge case handling"""

    def __init__(self, model, class_names=None, confidence_threshold=0.7):
        self.model = model
        self.class_names = class_names or [f"Digit {i}" for i in range(10)]
        self.confidence_threshold = confidence_threshold

        # Edge case detection parameters
        self.silence_threshold = 0.01
        self.noise_threshold = 0.8
        self.multi_digit_threshold = 0.5

        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        self.confidence_buffer = deque(maxlen=5)

    def preprocess_input(self, audio_data):
        """Efficient input preprocessing"""
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)

        # Quick silence detection
        energy = np.mean(np.square(audio_data))
        if energy < self.silence_threshold:
            return None, "silence"

        # Normalize efficiently
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # Simple noise detection (high variance)
        if np.var(audio_data) > self.noise_threshold:
            return None, "noise"

        # Convert to tensor
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.FloatTensor(audio_data)

        # Ensure correct shape
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)

        return audio_data, "valid"

    def detect_multiple_digits(self, logits, confidences):
        """Detect if multiple digits are present"""
        confident_predictions = confidences > self.multi_digit_threshold
        num_confident = torch.sum(confident_predictions).item()

        if num_confident > 1:
            sorted_indices = torch.argsort(confidences, descending=True)
            top_predictions = []

            for idx in sorted_indices[:num_confident]:
                if confidences[idx] > self.multi_digit_threshold:
                    top_predictions.append({
                        'digit': self.class_names[idx],
                        'confidence': confidences[idx].item()
                    })

            return True, top_predictions

        return False, None

    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions using recent history"""
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)

        if len(self.prediction_buffer) < 3:
            return prediction, confidence

        pred_counts = {}
        for pred in self.prediction_buffer:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        smoothed_prediction = max(pred_counts, key=pred_counts.get)

        relevant_confidences = [
            conf for pred, conf in zip(self.prediction_buffer, self.confidence_buffer)
            if pred == smoothed_prediction
        ]
        smoothed_confidence = np.mean(relevant_confidences)

        return smoothed_prediction, smoothed_confidence

    def predict_with_edge_cases(self, audio_input, use_smoothing=True):
        """Make prediction with comprehensive edge case handling"""
        processed_input, status = self.preprocess_input(audio_input)

        if status == "silence":
            return {
                'prediction': 'SILENCE',
                'confidence': 1.0,
                'status': 'silence_detected',
                'message': 'No speech detected'
            }

        if status == "noise":
            return {
                'prediction': 'NOISE',
                'confidence': 0.0,
                'status': 'noise_detected',
                'message': 'Too much background noise'
            }

        self.model.eval()
        with torch.no_grad():
            logits = self.model(processed_input)
            probabilities = torch.softmax(logits, dim=1)
            confidences = probabilities.squeeze()
            predicted_class = torch.argmax(logits, dim=1).item()
            max_confidence = torch.max(confidences).item()

        is_multiple, multiple_results = self.detect_multiple_digits(logits.squeeze(), confidences)

        if is_multiple:
            return {
                'prediction': 'MULTIPLE_DIGITS',
                'confidence': max_confidence,
                'status': 'multiple_digits',
                'digits': multiple_results,
                'message': f'Multiple digits detected: {[d["digit"] for d in multiple_results]}'
            }

        prediction = self.class_names[predicted_class]

        if use_smoothing:
            prediction, max_confidence = self.smooth_predictions(prediction, max_confidence)

        if max_confidence < self.confidence_threshold:
            return {
                'prediction': 'UNCERTAIN',
                'confidence': max_confidence,
                'status': 'low_confidence',
                'best_guess': prediction,
                'message': f'Low confidence prediction: {prediction} ({max_confidence:.2f})'
            }

        return {
            'prediction': prediction,
            'confidence': max_confidence,
            'status': 'success',
            'message': f'Predicted: {prediction} (confidence: {max_confidence:.2f})'
        }

    def batch_predict(self, audio_batch):
        """Efficient batch prediction"""
        results = [None] * len(audio_batch)

        valid_inputs = []
        valid_indices = []

        for i, audio in enumerate(audio_batch):
            processed, status = self.preprocess_input(audio)
            if status == "valid":
                valid_inputs.append(processed)
                valid_indices.append(i)
            else:
                results[i] = self.predict_with_edge_cases(audio, use_smoothing=False)

        if valid_inputs:
            # Pad tensors if shapes vary
            max_len = max(inp.shape[1] for inp in valid_inputs)
            padded_inputs = []
            for inp in valid_inputs:
                if inp.shape[1] < max_len:
                    pad_size = max_len - inp.shape[1]
                    padded = torch.nn.functional.pad(inp, (0, pad_size))
                    padded_inputs.append(padded)
                else:
                    padded_inputs.append(inp)

            batch_tensor = torch.cat(padded_inputs, dim=0)

            with torch.no_grad():
                batch_logits = self.model(batch_tensor)
                batch_probs = torch.softmax(batch_logits, dim=1)
                batch_preds = torch.argmax(batch_logits, dim=1)
                batch_confs = torch.max(batch_probs, dim=1)[0]

            for idx, batch_idx in enumerate(valid_indices):
                pred_idx = batch_preds[idx].item()
                confidence = batch_confs[idx].item()
                status = 'success' if confidence >= self.confidence_threshold else 'low_confidence'

                results[batch_idx] = {
                    'prediction': self.class_names[pred_idx],
                    'confidence': confidence,
                    'status': status
                }

        return results


class InferenceOptimizer:
    """Complete inference optimization system"""

    def __init__(self, model, input_shape=(1, 20)):
        self.model = model
        self.input_shape = input_shape
        self.optimizer = ModelOptimizer(model)
        self.predictor = None
        self.optimized_models = None

    def setup_optimization(self, class_names=None):
        """Setup complete optimization pipeline"""
        print("Setting up inference optimization...")

        self.optimized_models = self.optimizer.create_optimized_versions(self.input_shape)

        best_model = self.optimized_models.get('quantized', self.model)

        self.predictor = EfficientPredictor(best_model, class_names)

        print("Optimization setup complete!")
        return self.predictor

    def benchmark_optimization(self, num_samples=100):
        """Benchmark optimization improvements"""
        if not self.optimized_models:
            self.setup_optimization()

        test_input = torch.randn(self.input_shape)

        results = self.optimizer.benchmark_models(test_input, num_samples)

        if 'original' in results and 'quantized' in results:
            original_time = results['original']['avg_time_ms']
            optimized_time = results['quantized']['avg_time_ms']
            speedup = original_time / optimized_time if optimized_time > 0 else float('inf')

            print(f"\n📈 Optimization Results:")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Original: {original_time:.3f} ms")
            print(f"   Optimized: {optimized_time:.3f} ms")

        return results


# Example usage and demo
def demo_optimization():
    """Demonstration of inference optimization"""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)  # 10 digits
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleModel()
    optimizer = InferenceOptimizer(model)

    predictor = optimizer.setup_optimization()

    results = optimizer.benchmark_optimization()

    print("\n🧪 Testing Edge Cases:")

    test_cases = {
        'normal_audio': np.random.randn(20) * 0.5,
        'silence': np.zeros(20),
        'noise': np.random.randn(20) * 2.0,
        'low_energy': np.random.randn(20) * 0.005
    }

    for name, audio in test_cases.items():
        result = predictor.predict_with_edge_cases(audio)
        print(f"   {name}: {result['status']} - {result['message']}")

    print("\n📦 Testing Batch Prediction:")
    audio_batch = [np.random.randn(20) * 0.5 for _ in range(5)]
    batch_results = predictor.batch_predict(audio_batch)
    for i, res in enumerate(batch_results):
        print(f"  Sample {i}: {res}")

if __name__ == "__main__":
    demo_optimization()
