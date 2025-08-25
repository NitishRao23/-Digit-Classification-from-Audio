import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import seaborn as sns

class AudioFeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=13, n_mels=128, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_raw_waveform(self, audio: np.ndarray) -> np.ndarray:
        """Raw audio waveform features"""
        return audio.reshape(1, -1)  # Shape: (1, samples)
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Mel-frequency cepstral coefficients"""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return mfcc  # Shape: (n_mfcc, time_frames)
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Mel-scale spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return mel_spec  # Shape: (n_mels, time_frames)
    
    def extract_log_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Log-magnitude spectrogram"""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        log_spec = librosa.amplitude_to_db(np.abs(stft))
        return log_spec  # Shape: (freq_bins, time_frames)
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all feature types"""
        return {
            'raw_waveform': self.extract_raw_waveform(audio),
            'mfcc': self.extract_mfcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'log_spectrogram': self.extract_log_spectrogram(audio)
        }
    
    def visualize_features(self, audio: np.ndarray, title="Audio Features"):
        """Compare all feature representations visually"""
        features = self.extract_all_features(audio)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Raw waveform
        time = np.linspace(0, len(audio)/self.sr, len(audio))
        axes[0,0].plot(time, audio)
        axes[0,0].set_title('Raw Waveform')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Amplitude')
        
        # MFCCs
        librosa.display.specshow(features['mfcc'], sr=self.sr, 
                                hop_length=self.hop_length, x_axis='time', ax=axes[0,1])
        axes[0,1].set_title('MFCCs')
        axes[0,1].set_ylabel('MFCC Coefficients')
        
        # Mel Spectrogram
        librosa.display.specshow(librosa.power_to_db(features['mel_spectrogram']),
                                sr=self.sr, hop_length=self.hop_length,
                                x_axis='time', y_axis='mel', ax=axes[1,0])
        axes[1,0].set_title('Mel Spectrogram')
        
        # Log Spectrogram
        librosa.display.specshow(features['log_spectrogram'], sr=self.sr,
                                hop_length=self.hop_length, x_axis='time', 
                                y_axis='hz', ax=axes[1,1])
        axes[1,1].set_title('Log Spectrogram')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_stats(self, features: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Get statistics for each feature type"""
        stats = {}
        for name, feature in features.items():
            stats[name] = {
                'shape': feature.shape,
                'mean': np.mean(feature),
                'std': np.std(feature),
                'min': np.min(feature),
                'max': np.max(feature)
            }
        return stats
    
    def flatten_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Flatten 2D features for ML models"""
        flattened = {}
        for name, feature in features.items():
            if feature.ndim > 1:
                flattened[name] = feature.flatten()
            else:
                flattened[name] = feature
        return flattened
    
    def extract_statistical_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from raw audio"""
        return {
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(audio, sr=self.sr))),
            'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(audio, sr=self.sr))),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(audio, sr=self.sr))),
            'rms_energy': float(np.mean(librosa.feature.rms(y=audio))),
            'tempo': float(librosa.beat.tempo(y=audio, sr=self.sr)[0])
        }
    
    def batch_extract_features(self, audio_list: List[np.ndarray], 
                             feature_type='all') -> List[Dict[str, np.ndarray]]:
        """Extract features from multiple audio samples"""
        results = []
        
        for i, audio in enumerate(audio_list):
            if feature_type == 'all':
                features = self.extract_all_features(audio)
                # Add statistical features
                stats = self.extract_statistical_features(audio)
                features.update(stats)
            elif feature_type == 'mfcc':
                features = {'mfcc': self.extract_mfcc(audio)}
            elif feature_type == 'mel':
                features = {'mel_spectrogram': self.extract_mel_spectrogram(audio)}
            elif feature_type == 'log':
                features = {'log_spectrogram': self.extract_log_spectrogram(audio)}
            else:
                features = {'raw_waveform': self.extract_raw_waveform(audio)}
            
            results.append(features)
            
        print(f"Extracted features from {len(results)} audio samples")
        return results

# Usage functions
def compare_feature_sizes(extractor: AudioFeatureExtractor, audio: np.ndarray):
    """Compare feature representation sizes"""
    features = extractor.extract_all_features(audio)
    
    print("Feature Comparison:")
    print("-" * 40)
    for name, feature in features.items():
        size_kb = feature.nbytes / 1024
        print(f"{name:15} | Shape: {str(feature.shape):15} | Size: {size_kb:.2f} KB")
    
    # Original audio size
    orig_size = audio.nbytes / 1024
    print(f"{'Original':15} | Shape: {str(audio.shape):15} | Size: {orig_size:.2f} KB")

def create_feature_comparison_plot(features_dict: Dict[str, Dict[str, np.ndarray]], 
                                 labels: List[str]):
    """Plot features from multiple audio samples for comparison"""
    n_samples = len(features_dict)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    feature_types = ['raw_waveform', 'mfcc', 'mel_spectrogram', 'log_spectrogram']
    titles = ['Waveform', 'MFCC', 'Mel Spectrogram', 'Log Spectrogram']
    
    for i, (label, features) in enumerate(features_dict.items()):
        for j, (feat_type, title) in enumerate(zip(feature_types, titles)):
            ax = axes[i, j]
            
            if feat_type == 'raw_waveform':
                ax.plot(features[feat_type].flatten())
                ax.set_ylabel('Amplitude')
            else:
                im = ax.imshow(features[feat_type], aspect='auto', origin='lower')
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            ax.set_title(f'{title} - {label}')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13, n_mels=128)
    
    # Create test audio signals
    sr = 22050
    duration = 3
    t = np.linspace(0, duration, sr * duration)
    
    # Different test signals
    sine_wave = np.sin(2 * np.pi * 440 * t)  # Pure tone
    chirp = np.sin(2 * np.pi * t * (200 + 300 * t))  # Frequency sweep
    noise = np.random.normal(0, 0.1, len(t))  # White noise
    
    print("Audio Feature Extraction Pipeline")
    print("=" * 40)
    
    # Test with sine wave
    print("\nExtracting features from sine wave...")
    features = extractor.extract_all_features(sine_wave)
    stats = extractor.get_feature_stats(features)
    
    print("\nFeature Statistics:")
    for name, stat in stats.items():
        print(f"{name}: {stat['shape']} | Mean: {stat['mean']:.4f}")
    
    # Visualize features
    extractor.visualize_features(sine_wave, "Sine Wave Features")
    
    # Compare feature sizes
    compare_feature_sizes(extractor, sine_wave)
    
    # Batch processing
    audio_samples = [sine_wave, chirp, noise]
    batch_features = extractor.batch_extract_features(audio_samples, 'mfcc')
    
    print(f"\nBatch processing complete: {len(batch_features)} samples processed")