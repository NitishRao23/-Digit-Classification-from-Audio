import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple
import random

class AudioPreprocessor:
    def __init__(self, sr=22050, duration=3.0):
        self.sr = sr
        self.target_length = int(sr * duration)
    
    def load_and_resample(self, file_path: str) -> np.ndarray:
        """Load and resample audio to target sample rate"""
        audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
        return audio
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        return audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
    
    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Pad with zeros or truncate to target length"""
        if len(audio) < self.target_length:
            return np.pad(audio, (0, self.target_length - len(audio)))
        return audio[:self.target_length]
    
    def add_noise(self, audio: np.ndarray, factor=0.005) -> np.ndarray:
        """Add white noise"""
        return audio + np.random.normal(0, factor, audio.shape)
    
    def time_stretch(self, audio: np.ndarray, rate=1.0) -> np.ndarray:
        """Change speed without changing pitch"""
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return self.pad_or_truncate(stretched)
    
    def pitch_shift(self, audio: np.ndarray, steps=0) -> np.ndarray:
        """Shift pitch by semitones"""
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)
    
    def augment(self, audio: np.ndarray) -> List[np.ndarray]:
        """Apply augmentations and return list of variants"""
        variants = [audio]  # Original
        
        # Noise variants
        variants.extend([self.add_noise(audio, f) for f in [0.002, 0.005]])
        
        # Speed variants
        for rate in [0.9, 1.1]:
            variants.append(self.time_stretch(audio, rate))
        
        # Pitch variants
        for steps in [-1, 1]:
            variants.append(self.pitch_shift(audio, steps))
        
        return variants
    
    def preprocess(self, file_path: str, augment=False) -> List[np.ndarray]:
        """Complete preprocessing pipeline"""
        # Load and resample
        audio = self.load_and_resample(file_path)
        
        # Normalize
        audio = self.normalize(audio)
        
        # Fix length
        audio = self.pad_or_truncate(audio)
        
        # Apply augmentation if requested
        if augment:
            return self.augment(audio)
        
        return [audio]
    
    def batch_process(self, files: List[str], augment=False) -> List[np.ndarray]:
        """Process multiple files"""
        results = []
        for file in files:
            results.extend(self.preprocess(file, augment))
        return results

# Usage example
if __name__ == "__main__":
    processor = AudioPreprocessor(sr=22050, duration=3.0)
    
    # Create test audio
    t = np.linspace(0, 2, 44100)
    test_audio = np.sin(2 * np.pi * 440 * t)
    sf.write("test.wav", test_audio, 22050)
    
    # Process single file
    processed = processor.preprocess("test.wav", augment=True)
    print(f"Generated {len(processed)} samples")
    
    # Clean up
    import os
    os.remove("test.wav")