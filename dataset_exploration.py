import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
from datasets import load_dataset
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_fsdd_dataset():
    """Load the Free Spoken Digit Dataset from Hugging Face"""
    print("Loading Free Spoken Digit Dataset...")
    try:
        # Load the dataset
        dataset = load_dataset("mteb/free-spoken-digit-dataset")
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset_structure(dataset):
    """Explore the basic structure of the dataset"""
    print("\n" + "="*50)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*50)
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Number of samples: {len(split_data)}")
        print(f"  Features: {list(split_data.features.keys())}")
        
        # Show first few examples
        print(f"  Sample data structure:")
        for key, value in split_data[0].items():
            if key == 'audio':
                print(f"    {key}: array shape {np.array(value['array']).shape}, sample_rate: {value['sampling_rate']}")
            else:
                print(f"    {key}: {value} (type: {type(value)})")

def analyze_audio_properties(dataset):
    """Analyze audio file properties across the dataset"""
    print("\n" + "="*50)
    print("AUDIO PROPERTIES ANALYSIS")
    print("="*50)
    
    # We'll analyze the training split (or first available split)
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    durations = []
    sample_rates = []
    audio_lengths = []
    
    print("Analyzing audio properties...")
    for i, sample in enumerate(data):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(data)} samples...")
            
        audio_data = sample['audio']
        audio_array = np.array(audio_data['array'])
        sample_rate = audio_data['sampling_rate']
        
        duration = len(audio_array) / sample_rate
        durations.append(duration)
        sample_rates.append(sample_rate)
        audio_lengths.append(len(audio_array))
    
    # Create summary statistics
    audio_stats = {
        'Total samples': len(data),
        'Duration (seconds)': {
            'min': np.min(durations),
            'max': np.max(durations),
            'mean': np.mean(durations),
            'std': np.std(durations),
            'median': np.median(durations)
        },
        'Sample rates': {
            'unique_rates': list(set(sample_rates)),
            'most_common': Counter(sample_rates).most_common(1)[0]
        },
        'Audio length (samples)': {
            'min': np.min(audio_lengths),
            'max': np.max(audio_lengths),
            'mean': np.mean(audio_lengths),
            'std': np.std(audio_lengths),
            'median': np.median(audio_lengths)
        }
    }
    
    return audio_stats, durations, sample_rates, audio_lengths

def analyze_label_distribution(dataset):
    """Analyze the distribution of digits and speakers"""
    print("\n" + "="*50)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*50)
    
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    # Extract labels and speaker info if available
    labels = []
    speakers = []
    
    for sample in data:
        labels.append(sample['label'])
        # Check if speaker info is available in filename or other field
        if 'file' in sample:
            # FSDD filenames typically follow pattern: {digit}_{speaker}_{repetition}.wav
            filename = sample['file']
            parts = filename.split('_')
            if len(parts) >= 2:
                speakers.append(parts[1])
        elif 'speaker_id' in sample:
            speakers.append(sample['speaker_id'])
    
    # Analyze digit distribution
    digit_counts = Counter(labels)
    print(f"Digit distribution:")
    for digit in sorted(digit_counts.keys()):
        print(f"  Digit {digit}: {digit_counts[digit]} samples")
    
    # Analyze speaker distribution if available
    if speakers:
        speaker_counts = Counter(speakers)
        print(f"\nSpeaker distribution:")
        print(f"  Number of unique speakers: {len(speaker_counts)}")
        print(f"  Samples per speaker range: {min(speaker_counts.values())} - {max(speaker_counts.values())}")
        most_common_speakers = speaker_counts.most_common(5)
        print(f"  Top 5 speakers: {most_common_speakers}")
    
    return digit_counts, speaker_counts if speakers else None

def create_visualizations(durations, sample_rates, digit_counts, audio_stats):
    """Create visualizations for dataset analysis"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Duration distribution
    axes[0, 0].hist(durations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Audio Durations')
    axes[0, 0].set_xlabel('Duration (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.2f}s')
    axes[0, 0].legend()
    
    # Sample rate distribution
    unique_rates = list(set(sample_rates))
    rate_counts = [sample_rates.count(rate) for rate in unique_rates]
    axes[0, 1].bar(unique_rates, rate_counts, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Sample Rate Distribution')
    axes[0, 1].set_xlabel('Sample Rate (Hz)')
    axes[0, 1].set_ylabel('Count')
    
    # Digit distribution
    digits = sorted(digit_counts.keys())
    counts = [digit_counts[digit] for digit in digits]
    axes[1, 0].bar(digits, counts, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Distribution of Digits')
    axes[1, 0].set_xlabel('Digit')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks(digits)
    
    # Box plot of durations by digit (if we can extract digit from samples)
    # For now, just show overall duration statistics
    axes[1, 1].boxplot([durations], labels=['All Digits'])
    axes[1, 1].set_title('Audio Duration Statistics')
    axes[1, 1].set_ylabel('Duration (seconds)')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def sample_audio_analysis(dataset, num_samples=5):
    """Analyze and visualize sample audio files"""
    print("\n" + "="*50)
    print("SAMPLE AUDIO ANALYSIS")
    print("="*50)
    
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    # Select random samples from different digits
    sample_indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_indices):
        sample = data[idx]
        audio_data = np.array(sample['audio']['array'])
        sample_rate = sample['audio']['sampling_rate']
        label = sample['label']
        
        # Time domain plot
        time = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
        axes[i, 0].plot(time, audio_data)
        axes[i, 0].set_title(f'Digit {label} - Time Domain (Sample {idx})')
        axes[i, 0].set_xlabel('Time (seconds)')
        axes[i, 0].set_ylabel('Amplitude')
        
        # Frequency domain plot (spectrogram)
        D = librosa.stft(audio_data.astype(float))
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=axes[i, 1])
        axes[i, 1].set_title(f'Digit {label} - Spectrogram')
        
    plt.tight_layout()
    plt.savefig('sample_audio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_sample_audio_files(dataset, output_dir='data/samples', num_per_digit=2):
    """Save sample audio files for manual listening"""
    print("\n" + "="*50)
    print("SAVING SAMPLE AUDIO FILES")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    # Group samples by digit
    digit_samples = {}
    for i, sample in enumerate(data):
        digit = sample['label']
        if digit not in digit_samples:
            digit_samples[digit] = []
        digit_samples[digit].append((i, sample))
    
    # Save samples for each digit
    for digit in sorted(digit_samples.keys()):
        samples = digit_samples[digit][:num_per_digit]  # Take first N samples
        
        for j, (idx, sample) in enumerate(samples):
            audio_data = np.array(sample['audio']['array'])
            sample_rate = sample['audio']['sampling_rate']
            
            filename = f"digit_{digit}_sample_{j+1}_idx_{idx}.wav"
            filepath = os.path.join(output_dir, filename)
            
            sf.write(filepath, audio_data, sample_rate)
            print(f"  Saved: {filename}")

def main():
    """Main execution function"""
    print("Starting FSDD Dataset Exploration...")
    
    # Load dataset
    dataset = load_fsdd_dataset()
    if dataset is None:
        return
    
    # Explore dataset structure
    explore_dataset_structure(dataset)
    
    # Analyze audio properties
    audio_stats, durations, sample_rates, audio_lengths = analyze_audio_properties(dataset)
    
    # Print audio statistics
    print("\nAudio Statistics Summary:")
    for key, value in audio_stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    # Analyze label distribution
    digit_counts, speaker_counts = analyze_label_distribution(dataset)
    
    # Create visualizations
    create_visualizations(durations, sample_rates, digit_counts, audio_stats)
    
    # Analyze sample audio files
    sample_audio_analysis(dataset)
    
    # Save sample files for manual listening
    save_sample_audio_files(dataset)
    
    print("\n" + "="*50)
    print("EXPLORATION COMPLETE!")
    print("="*50)
    print("Generated files:")
    print("  - dataset_analysis.png")
    print("  - sample_audio_analysis.png")
    print("  - data/samples/ (audio files for listening)")

if __name__ == "__main__":
    main()