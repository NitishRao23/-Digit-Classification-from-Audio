import pyaudio
import numpy as np
import threading
import queue
import time
from collections import deque
import matplotlib.pyplot as plt
from scipy import signal
import librosa

class AudioProcessor:
    """Real-time audio processing with voice activity detection"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paFloat32
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=int(sample_rate * 5))  # 5 seconds buffer
        self.processed_buffer = queue.Queue()
        
        # VAD parameters
        self.vad_threshold = 0.02
        self.silence_duration = 0.5  # seconds
        self.min_speech_duration = 0.2  # seconds
        
        # State tracking
        self.is_recording = False
        self.is_speaking = False
        self.silence_start = None
        self.speech_start = None
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def start_recording(self):
        """Start real-time audio capture"""
        print("Starting audio capture...")
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.is_recording = True
        self.stream.start_stream()
        print(f"Recording started - Sample rate: {self.sample_rate} Hz")
        
    def stop_recording(self):
        """Stop audio capture"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.is_recording = False
        print("Recording stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Real-time audio callback function"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to circular buffer
        self.audio_buffer.extend(audio_data)
        
        # Process audio chunk
        processed_chunk = self.preprocess_audio(audio_data)
        
        # Voice activity detection
        is_voice = self.detect_voice_activity(processed_chunk)
        
        # Handle speech/silence state changes
        self._update_speech_state(is_voice, processed_chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def preprocess_audio(self, audio_chunk):
        """Preprocess audio chunk for better quality"""
        # Normalize audio
        audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-8)
        
        # Apply high-pass filter to remove low-frequency noise
        if len(audio_chunk) > 10:
            sos = signal.butter(5, 80, btype='high', fs=self.sample_rate, output='sos')
            audio_chunk = signal.sosfilt(sos, audio_chunk)
        
        # Simple noise gate
        audio_chunk[np.abs(audio_chunk) < 0.005] = 0
        
        return audio_chunk
    
    def detect_voice_activity(self, audio_chunk):
        """Simple voice activity detection based on energy and spectral features"""
        # Energy-based detection
        energy = np.mean(np.square(audio_chunk))
        
        # Spectral features
        if len(audio_chunk) >= 512:
            # Zero crossing rate
            zcr = np.mean(np.abs(np.diff(np.sign(audio_chunk))))
            
            # Spectral centroid (brightness)
            fft = np.fft.rfft(audio_chunk)
            freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sample_rate)
            magnitude = np.abs(fft)
            
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0
            
            # Voice activity criteria
            is_voice = (
                energy > self.vad_threshold and
                zcr < 0.3 and  # Not too noisy
                200 < spectral_centroid < 4000  # Human voice frequency range
            )
        else:
            is_voice = energy > self.vad_threshold
        
        return is_voice
    
    def _update_speech_state(self, is_voice, audio_chunk):
        """Update speech state and handle transitions"""
        current_time = time.time()
        
        if is_voice:
            if not self.is_speaking:
                # Start of speech
                if self.speech_start is None:
                    self.speech_start = current_time
                elif current_time - self.speech_start >= self.min_speech_duration:
                    self.is_speaking = True
                    self.silence_start = None
                    print("🎤 Speech detected")
            else:
                # Continue speech
                self.silence_start = None
        else:
            if self.is_speaking:
                # Potential end of speech
                if self.silence_start is None:
                    self.silence_start = current_time
                elif current_time - self.silence_start >= self.silence_duration:
                    # End of speech confirmed
                    self.is_speaking = False
                    self.speech_start = None
                    print("🔇 Speech ended")
                    
                    # Extract speech segment
                    self._extract_speech_segment()
            else:
                # Reset speech start if not enough duration
                self.speech_start = None
    
    def _extract_speech_segment(self):
        """Extract and save the speech segment"""
        if len(self.audio_buffer) > 0:
            # Convert buffer to numpy array
            speech_audio = np.array(list(self.audio_buffer))
            
            # Add to processed buffer
            self.processed_buffer.put({
                'audio': speech_audio,
                'timestamp': time.time(),
                'duration': len(speech_audio) / self.sample_rate
            })
    
    def get_latest_speech(self):
        """Get the latest processed speech segment"""
        try:
            return self.processed_buffer.get_nowait()
        except queue.Empty:
            return None
    
    def get_live_audio_level(self):
        """Get current audio level for visualization"""
        if len(self.audio_buffer) > 0:
            recent_audio = list(self.audio_buffer)[-self.chunk_size:]
            return np.sqrt(np.mean(np.square(recent_audio)))
        return 0
    
    def visualize_audio(self, duration=10):
        """Real-time audio visualization"""
        print("Starting audio visualization (press Ctrl+C to stop)...")
        
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        # Setup plots
        x_time = np.linspace(0, duration, int(duration * self.sample_rate))
        line1, = ax1.plot(x_time, np.zeros(len(x_time)))
        ax1.set_ylim(-1, 1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Live Audio Waveform')
        ax1.grid(True)
        
        # Frequency domain
        freqs = np.fft.rfftfreq(2048, 1/self.sample_rate)
        line2, = ax2.plot(freqs, np.zeros(len(freqs)))
        ax2.set_xlim(0, 4000)
        ax2.set_ylim(0, 0.1)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Live Audio Spectrum')
        ax2.grid(True)
        
        try:
            while self.is_recording:
                if len(self.audio_buffer) >= len(x_time):
                    # Time domain
                    audio_data = np.array(list(self.audio_buffer)[-len(x_time):])
                    line1.set_ydata(audio_data)
                    
                    # Frequency domain
                    if len(audio_data) >= 2048:
                        fft_data = np.fft.rfft(audio_data[-2048:])
                        magnitude = np.abs(fft_data) / len(fft_data)
                        line2.set_ydata(magnitude)
                        
                        # Color based on voice activity
                        recent_chunk = audio_data[-self.chunk_size:]
                        is_voice = self.detect_voice_activity(recent_chunk)
                        color = 'red' if is_voice else 'blue'
                        line1.set_color(color)
                
                # Update speech status in title
                status = "🎤 SPEAKING" if self.is_speaking else "🔇 SILENT"
                level = self.get_live_audio_level()
                ax1.set_title(f'Live Audio Waveform - {status} (Level: {level:.3f})')
                
                plt.pause(0.05)
                
        except KeyboardInterrupt:
            print("Visualization stopped")
        
        plt.ioff()
        plt.close()
    
    def save_audio_segment(self, filename, audio_data=None):
        """Save audio segment to file"""
        if audio_data is None:
            if len(self.audio_buffer) == 0:
                print("No audio data to save")
                return
            audio_data = np.array(list(self.audio_buffer))
        
        # Save as WAV file using scipy
        from scipy.io import wavfile
        
        # Convert to int16 for WAV format
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_int16)
        print(f"Audio saved to: {filename}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_recording()
        self.audio.terminate()

class AudioStreamManager:
    """High-level manager for audio streaming operations"""
    
    def __init__(self, sample_rate=16000):
        self.processor = AudioProcessor(sample_rate=sample_rate)
        self.is_running = False
    
    def start_stream(self, with_visualization=False):
        """Start audio streaming"""
        self.processor.start_recording()
        self.is_running = True
        
        if with_visualization:
            # Run visualization in separate thread
            viz_thread = threading.Thread(target=self.processor.visualize_audio, args=(10,))
            viz_thread.daemon = True
            viz_thread.start()
        
        print("Audio stream started. Voice activity detection enabled.")
        print("Speak into the microphone...")
        
        return self.processor
    
    def monitor_speech(self, callback=None):
        """Monitor for speech segments with optional callback"""
        try:
            while self.is_running:
                speech_segment = self.processor.get_latest_speech()
                
                if speech_segment:
                    print(f"📝 Speech captured: {speech_segment['duration']:.2f}s")
                    
                    if callback:
                        callback(speech_segment)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Monitoring stopped")
            self.stop_stream()
    
    def stop_stream(self):
        """Stop audio streaming"""
        self.is_running = False
        self.processor.cleanup()
        print("Audio stream stopped")

# Example usage functions
def simple_recording_demo():
    """Simple recording demonstration"""
    print("=== Simple Recording Demo ===")
    
    stream_manager = AudioStreamManager(sample_rate=16000)
    processor = stream_manager.start_stream(with_visualization=True)
    
    try:
        # Record for 10 seconds
        print("Recording for 10 seconds...")
        time.sleep(10)
        
        # Save the recording
        processor.save_audio_segment("recording.wav")
        
    finally:
        stream_manager.stop_stream()

def speech_detection_demo():
    """Speech detection with callback demo"""
    print("=== Speech Detection Demo ===")
    
    def on_speech_detected(speech_segment):
        """Callback function for when speech is detected"""
        duration = speech_segment['duration']
        timestamp = speech_segment['timestamp']
        
        print(f"🎯 Speech segment detected!")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Time: {time.ctime(timestamp)}")
        
        # Save each speech segment
        filename = f"speech_{int(timestamp)}.wav"
        # You could process the audio here or save it
        print(f"   Could save to: {filename}")
    
    stream_manager = AudioStreamManager()
    processor = stream_manager.start_stream()
    
    # Monitor speech with callback
    stream_manager.monitor_speech(callback=on_speech_detected)

def main():
    """Main demo function"""
    print("Audio Processing Pipeline Demo")
    print("1. Simple Recording")
    print("2. Speech Detection")
    
    choice = input("Choose demo (1 or 2): ").strip()
    
    if choice == "1":
        simple_recording_demo()
    elif choice == "2":
        speech_detection_demo()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()