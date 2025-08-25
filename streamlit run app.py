import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import io
import tempfile
from audio_recorder_streamlit import audio_recorder
import pickle
# Assuming you have your trained model saved
# import your_model_module

st.set_page_config(
    page_title="Audio Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Audio Digit Classifier")
st.write("Upload an audio file or record your voice saying a digit (0-9)")

# Load your trained model (replace with your actual model loading)
@st.cache_resource
def load_model():
    # Replace this with your actual model loading
    # model = pickle.load(open('digit_model.pkl', 'rb'))
    # return model
    st.warning("⚠️ Replace this with your actual trained model loading code")
    return None

model = load_model()

def extract_features(audio_data, sample_rate=8000):
    """Extract MFCC features from audio data"""
    # Ensure audio is the right length and sample rate
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Extract MFCC features (adjust parameters based on your model)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    return mfccs_mean

def predict_digit(audio_data, sample_rate):
    """Predict digit from audio data"""
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        # Extract features
        features = extract_features(audio_data, sample_rate)
        
        # Make prediction (adjust based on your model)
        # prediction = model.predict([features])[0]
        # confidence = model.predict_proba([features]).max()
        
        # Dummy prediction for demo (replace with actual prediction)
        prediction = np.random.randint(0, 10)
        confidence = np.random.uniform(0.7, 0.99)
        
        return str(prediction), confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def plot_audio_waveform(audio_data, sample_rate):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
    ax.plot(time, audio_data)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    ax.grid(True, alpha=0.3)
    return fig

def plot_spectrogram(audio_data, sample_rate):
    """Plot audio spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.stft(audio_data)
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(DB, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    return fig

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload a WAV, MP3, FLAC, or M4A file containing a spoken digit"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load audio file
        try:
            audio_data, sample_rate = librosa.load(tmp_file_path, sr=8000)
            
            st.success("✅ Audio file loaded successfully!")
            
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Make prediction
            prediction, confidence = predict_digit(audio_data, sample_rate)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            col_pred, col_conf = st.columns(2)
            
            with col_pred:
                st.metric("Predicted Digit", prediction)
            
            with col_conf:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Show confidence bar
            st.progress(confidence)
            
            # Display visualizations
            st.subheader("📊 Audio Analysis")
            
            # Waveform
            with st.expander("View Waveform"):
                fig_wave = plot_audio_waveform(audio_data, sample_rate)
                st.pyplot(fig_wave)
            
            # Spectrogram
            with st.expander("View Spectrogram"):
                fig_spec = plot_spectrogram(audio_data, sample_rate)
                st.pyplot(fig_spec)
                
        except Exception as e:
            st.error(f"❌ Error processing audio file: {str(e)}")

with col2:
    st.header("🎙️ Record Audio")
    st.write("Click the microphone button to record a digit")
    
    # Audio recorder component
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e8b4b8",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )
    
    if audio_bytes:
        st.success("✅ Audio recorded!")
        
        # Display audio player
        st.audio(audio_bytes, format='audio/wav')
        
        # Convert bytes to numpy array
        try:
            # Save audio bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            # Load audio
            audio_data, sample_rate = librosa.load(tmp_file_path, sr=8000)
            
            # Make prediction
            prediction, confidence = predict_digit(audio_data, sample_rate)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            col_pred, col_conf = st.columns(2)
            
            with col_pred:
                st.metric("Predicted Digit", prediction)
            
            with col_conf:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Show confidence bar
            st.progress(confidence)
            
            # Display visualizations
            st.subheader("📊 Audio Analysis")
            
            # Waveform
            with st.expander("View Waveform"):
                fig_wave = plot_audio_waveform(audio_data, sample_rate)
                st.pyplot(fig_wave)
            
            # Spectrogram
            with st.expander("View Spectrogram"):
                fig_spec = plot_spectrogram(audio_data, sample_rate)
                st.pyplot(fig_spec)
                
        except Exception as e:
            st.error(f"❌ Error processing recorded audio: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application classifies spoken digits (0-9) from audio input.
    
    **Features:**
    - 📁 Upload audio files
    - 🎙️ Real-time recording
    - 📊 Audio visualization
    - 🎯 Confidence scores
    
    **Supported Formats:**
    - WAV, MP3, FLAC, M4A
    """)
    
    st.header("⚙️ Settings")
    
    # Add some settings
    show_debug = st.checkbox("Show debug information")
    
    if show_debug:
        st.subheader("🐛 Debug Info")
        st.write("Model loaded:", model is not None)
        st.write("Streamlit version:", st.__version__)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Audio Digit Classification Challenge</p>
    </div>
    """,
    unsafe_allow_html=True
)