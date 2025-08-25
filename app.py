import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import torch
import torch.nn as nn
import librosa
import av
import queue
import threading
import time

# Global communication
AUDIO_BUFFER = queue.Queue(maxsize=100)
PREDICTION = {'digit': None, 'confidence': 0.0, 'timestamp': 0}
PREDICTION_LOCK = threading.Lock()

# Dummy model (replace with your trained model)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleModel()
model.eval()

# Feature extraction
def extract_features(audio, sr):
    if len(audio) < 1024:
        return None
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
    return np.mean(mfcc, axis=1)

# Prediction
def predict(features):
    try:
        x = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()
        return pred, conf
    except Exception as e:
        print("Prediction error:", e)
        return None, None

# Audio callback
def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
    try:
        audio = frame.to_ndarray()
        if audio.ndim > 1:
            audio = audio[:, 0]
        AUDIO_BUFFER.put_nowait((audio, frame.sample_rate))
    except queue.Full:
        pass
    return frame

# Streamlit UI
st.title("🎤 Real-Time Digit Recognition")
st.write("Click **Start** and say a digit (0–9).")

webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

status = st.empty()
prediction_display = st.empty()

if webrtc_ctx.state.playing:
    status.success("🎙️ Recording...")

    # Accumulate audio
    chunks = []
    sr = 16000

    while True:
        try:
            # Get latest audio
            while not AUDIO_BUFFER.empty():
                chunk, sr = AUDIO_BUFFER.get_nowait()
                chunks.append(chunk)

            # Use latest ~1s of audio
            total = np.concatenate(chunks[-5:]) if chunks else np.array([])
            if len(total) > sr // 2:
                feats = extract_features(total, sr)
                if feats is not None:
                    pred, conf = predict(feats)
                    if pred is not None:
                        with PREDICTION_LOCK:
                            PREDICTION.update({
                                'digit': pred, 'confidence': conf, 'timestamp': time.time()
                            })

            # Display result
            with PREDICTION_LOCK:
                if PREDICTION['digit'] is not None:
                    prediction_display.markdown(
                        f"### 🎯 Predicted: **{PREDICTION['digit']}**\n"
                        f"**Confidence:** {PREDICTION['confidence']:.1%}"
                    )
                else:
                    prediction_display.markdown("🎧 Listening...")

            time.sleep(0.2)
        except Exception as e:
            st.error(f"Error: {e}")
            break

else:
    status.info("🔘 Click Start to begin.")
    prediction_display.markdown("Awaiting audio...")

