import threading
import time
import queue
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import librosa
import joblib
import sounddevice as sd
import streamlit as st


SAMPLE_RATE = 22050
CHUNK_DURATION = 1.0          # seconds per inference window
N_MFCC = 40
MODEL_PATH = "model.pkl"
MAX_LOG_ENTRIES = 200
REFRESH_INTERVAL = 1.0        # seconds between dashboard refreshes


def extract_features(audio: np.ndarray) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)

    def stats(x):
        return np.array([x.mean(), x.std(), x.min(), x.max()])

    return np.concatenate([
        mfccs.mean(axis=1), mfccs.std(axis=1),
        mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1),
        mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1),
        stats(centroid), stats(rolloff), stats(zcr), stats(rms),
    ]).reshape(1, -1)

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.label = "—"
        self.confidence = 0.0
        self.is_abnormal = False
        self.total_windows = 0
        self.abnormal_windows = 0
        self.start_time = datetime.now()
        self.log: deque = deque(maxlen=MAX_LOG_ENTRIES)

    def update(self, label: str, confidence: float, is_abnormal: bool):
        with self.lock:
            self.label = label
            self.confidence = confidence
            self.is_abnormal = is_abnormal
            self.total_windows += 1
            if is_abnormal:
                self.abnormal_windows += 1
            self.log.appendleft({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Status": label,
                "Confidence": f"{confidence * 100:.1f}%",
            })

    def snapshot(self):
        with self.lock:
            elapsed = datetime.now() - self.start_time
            return {
                "label": self.label,
                "confidence": self.confidence,
                "is_abnormal": self.is_abnormal,
                "total_windows": self.total_windows,
                "abnormal_windows": self.abnormal_windows,
                "uptime": str(elapsed).split(".")[0],
                "anomaly_rate": (
                    self.abnormal_windows / self.total_windows
                    if self.total_windows > 0 else 0.0
                ),
                "log": list(self.log),
            }


SMOOTHING_WINDOWS = 5  # number of windows to average before deciding label


def inference_loop(state: SharedState, model, threshold: float, error_q: queue.Queue):
    """Background thread: continuously captures audio and runs inference."""
    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    prob_buffer = deque(maxlen=SMOOTHING_WINDOWS)
    try:
        while True:
            audio = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()
            features = extract_features(audio)
            probs = model.predict_proba(features)[0]
            prob_buffer.append(probs[1])

            if len(prob_buffer) < SMOOTHING_WINDOWS:
                continue  # wait until buffer is full before classifying

            avg_prob = float(np.mean(prob_buffer))
            class_id = 1 if avg_prob >= threshold else 0
            label = "ABNORMAL" if class_id == 1 else "NORMAL"
            confidence = avg_prob if class_id == 1 else 1.0 - avg_prob
            state.update(label, confidence, class_id == 1)
    except Exception as e:
        error_q.put(str(e))

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔊",
    layout="centered",
)

# Load model once per session
@st.cache_resource
def load_model():
    try:
        saved = joblib.load(MODEL_PATH)
        return saved["model"], saved["threshold"], None
    except FileNotFoundError:
        return None, None, f"Model file '{MODEL_PATH}' not found. Run `python train.py` first."


model, threshold, load_error = load_model()

if load_error:
    st.error(load_error)
    st.stop()


# Initialize shared state and background thread once per session
if "state" not in st.session_state:
    st.session_state.state = SharedState()
    st.session_state.error_q = queue.Queue()
    t = threading.Thread(
        target=inference_loop,
        args=(st.session_state.state, model, threshold, st.session_state.error_q),
        daemon=True,
    )
    t.start()

state: SharedState = st.session_state.state
error_q: queue.Queue = st.session_state.error_q

# Surface any background thread errors
if not error_q.empty():
    err = error_q.get_nowait()
    st.error(f"Audio capture error: {err}")
    st.stop()

# Read snapshot
snap = state.snapshot()

st.title("Predictive Maintenance Monitor")
st.caption("Real-time audio classification — normal vs. abnormal operation")

st.divider()

# Status indicator
if snap["label"] == "—":
    st.info("Waiting for first audio window...")
else:
    if snap["is_abnormal"]:
        st.markdown(
            f"<h1 style='color:#e53935;text-align:center'>🔴 {snap['label']}</h1>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h1 style='color:#43a047;text-align:center'>🟢 {snap['label']}</h1>",
            unsafe_allow_html=True,
        )

# Confidence bar
if snap["label"] != "—":
    st.metric("Confidence", f"{snap['confidence'] * 100:.1f}%")
    st.progress(snap["confidence"])

st.divider()

# Stats row
col1, col2, col3 = st.columns(3)
col1.metric("Uptime", snap["uptime"])
col2.metric("Windows Analyzed", snap["total_windows"])
col3.metric("Anomaly Rate", f"{snap['anomaly_rate'] * 100:.1f}%")

st.divider()

# Log table
st.subheader("Event Log")
if snap["log"]:
    st.dataframe(snap["log"], use_container_width=True, height=300)
else:
    st.caption("No events recorded yet.")

# Auto-refresh
time.sleep(REFRESH_INTERVAL)
st.rerun()
