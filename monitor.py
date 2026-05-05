import time
import sys
from collections import deque
from datetime import datetime

import numpy as np
import librosa
import joblib
import sounddevice as sd


SAMPLE_RATE = 22050
CHUNK_DURATION = 1.0
N_MFCC = 40
MODEL_PATH = "model.pkl"
SMOOTHING_WINDOWS = 5


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


def main():
    try:
        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        threshold = saved["threshold"]
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Run python3 train.py first.")
        sys.exit(1)

    print(f"Loaded model (threshold: {threshold:.2f})")

    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    total = 0
    anomalies = 0
    prob_buffer = deque(maxlen=SMOOTHING_WINDOWS)

    print("Listening... Press Ctrl+C to stop.\n")
    print(f"{'Time':<12} {'Status':<12} {'Confidence':>10}  {'Anomaly Rate':>12}")
    print("-" * 50)

    try:
        while True:
            audio = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()

            features = extract_features(audio)
            probs = model.predict_proba(features)[0]
            prob_buffer.append(probs[1])

            if len(prob_buffer) < SMOOTHING_WINDOWS:
                continue

            avg_prob = float(np.mean(prob_buffer))
            class_id = 1 if avg_prob >= threshold else 0
            label = "ABNORMAL" if class_id == 1 else "NORMAL"
            confidence = avg_prob if class_id == 1 else 1.0 - avg_prob

            total += 1
            if class_id == 1:
                anomalies += 1

            anomaly_rate = anomalies / total * 100
            ts = datetime.now().strftime("%H:%M:%S")
            flag = " <--" if class_id == 1 else ""
            print(f"{ts:<12} {label:<12} {confidence * 100:>9.1f}%  {anomaly_rate:>11.1f}%{flag}")

    except KeyboardInterrupt:
        print(f"\nStopped. {total} windows analyzed, {anomalies} anomalies detected.")


if __name__ == "__main__":
    main()
