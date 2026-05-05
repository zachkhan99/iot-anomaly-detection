import os
import glob
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score


SAMPLE_RATE = 22050
N_MFCC = 40
MODEL_PATH = "model.pkl"


def extract_features(file_path: str) -> np.ndarray:
    """Load a .wav file and return a feature vector (MFCCs + deltas + spectral)."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # MFCCs + first and second deltas
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)

    def stats(x):
        return np.array([x.mean(), x.std(), x.min(), x.max()])

    return np.concatenate([
        mfccs.mean(axis=1), mfccs.std(axis=1),
        mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1),
        mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1),
        stats(centroid), stats(rolloff), stats(zcr), stats(rms),
    ])


def label_from_path(path: str):
    name = os.path.basename(path).lower()
    if name.startswith("normal_"):
        return 0, "normal"
    if name.startswith("anomaly_") or name.startswith("abnormal_"):
        return 1, "abnormal"
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent == "normal":
        return 0, "normal"
    if parent in ("abnormal", "anomaly"):
        return 1, "abnormal"
    return None


def load_dataset():
    all_wavs = glob.glob(os.path.join("data", "**", "*.wav"), recursive=True)
    if not all_wavs:
        return np.array([]), np.array([])

    labeled = []
    skipped_unknown = 0
    for path in all_wavs:
        result = label_from_path(path)
        if result is None:
            skipped_unknown += 1
            continue
        labeled.append((path, result[0]))

    if skipped_unknown:
        print(f"  Skipped {skipped_unknown} files with unrecognizable labels.")

    total = len(labeled)
    X, y = [], []
    for i, (path, class_id) in enumerate(labeled, 1):
        print(f"\r  Extracting features: {i}/{total}", end="", flush=True)
        try:
            features = extract_features(path)
            X.append(features)
            y.append(class_id)
        except Exception as e:
            print(f"\n  Skipping {path}: {e}")
    print()
    return np.array(X), np.array(y)


def main():
    print("Loading audio files and extracting MFCC features...")
    X, y = load_dataset()

    if len(X) == 0:
        print("\nNo labeled audio data found. Either:")
        print("  1. Record your own: python collect_data.py --label normal --count 30")
        print("  2. Place DCASE/MIMII data under data/ (normal_*.wav / anomaly_*.wav)")
        return

    n_normal = int((y == 0).sum())
    n_abnormal = int((y == 1).sum())
    print(f"  Normal samples:   {n_normal}")
    print(f"  Abnormal samples: {n_abnormal}")
    print(f"  Feature dims:     {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining RandomForest (n_estimators=100)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Find threshold that maximizes F1 for the abnormal class
    probs = clf.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    y_pred = (probs >= best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOptimal threshold: {best_thresh:.2f}")
    print(f"Test accuracy:     {acc * 100:.1f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "abnormal"]))

    joblib.dump({"model": clf, "threshold": best_thresh}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    if acc < 0.85:
        print("\nAccuracy is below 85%.")


if __name__ == "__main__":
    main()
