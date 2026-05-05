# IoT Anomaly Detection

Real-time acoustic anomaly detection for IoT/industrial devices using a Random Forest classifier trained on MFCC audio features. Captures audio from a microphone, classifies each window as normal or abnormal, and displays results via a live Streamlit dashboard.

## How It Works

1. **Collect** labeled audio samples from your device (normal and abnormal operation)
2. **Train** a Random Forest model on MFCC + spectral features extracted from those samples
3. **Monitor** live audio in real time via the terminal or a web dashboard

## Setup

```bash
pip install -r requirements.txt
```

Requires a working microphone and PortAudio (macOS: `brew install portaudio`).

## Usage

### 1. Collect Data

```bash
python collect_data.py --label normal --count 30 --duration 2
python collect_data.py --label abnormal --count 30 --duration 2
```

Saves `.wav` clips to `data/normal/` and `data/abnormal/`. You can also use pre-labeled datasets (e.g. DCASE, MIMII) by placing files named `normal_*.wav` / `anomaly_*.wav` under `data/`.

### 2. Train

```bash
python train.py
```

Extracts features, trains the model with an optimized classification threshold, and saves `model.pkl`.

### 3. Monitor

**Terminal:**
```bash
python monitor.py
```

**Dashboard:**
```bash
streamlit run dashboard.py
```

## Project Structure

```
collect_data.py   # Record labeled audio samples
train.py          # Feature extraction and model training
monitor.py        # Real-time terminal monitor
dashboard.py      # Streamlit live dashboard
requirements.txt
```

## Features

- 40 MFCCs + delta + delta-delta coefficients
- Spectral centroid, rolloff, ZCR, RMS energy
- Threshold tuned to maximize F1 on the abnormal class
- 5-window probability smoothing to reduce false positives
