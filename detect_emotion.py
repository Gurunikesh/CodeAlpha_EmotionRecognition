import numpy as np
import os
import sys

# ── Check dependencies ───────────────────────────────────────
try:
    import librosa
except ImportError:
    sys.exit("Run: pip install librosa")

try:
    import tensorflow as tf
except ImportError:
    sys.exit("Run: pip install tensorflow")

try:
    import joblib
except ImportError:
    sys.exit("Run: pip install joblib")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Emotions in alphabetical order (how LabelEncoder sorts them)
EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


# ─────────────────────────────────────────────
# STEP 1 — EXTRACT FEATURES FROM AUDIO
# ─────────────────────────────────────────────

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3.0)

    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean   = np.mean(mfcc, axis=1)                                  # (40,)
    mfcc_std    = np.std(mfcc, axis=1)                                   # (40,)

    chroma      = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
    chroma_mean = np.mean(chroma, axis=1)                                # (12,)

    mel         = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_mean    = np.mean(librosa.power_to_db(mel), axis=1)              # (128,)

    return np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean]) # (220,)


# ─────────────────────────────────────────────
# STEP 2 — LOAD MODEL & PREDICT
# ─────────────────────────────────────────────

def predict(audio_path, model_path, scaler_path):

    # Validate files
    if not os.path.exists(audio_path):
        sys.exit(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        sys.exit(f"Model not found: {model_path}\n  Run emotion_recognition.py first.")
    if not os.path.exists(scaler_path):
        sys.exit(f"Scaler not found: {scaler_path}\n Run emotion_recognition.py first.")

    print(f"\nProcessing : {os.path.basename(audio_path)}")

    # Extract features
    features = extract_features(audio_path)                    # (220,)

    # Normalize using saved scaler
    scaler          = joblib.load(scaler_path)
    features_scaled = scaler.transform([features])             # (1, 220)

    # Reshape for CNN input
    features_3d = features_scaled.reshape(1, features_scaled.shape[1], 1)  # (1, 220, 1)

    # Predict
    model   = tf.keras.models.load_model(model_path)
    probs   = model.predict(features_3d, verbose=0)[0]         # (8,)
    top_idx = np.argmax(probs)

    return EMOTIONS[top_idx], probs[top_idx] * 100, probs


# ─────────────────────────────────────────────
# STEP 3 — DISPLAY RESULTS
# ─────────────────────────────────────────────

def show_result(audio_path, emotion, confidence, probs):

    print("\n" + "═" * 52)
    print(" EMOTION DETECTION RESULT")
    print("═" * 52)
    print(f"   File       : {os.path.basename(audio_path)}")
    print(f"   Detected   : {emotion.upper()}")
    print(f"   Confidence : {confidence:.1f}%")
    print("─" * 52)
    print("   All scores:")

    for idx in np.argsort(probs)[::-1]:
        filled = int(probs[idx] * 28)
        bar    = "|" * filled + ":" * (28 - filled)
        print(f"   {EMOTIONS[idx]:<12} {bar}  {probs[idx]*100:5.1f}%")

    print("═" * 52 + "\n")

# ✏️ Path to your .wav audio file
AUDIO_FILE  = "audio1.wav"

# ✏️ Path to trained model (created by emotion_recognition.py)
MODEL_FILE  = "emotion_model.keras"

# ✏️ Path to saved scaler (created by emotion_recognition.py)
SCALER_FILE = "scaler.pkl"

emotion, confidence, probs = predict(AUDIO_FILE, MODEL_FILE, SCALER_FILE)
show_result(AUDIO_FILE, emotion, confidence, probs)


    