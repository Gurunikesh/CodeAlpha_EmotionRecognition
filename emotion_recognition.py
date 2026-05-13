import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
warnings.filterwarnings("ignore")
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
    BatchNormalization, Flatten
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



# EMOTION MAPPING (RAVDESS standard)

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

COLORS = [
    "#3498db", "#95a5a6", "#f1c40f", "#2980b9",
    "#e74c3c", "#8e44ad", "#27ae60", "#e67e22"
]

print("=" * 40)
print("EMOTION RECOGNITION FROM SPEECH")
print("=" * 40)


# FEATURE EXTRACTION

def extract_features(file_path, n_mfcc=40, n_chroma=12, n_mel=128):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)        # (40,)
        mfcc_std = np.std(mfcc, axis=1)           # (40,)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1)     # (12,)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel)
        mel_mean = np.mean(librosa.power_to_db(mel), axis=1)   # (128,)

        # Concatenate → (220,)
        features = np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean])
        return features
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return None



# LOAD RAVDESS DATASET

def load_ravdess(data_dir="RAVDESS"):
    X, y = [], []

    if not os.path.exists(data_dir):
        return None, None

    from tqdm import tqdm
    wav_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"\n  Found {len(wav_files)} audio files in '{data_dir}/'")

    for path in tqdm(wav_files, desc="  Extracting features"):
        filename = os.path.basename(path)
        parts = filename.replace(".wav", "").split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        if emotion_code not in EMOTION_MAP:
            continue
        emotion = EMOTION_MAP[emotion_code]
        features = extract_features(path)
        if features is not None:
            X.append(features)
            y.append(emotion)

    return np.array(X), np.array(y)

# BUILD CNN-LSTM MODEL

def build_model(input_shape, num_classes):
    
    model = Sequential([
        # CNN Block 1
        Conv1D(64, kernel_size=3, activation="relu", padding="same",
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # CNN Block 2
        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # LSTM Block
        LSTM(128, return_sequences=False),
        Dropout(0.4),

        # Classifier
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# MAIN PIPELINE

print("\n Loading Data...")

if os.path.exists("RAVDESS"):
    X, y = load_ravdess("RAVDESS")
    if X is None or len(X) == 0:
        print("Data not found ")
        sys.exit()
        
else:
    sys.exit()

print(f"Samples: {len(X)} | Feature dim: {X.shape[1]}")

# ── Label Encoding ───────────────────────────
print("\n Preprocessing...")

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {list(le.classes_)}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import joblib
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved")

# Reshape for Conv1D: (samples, timesteps, features)
X_3d = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# One-hot encode labels
y_cat = to_categorical(y_enc, num_classes=num_classes)

# Train/val/test split
X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X_3d, y_cat, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)


# ── EDA Plot ─────────────────────────────────
emotion_counts = pd.Series(y).value_counts()
plt.figure(figsize=(10, 4))
emotion_counts.plot(kind="bar", color=COLORS[:len(emotion_counts)], edgecolor="black")
plt.title("Emotion Class Distribution (RAVDESS / Synthetic)", fontsize=13, fontweight="bold")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("emotion_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Build Model ──────────────────────────────
print("\n Building CNN-LSTM Model...")



model = build_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    num_classes=num_classes
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
]

# ── Train ────────────────────────────────────
print("\n Training Model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ─────────────────────────────────
print("\n Evaluation...")

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test_raw, y_pred)
print(f"\n Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n Classification Report:")
print(classification_report(y_test_raw, y_pred, target_names=le.classes_))

# ── Plots ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Emotion Recognition — Evaluation", fontsize=15, fontweight="bold")

# Training curves
axes[0].plot(history.history["accuracy"], label="Train Acc", color="#2ecc71", lw=2)
axes[0].plot(history.history["val_accuracy"], label="Val Acc", color="#e74c3c", lw=2, ls="--")
axes[0].set_title("Training & Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss", color="#3498db", lw=2)
axes[1].plot(history.history["val_loss"], label="Val Loss", color="#e67e22", lw=2, ls="--")
axes[1].set_title("Training & Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

# Confusion matrix
cm = confusion_matrix(y_test_raw, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")
plt.setp(axes[2].get_xticklabels(), rotation=30, ha="right")
plt.setp(axes[2].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("emotion_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()

model.save("emotion_model.keras")
print("Model saved ")    

