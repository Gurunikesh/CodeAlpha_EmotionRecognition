# Emotion Recognition from Speech using RAVDESS

This project detects human emotions from speech audio using a CNN-LSTM deep learning model trained on the RAVDESS dataset.

## Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Dataset: www.kaggle.com/datasets/gurunikeshs/ravdess-speech-emotion-recognition-dataset

## Emotions Detected
- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Technologies Used
- Python
- TensorFlow / Keras
- Librosa
- Scikit-learn
- Matplotlib
- Seaborn

## Model Architecture
- Conv1D
- BatchNormalization
- MaxPooling1D
- LSTM
- Dense Layers

## Accuracy
Test Accuracy: ~46%

## Files
- emotion_recognition.py → Training script
- detect_emotion.py → Predict emotion from audio
- emotion_model.keras → Trained model
- scaler.pkl → Feature scaler
- emotion_distribution.png → Dataset distribution
- emotion_evaluation.png → Accuracy, loss, confusion matrix

## How to Run

### Train Model
python emotion_recognition.py

### Detect Emotion
python detect_emotion.py
