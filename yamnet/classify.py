import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import librosa

# Load model once
model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load class names
with open("yamnet_class_map.csv", "r") as f:
    class_names = [line.strip().split(",")[2] for line in f.readlines()]

def classify_audio(wav_path):
    waveform, sr = librosa.load(wav_path, sr=16000)
    waveform = waveform[:16000 * 10]  # Limit to 10 sec max
    scores, _, _ = model(waveform)

    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top5_idx = mean_scores.argsort()[-5:][::-1]

    top5 = [
        {"label": class_names[i], "confidence": round(float(mean_scores[i]), 4)}
        for i in top5_idx
    ]
    return top5[0]["label"], top5
