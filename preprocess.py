import librosa
import numpy as np

def extract_features(file, max_pad_len=200):
    try:

        audio, sr = librosa.load(file, duration=3)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        features = np.vstack((mfcc, chroma))

        if features.shape[1] < max_pad_len:
            pad_width = max_pad_len - features.shape[1]
            features = np.pad(features, ((0,0),(0,pad_width)), mode='constant')
        else:
            features = features[:, :max_pad_len]

        return features

    except Exception as e:
        print("Error extracting features:", e)
        return None
