import librosa
import numpy as np

def extract_features(file, max_pad_len=200):
    """
    Extract MFCC features from audio file
    """
    try:
        audio, sr = librosa.load(file, duration=3)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print("Error extracting features:", e)
        return None