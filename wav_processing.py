import numpy as np

def initialize_features(name):
    features = np.zeros(37)
    arr = np.loadtxt("c:\\Users\\EMAN\Desktop\\features_wav_h5.txt", dtype=float)
    rand_n = np.random.randn(37)
    features = arr + rand_n
    return features
