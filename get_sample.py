import librosa
import soundfile
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import numpy as np

amplify = 2
for i in range(16):
    print(i)
    audio = plt.imread("sample_{}.png".format(i))
    #audio = np.maximum(audio - np.percentile(audio, 15), 0)
    #audio -= np.min(audio)
    #audio /= np.max(audio)
    #plt.imshow(audio)
    #plt.show()
    S = amplify * np.asarray(audio[:, :, 0])#, dtype=np.float64)
    sr = 22050

    M = librosa.feature.inverse.mel_to_stft(S[:,:1600])
    print("oibe")
    y = librosa.griffinlim(M)
    print("twj9")
    soundfile.write('sample_{}.wav'.format(i), y, sr)