import librosa
import soundfile
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import numpy as np

amplify = 2
S = amplify * np.asarray(plt.imread("img/25/2816.png")[:, :, 0])#, dtype=np.float64)
sr = 22050

M = librosa.feature.inverse.mel_to_stft(S[:,:1600])
print("oibe")
y = librosa.griffinlim(M)
print("twj9")
soundfile.write('sample.wav', y, sr)