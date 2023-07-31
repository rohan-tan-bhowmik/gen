import librosa
import soundfile
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import numpy as np

dirs = ["zzz.png"]#["sample_{}.png".format(i) for i in range(16)]

amplify = 4
for dir in dirs:
    audio = (plt.imread(dir) * 25)
    plt.imshow(audio)
    plt.show()
    #audio = np.maximum(audio - np.percentile(audio, 15), 0)
    #audio -= np.min(audio)
    #audio /= np.max(audio)
    #plt.imshow(audio)
    #plt.show()
    S = np.asarray(audio[:, :, 0])#, dtype=np.float64)
    print(S.shape)
    sr = 22050
    
    print("oibe")
    y = librosa.griffinlim(S)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
    print("twj9")
    print('{}_output.wav'.format(dir.split(".")[0]))
    soundfile.write('{}_output.wav'.format(dir.split(".")[0].split("/")[-1]), y, sr)