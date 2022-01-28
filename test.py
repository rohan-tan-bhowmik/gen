import glob
from os.path import join
import librosa
import matplotlib.pyplot as plt

files = []
for ext in ('*.m4a', '*.ogg', '*.mp3'):
   files.extend(glob.glob(join("light/*/", ext)))


for dir in files:
    y, sr = librosa.core.load(dir)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    plt.imshow(S)
    plt.show()