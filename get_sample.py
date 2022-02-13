import librosa
import soundfile
from scipy.ndimage.filters import gaussian_filter

y, sr = librosa.core.load(dir)
print(sr)
#S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512, fmax=4000))
S = gaussian_filter(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512, fmax=8000), sigma=0.5)

M = librosa.feature.inverse.mel_to_stft(S[:,:1600])
print("oibe")
y = librosa.griffinlim(M)
print("twj9")
soundfile.write('sample.wav', y, sr)