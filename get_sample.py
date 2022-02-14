import librosa
import soundfile
from scipy.ndimage.filters import gaussian_filter

y, sr = librosa.core.load("kikuwu\Kikuo - 3rdAlbum _きくおミク2 (Kikuo Miku 2)_ X-FADE DEMO PV\Kikuo - 3rdAlbum _きくおミク2 (Kikuo Miku 2)_ X-FADE DEMO PV (128kbit_AAC).m4a")
print(sr)
#S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512, fmax=4000))
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, fmax=sr/2)

M = librosa.feature.inverse.mel_to_stft(S[:,:1600])
print("oibe")
y = librosa.griffinlim(M)
print("twj9")
soundfile.write('sample.wav', y, sr)