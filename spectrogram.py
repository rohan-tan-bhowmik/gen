import glob
import os
from os.path import join
import librosa
import soundfile
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

files = []
for ext in ('*.m4a', '*.ogg', '*.mp3'):
   files.extend(glob.glob(join("kikuwu/*/", ext)))
'''
for i in range(len(files)):
   os.mkdir("img\\{}".format(i))
'''
for i in range(len(files)):
   os.mkdir("img\\{}".format(i))

num = 0
for dir in files:
   print(dir.split("\\")[-1])

   y, sr = librosa.core.load(dir)
   print(sr)
   #S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512, fmax=4000))
   S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512, fmax=sr/2)
   plt.title(dir.split("\\")[-1])
   length = 512
   stride = 128
   for i in range(0, len(S[0]) - length, stride):
      im = Image.fromarray(S[:,i:i+length])
      if im.mode != 'RGB':
         im = im.convert('RGB')
      im.save("img\\{}\\{}.png".format(num, i))
      print("{}/{}".format(i, len(S[0])))
      #plt.imshow(S[:,i:i+length])
      #plt.show()
   num += 1  

   print("PROGRESS: {}/{}".format(num, len(files)))
   '''
   M = librosa.feature.inverse.mel_to_stft(S[:,:1600])
   print("oibe")
   y = librosa.griffinlim(M)
   print("twj9")
   soundfile.write('sample.wav', y, sr)
   '''