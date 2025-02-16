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

print(files)

#for i in range(len(files)):
#   os.mkdir("img\\{}".format(i))

for i in range(len(files)):
   pass
   #os.mkdir("kikuwu_img/{}".format(i))
   #break

num = 0
for dir in files:
   print(dir.split("/")[-1])

   y, sr = librosa.core.load(dir)
   print(sr)

   length = 512
   stride = 128
   margin = 128
   #S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512, fmax=4000))
   S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512, fmax=sr/2)#sr/2)
   plt.title(dir.split("/")[-1])
   
   for i in range(len(S)):
      for j in range(len(S)):
         break
         S[i, j] = S[i, j] // 1

   j = 0
   for i in range(0, len(S[0]) - length - margin, stride):
      im = Image.fromarray(S[:,i:i+length + margin])
      if im.mode != 'RGB':
         im = im.convert('RGB')
      im.save("kikuwu_img/{}/{}.png".format(num, j))
      print("{}/{}".format(i, len(S[0])))
      #plt.imshow(S[:,i:i+length])
      #plt.show()
      j += 1
      print("PROGRESS: {}/{}".format(i, len(S)))

   num += 1  

   print("PROGRESS: {}/{}".format(num, len(files)))
   break
   '''
   M = librosa.feature.inverse.mel_to_stft(S[:,:1600])
   print("oibe")
   y = librosa.griffinlim(M)
   print("twj9")
   soundfile.write('sample.wav', y, sr)
   '''