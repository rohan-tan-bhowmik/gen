import glob
import os
from os.path import join
import librosa
import soundfile
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

list = open("img/list.txt", 'w', encoding="utf-8")

files = []
for ext in ('*.m4a', '*.ogg', '*.mp3'):
   files.extend(glob.glob(join("kikuwu/*/", ext)))

count = 0
for file in files:
    list.write("NUMBER {}: {}\n".format(count, file))
    count += 1

list.close()