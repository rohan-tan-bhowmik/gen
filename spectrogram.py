import os
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
from numba import jit
import librosa
import pandas as pd
import IPython.display as ipd
import glob
from os.path import join

import sys
sys.path.append('..')
import libfmp.c2

#%matplotlib inline
files = []
for ext in ('*.m4a', '*.ogg', '*.mp3'):
   files.extend(glob.glob(join("kikuwu/*/", ext)))

i = 0
for file in files:
    i += 1
    print("PROGRESS: {0}/{1}: {2}".format(i, len(files), file))
    # Load wav
    #fn_wav = "kikuwu/(HD) Red Riding Hood's Wolf PV【KIKUO】- English Subs/(HD) Red Riding Hood's Wolf PV【KIKUO】- English Subs (152kbit_Opus).ogg"
    Fs = 22050
    x, Fs = librosa.load(file, sr=Fs)

    # Compute Magnitude STFT
    N = 4096
    H = 1024
    X, T_coef, F_coef = libfmp.c2.stft_convention_fmp(x, Fs, N, H)
    Y = np.abs(X)

    #eps = np.finfo(float).eps
    #plt.imshow(10 * np.log10(eps + Y), origin='lower', aspect='auto', cmap='gray_r', extent=[T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]])

    # Plot spectrogram
    #eps = np.finfo(float).eps
    #plt.imshow(Y, cmap='gray')#, origin='lower', aspect='auto', cmap='gray_r', extent=[T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]])

    plt.imsave("kikuwu_specs/{}.png".format(file.split("/")[-1].split(".")[0]), Y, cmap='gray')
    #plt.show()