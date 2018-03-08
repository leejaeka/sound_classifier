import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import librosa.display

relevant_path = "./../features_mel_spectrograms/"
# get files from directory and do all or a few, depending on range extracted below
sound_file_paths = [relevant_path+f for f in os.listdir(relevant_path)]

sound_file_paths

sound_file_mel_spectrograms=[]
sound_file_mel_spectrograms_sizes_128by=[]

for x in sound_file_paths:
    mel_spectrogram=np.load(x)
    sound_file_mel_spectrograms.append(mel_spectrogram)
    sound_file_mel_spectrograms_sizes_128by.append(mel_spectrogram.shape[1])
    print(mel_spectrogram.shape)
 
sound_file_mel_spectrograms_sizes_128by.sort() 
sound_file_mel_spectrograms_sizes_128by