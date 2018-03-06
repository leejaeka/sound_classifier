import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import librosa.display
#import librosa.power_to_db

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp, sr=None)
        raw_sounds.append(X)
        print(sr)
    return sr,raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        print('going to plot',f)
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot')
    plt.show()

def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram')
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.power_to_db(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram')
    plt.show()

def plot_mel_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        print(n,f)
        S = librosa.feature.melspectrogram(f, sr=22050, n_mels=128)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 4: Mel spectrogram')
    plt.show()


import os
import re
relevant_path = "C:/Users/eurico/WordDocuments/QMUL_MSc/DataAnalytics/coursework/audio-cats-and-dogs/cats_dogs/"
# get files from directory and do all or a few, depending on range extracted below
sound_file_paths = [relevant_path+f for f in os.listdir(relevant_path)][0:3]
sound_names=sound_file_paths[0:3]

import audioread
import numpy as np
import scipy.signal
import scipy.fftpack as fft
import resampy
with audioread.audio_open(os.path.realpath(sound_file_paths[0])) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
print('sr_native:',sr_native)
print('n_channels:',n_channels)

print('going to load:',sound_file_paths)
sr,raw_sounds = load_sound_files(sound_file_paths)
print('sampling rate:',sr)

print('going to plot wav')
plot_waves(sound_names,raw_sounds)
print('going to plot specgram')
plot_specgram(sound_names,raw_sounds)
#print('going to plot log_power_specgram')
#plot_log_power_specgram(sound_names,raw_sounds)
print('going to plot mel_specgram')
plot_mel_specgram(sound_names,raw_sounds)



