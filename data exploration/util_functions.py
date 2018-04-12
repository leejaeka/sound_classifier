import h5py
import numpy as np
import tensorflow as tf
import pandas as pd
import librosa
import math, glob, re, audioread
from sklearn.utils import shuffle

def capture_class(file_id):
    if 'cat' in file_id:
        class_name = 0
    elif 'dog' in file_id:
        class_name = 1
    else:
        class_name = -1
    return class_name

def colour_class(x):
    if x == 0:
        return 'red' # cat
    else:
        return 'blue' # dog

def random_shuffle(dataframe, seed=0, test_size=0.3, num_classes=2):

    training_frames, test_frames = [],[]

    #This assumes we're labeling our classes from 0 to num_classes
    for i in range(num_classes):
        #Filter on the class and shuffle
        shuffled_df = shuffle(dataframe[dataframe['Label'] == i], random_state=seed)
        #Split into training and test
        total_rows = shuffled_df.shape[0]
        test_rows = int(total_rows*test_size)
        shuffled_df.iloc[0:]

        training_frames.append(shuffled_df.iloc[0:total_rows-test_rows])
        test_frames.append(shuffled_df.iloc[total_rows-test_rows:total_rows])

    training_df = pd.concat(training_frames)
    test_df = pd.concat(test_frames)
    return training_df, test_df

def load_and_save_mel_data(files_path, sr=16000, dest_path=''):

    # get all wav files in folder
    sound_file_paths = glob.glob(files_path + '*.wav')

    #iterate over files and extract mels, save them as numpy files.
    for file in sound_file_paths:
        ts, sr = librosa.load(file,sr=sr)
        S = librosa.feature.melspectrogram(ts, sr=sr, n_mels=128)
        melogram=librosa.power_to_db(S, ref=np.max)
        filename = file.split('/')[-1].split('.')[0]
        np.save(dest_path+filename, melogram)

def load_and_save_mel_delta_data(files_path, sr=16000, dest_path=''):

    # get all wav files in folder
    sound_file_paths = glob.glob(files_path + '*.wav')

    #iterate over files and extract MFCCs, save them as numpy files.
    for file in sound_file_paths:
        ts, sr = librosa.load(file,sr=sr)
        S = librosa.feature.melspectrogram(ts, sr=sr, n_mels=128)
        melogram=librosa.power_to_db(S, ref=np.max)
        mfcc_delta = librosa.feature.delta(melogram)
        filename = file.split('/')[-1].split('.')[0]
        np.save(dest_path+filename, mfcc_delta)

def load_features_with_deltas_stacking_nosplit():
    filelist_mels = glob.glob('../data_processed/features_mel_spectrograms/*.npy')
    labels = []
    data_mels = []
    data_deltas =[]

    for file_mels in filelist_mels:
        # first load the mels
        nfile_mels = np.load(file_mels)
        filenumber =  int(re.findall('\d+', file_mels )[0])
        if 'cat' in file_mels:
            label = ((0,filenumber)) #'cat'
        else:
            label = ((1,filenumber)) #'dog'
        crop = int(nfile_mels.shape[1] / 28)
        for i in list(range(int(crop))):
            labels.append(label)
            data_mels.append(nfile_mels[:,i*28:(i+1)*28])
        # now load the deltas
        file_delta = file_mels.replace('data_processed/features_mel_spectrograms', 'data_processed/features_delta_spectograms')
        nfile_delta = np.load(file_delta)
        crop = int(nfile_delta.shape[1] / 28)
        for i in list(range(int(crop))):
            data_deltas.append(nfile_delta[:,i*28:(i+1)*28])
    # and now stack horizontally on 2nd axis, so input is e.g.
    # (2029, 128, 28) + (2029, 128, 28) and output is (2029, 256, 28)
    data = np.hstack((np.array(data_mels),np.array(data_deltas)))

    # return data and labels for unsupersized learning
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int16)

def load_features_with_deltas_stacking_nosplit_noslicing():
    filelist_mels = glob.glob('../data_processed/features_mel_spectrograms/*.npy')
    labels = []
    data_mels = []
    data_deltas =[]

    for file_mels in filelist_mels:
        # first load the mels
        nfile_mels = np.load(file_mels)
        filenumber =  int(re.findall('\d+', file_mels )[0])
        if 'cat' in file_mels:
            label = ((0,filenumber)) #'cat'
        else:
            label = ((1,filenumber)) #'dog'
        labels.append(label)
        data_mels.append(nfile_mels)

    # return data and labels for unsupersized learning

    return data_mels, np.array(labels, dtype=np.int16)

def load_sound_files(file_paths):
    sound_file_paths = glob.glob(file_paths + '*.wav')
    raw_sounds, raw_labels = [], []

    for fp in sound_file_paths:
        X,sr = librosa.load(fp, sr=None)
        if (fp.find('barking') != -1):
            labelfile=1
        else:
            labelfile=0
        raw_labels.append(labelfile)
        raw_sounds.append(X)
    return sr,raw_sounds,raw_labels

def energy(raw_sound):
    energy = sum(abs(raw_sound**2))/len(raw_sound)
    return energy

def rmse(raw_sound):
    rmse = np.sum(librosa.feature.rmse(y=raw_sound))
    return rmse
