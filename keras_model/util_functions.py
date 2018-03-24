import h5py
import numpy as np
import tensorflow as tf
import math
import glob
from sklearn.model_selection import train_test_split
import re
import pickle
import pandas as pd

def get_class_name(filename):
    if 'cat' in filename:
        class_name = 0
    else:
        class_name = 1
    return class_name

def load_data(dataset='Train'):
    return pd.read_pickle('../data_processed/' + dataset + '_MELandDeltas.pkl')

def get_dimensions(shape='mel_only', frames=None):
    if shape=='mel_only':
        mel_height = 128
        mel_depth = 1
    elif shape=='mel_delta_stacked':
        mel_height = 256
        mel_depth = 1
    elif shape=='mel_delta_channels':
        mel_height = 128
        mel_depth = 2
    mel_width = int(frames.shape[0]/mel_height/mel_depth)
    return mel_height, mel_width, mel_depth

def extract_mel_spectrograms(dataset='Train', features=['Mel'], shape='mel_only'):

    df = load_data(dataset)

    #Where it will be stored
    files = []
    labels = []
    data = []

    #List of file names in the dataset
    file_names = list(df.File_id.unique())

    for file in file_names:

        class_name = get_class_name(file)
        #Filter for the file and extract needed features
        frames = np.array(df[df['File_id'] == file][features])
        frames = frames.ravel()
        frames = np.concatenate(frames)

        #obtain some dimentions about the set to load
        mel_height, mel_width, mel_depth = get_dimensions(shape=shape, frames=frames)

        #Combine all the frames into a mel_spectrogram
        try:
            mel = np.reshape(frames, (mel_height, mel_width, mel_depth))
        except ValueError:
            print(file)
            mel = np.reshape(frames, (mel_height, mel_width, mel_depth))


        #each mel needs to be chopped into segments of 28 width
        batch_size = int(mel.shape[1] / 28)
        for i in list(range(batch_size)):
            labels.append(class_name)
            files.append(file)
            data.append(mel[:,i*28:(i+1)*28])

    return np.array(data, dtype=np.float32), np.array(labels), np.array(files)

def load_features():
    filelist = glob.glob('features_mel_spectrograms/*.npy')
    labels = []
    data = []

    for file in filelist:
        nfile = np.load(file)
        if 'cat' in file:
            label = 0 #'cat'
        else:
            label = 1 #'dog'
        crop = int(nfile.shape[1] / 28)
        for i in list(range(int(crop))):
            labels.append(label)
            data.append(nfile[:,i*28:(i+1)*28])

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=59)
    # return X_train, X_test, y_train, y_test
    return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(y_train), np.array(y_test)




def load_features_with_deltas_stacking():
    filelist_mels = glob.glob('features_mel_spectrograms/*.npy')
    labels = []
    data_mels = []
    data_deltas =[]

    for file_mels in filelist_mels:
        # first load the mels
        nfile_mels = np.load(file_mels)
        if 'cat' in file_mels:
            label = 0 #'cat'
        else:
            label = 1 #'dog'
        crop = int(nfile_mels.shape[1] / 28)
        for i in list(range(int(crop))):
            labels.append(label)
            data_mels.append(nfile_mels[:,i*28:(i+1)*28])
        # now load the deltas
        file_delta = file_mels.replace('features_mel_spectrograms', 'features_delta_spectograms')
        nfile_delta = np.load(file_delta)
        crop = int(nfile_delta.shape[1] / 28)
        for i in list(range(int(crop))):
            data_deltas.append(nfile_delta[:,i*28:(i+1)*28])
    # and now stack horizontally on 2nd axis, so input is e.g.
    # (2029, 128, 28) + (2029, 128, 28) and output is (2029, 256, 28)
    data = np.hstack((np.array(data_mels),np.array(data_deltas)))

    # now split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=59)
    # return X_train, X_test, y_train, y_test
    return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(y_train), np.array(y_test)

def load_features_with_deltas_stacking_extend_maximum_fixed_size():
    import glob
    filelist_mels = glob.glob('features_mel_spectrograms/*.npy')
    labels = []
    data_mels = []
    data_deltas =[]
    maxsize=562  # from http://localhost:8888/notebooks/mel_spectrogram/import_mel_npy.ipynb

    for file_mels in filelist_mels:
        # first load the mels
        nfile_mels = np.load(file_mels)
        if 'cat' in file_mels:
            label = 0 #'cat'
        else:
            label = 1 #'dog'
        labels.append(label)
        ## now extend file to maximum size
        while nfile_mels.shape[1] < maxsize:
           nfile_mels=np.hstack((nfile_mels,nfile_mels))
        nfile_mels=nfile_mels[:,0:maxsize]
        data_mels.append(nfile_mels)
        # now load the deltas
        file_delta = file_mels.replace('features_mel_spectrograms', 'features_delta_spectograms')
        nfile_delta = np.load(file_delta)
        ## now extend file to maximum size
        while nfile_delta.shape[1] < maxsize:
           nfile_delta=np.hstack((nfile_delta,nfile_delta))
        nfile_delta=nfile_delta[:,0:maxsize]
        data_deltas.append(nfile_delta)
    # and now stack horizontally on 2nd axis, so input is e.g.
    # (277, 128, maxsize) + (277, 128, maxsize) and output is (277, 256, maxsize)
    data = np.hstack((np.array(data_mels),np.array(data_deltas)))
    labels=np.array(labels)
    #
    print(data.shape)
    print(labels.shape)
    #
    # now split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=59)
    #
    print( X_train.shape)
    print( X_test.shape)
    print( y_train.shape)
    print( y_test.shape)
    # return X_train, X_test, y_train, y_test
    return np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32), \
           np.array(y_train, dtype=np.float32), np.array(y_test, dtype=np.float32)

def load_features_with_deltas_stacking_nosplit():
    filelist_mels = glob.glob('features_mel_spectrograms/*.npy')
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
        file_delta = file_mels.replace('features_mel_spectrograms', 'features_delta_spectograms')
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
    filelist_mels = glob.glob('features_mel_spectrograms/*.npy')
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

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:]
    shuffled_Y = Y[permutation].reshape((Y.shape[0],1))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in list(range(0, int(num_complete_minibatches))):
        mini_batch_X = shuffled_X[int(k * mini_batch_size) : int(k * mini_batch_size + mini_batch_size),:,:]
        mini_batch_Y = shuffled_Y[int(k * mini_batch_size) : int(k * mini_batch_size + mini_batch_size), :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[int(num_complete_minibatches * mini_batch_size) : int(m), :,:]
        mini_batch_Y = shuffled_Y[int(num_complete_minibatches * mini_batch_size) : int(m), :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
