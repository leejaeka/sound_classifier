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

def load_data(dataset='training'):
    return pd.read_pickle('../data_processed/' + dataset + '_set.pkl')

def get_dimensions(mel_shape, shape='stacked'):
    if shape =='flat':
        mel_depth = 1
        mel_height = 256
    elif shape == 'stacked':
        mel_depth = 2
        mel_height = 128

    mel_width = int(mel_shape[1])
    return mel_height, mel_width, mel_depth

def process_files(dataset='training', features=['Mel'], shape='mel_only', window_size=28):

    df = load_data(dataset=dataset)

    #Where it will be stored
    files, labels, data = [],[],[]

    #List of file names in the dataset
    file_names = list(df.File_id.unique())

    for index, row in df.iterrows():

        #Load the needed columns, and stack them, move the volume dim to the end
        mel = np.array(row[features])
        mel = np.stack((mel))

        #obtain some dimentions about the set to load
        if len(features) > 1:
            mel_height, mel_width, mel_depth = get_dimensions(shape=shape, mel_shape=mel.shape)
        else:
            mel_height, mel_width, mel_depth = mel.shape[1], mel.shape[2], mel.shape[0]

        #each mel needs to be chopped into segments of window_size width
        batch_size = int(mel.shape[2] / window_size)

        #reshape mel and remove parts that will be ignored
        mel = np.reshape(mel[:,:,0:batch_size*window_size], (mel_depth, mel_height, batch_size*window_size))

        for i in list(range(batch_size)):
            labels.append(row['Label'])
            files.append(row['File_id'])
            data.append(mel[:,:,i*window_size:(i+1)*window_size])

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
