import h5py
import numpy as np
import tensorflow as tf
import math
import glob
from sklearn.model_selection import train_test_split
import re

def load_features():
    filelist = glob.glob('data_processed/features_mel_spectrograms/*.npy')
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

def load_features_with_deltas_stacking_dataframes(train_set_path,test_set_path):
    # filenames for now
    train_set_path='./Features_sets/Train_MELandDeltas.pkl'
    test_set_path='./Features_sets/Test_MELandDeltas.pkl'

    import pandas as pd
    import numpy as np
    df_train=pd.read_pickle(train_set_path)
    df_test=pd.read_pickle(test_set_path)

    # sort
    # df_train.sort_index(kind='mergesort').sort_values(by='File_id',kind='mergesort')
    # df_test.sort_index(kind='mergesort').sort_values(by='File_id',kind='mergesort')

    # count rows per animal

    rows_per_animal_train=df_train.groupby(['File_id']).size()
    rows_per_animal_test=df_test.groupby(['File_id']).size()

    # find max size
    maxsize=max(int(rows_per_animal_train.max()),int(rows_per_animal_test.max()))
    maxsize
    # group by

    filelist_mels_train = list(df_train.File_id.unique())
    filelist_mels_test = list(df_test.File_id.unique())
    filelist_mels_train.sort()
    filelist_mels_test.sort()

    # 277?
    len(filelist_mels_train)+len(filelist_mels_test)

    # first do train
    # empty
    fileids_train = []
    labels_train = []
    data_mels = []
    data_deltas =[]

    for file_mels in filelist_mels_train:
        # first load the mels
        nfile_mels = df_train.loc[df_train['File_id'] == file_mels]['Mel']
        # label
        label = int(df_train.loc[df_train['File_id'] == file_mels].head(1)['Label'])  # 0 for 'cat', 1 for 'dog'
        # sort
        nfile_mels.sort_index(kind='mergesort')
        # array
        nfile_mels = np.array(nfile_mels)
        nfile_mels = np.transpose(np.stack(nfile_mels, axis=0))
        nfile_mels.shape
        # now split into minimum common denominator size
        crop = int(nfile_mels.shape[1] / 28)
        for i in list(range(int(crop))):
            labels_train.append(label)
            fileids_train.append(file_mels)
            data_mels.append(nfile_mels[:,i*28:(i+1)*28])
        # now load the deltas
        nfile_delta = df_train.loc[df_train['File_id'] == file_mels]['Mel_deltas']
        # sort
        nfile_delta.sort_index(kind='mergesort')
        # array
        nfile_delta = np.array(nfile_delta)
        nfile_delta = np.transpose(np.stack(nfile_delta, axis=0))
        nfile_delta.shape
        #
        crop = int(nfile_delta.shape[1] / 28)
        for i in list(range(int(crop))):
            data_deltas.append(nfile_delta[:,i*28:(i+1)*28])
    # and now stack horizontally on 2nd axis, so input is e.g.
    # (2029, 128, 28) + (2029, 128, 28) and output is (2029, 256, 28)
    data_train = np.hstack((np.array(data_mels),np.array(data_deltas)))

    data_train.shape

    # first do test
    # empty
    fileids_test = []
    labels_test = []
    data_mels = []
    data_deltas =[]

    for file_mels in filelist_mels_test:
        # first load the mels
        nfile_mels = df_test.loc[df_test['File_id'] == file_mels]['Mel']
        # label
        label = int(df_test.loc[df_test['File_id'] == file_mels].head(1)['Label'])  # 0 for 'cat', 1 for 'dog'
        # sort
        nfile_mels.sort_index(kind='mergesort')
        # array
        nfile_mels = np.array(nfile_mels)
        nfile_mels = np.transpose(np.stack(nfile_mels, axis=0))
        nfile_mels.shape
        # now split into minimum common denominator size
        crop = int(nfile_mels.shape[1] / 28)
        for i in list(range(int(crop))):
            labels_test.append(label)
            fileids_test.append(file_mels)
            data_mels.append(nfile_mels[:,i*28:(i+1)*28])
        # now load the deltas
        nfile_delta = df_test.loc[df_test['File_id'] == file_mels]['Mel_deltas']
        # sort
        nfile_delta.sort_index(kind='mergesort')
        # array
        nfile_delta = np.array(nfile_delta)
        nfile_delta = np.transpose(np.stack(nfile_delta, axis=0))
        nfile_delta.shape
        #
        crop = int(nfile_delta.shape[1] / 28)
        for i in list(range(int(crop))):
            data_deltas.append(nfile_delta[:,i*28:(i+1)*28])
    # and now stack horizontally on 2nd axis, so input is e.g.
    # (2029, 128, 28) + (2029, 128, 28) and output is (2029, 256, 28)
    data_test = np.hstack((np.array(data_mels),np.array(data_deltas)))

    data_test.shape

    # now split and return
    return np.array(data_train, dtype=np.float32), np.array(labels_train, dtype=np.float32), \
           np.array(data_test, dtype=np.float32), np.array(labels_test, dtype=np.float32), \
           np.array(fileids_train, dtype=np.str), np.array(fileids_test, dtype=np.str),


def load_features_with_deltas_stacking():
    filelist_mels = glob.glob('data_processed/features_mel_spectrograms/*.npy')
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
        file_delta = file_mels.replace('data_processed/features_mel_spectrograms', 'data_processed/features_delta_spectograms')
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
    filelist_mels = glob.glob('data_processed/features_mel_spectrograms/*.npy')
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
        file_delta = file_mels.replace('data_processed/features_mel_spectrograms', 'data_processed/features_delta_spectograms')
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
    filelist_mels = glob.glob('data_processed/features_mel_spectrograms/*.npy')
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
    filelist_mels = glob.glob('data_processed/features_mel_spectrograms/*.npy')
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
