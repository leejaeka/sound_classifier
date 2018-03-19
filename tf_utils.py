import h5py
import numpy as np
import tensorflow as tf
import math
import glob
from sklearn.model_selection import train_test_split

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
