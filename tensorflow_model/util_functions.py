import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

# import h5py
# import tensorflow as tf

# import glob
# import re

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
