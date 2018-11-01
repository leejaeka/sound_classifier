import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data(dataset='training'):
    return pd.read_pickle('../data_processed/' + dataset + '_set.pkl')

def get_dimensions(mel_shape, shape='stacked'):
    if shape =='flat':
        mel_depth = 1
        mel_height = 40
    elif shape == 'stacked':
        mel_depth = 2
        mel_height = 20

    mel_width = int(mel_shape[1])
    return mel_height, mel_width, mel_depth

def process_files(dataset='training', features=['mfccs'], shape='flat', window_size=28):

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

def decide_class(prediction):
    if prediction < 0.5:
        file_prediction = 0
    else:
        file_prediction = 1
    return file_prediction

def get_class_label(index):
    if index == 0:
        label = 'cat'
    else:
        label = 'dog'
    return label

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
