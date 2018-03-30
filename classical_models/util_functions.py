import numpy as np
import pandas as pd

def load_data(dataset='training', path='../data_processed/'):
    return pd.read_pickle(path + dataset + '_set.pkl')

def process_files_to_mfccs(dataset='training', path='../data_processed/', target_column='mfccs'):

    df = load_data(dataset=dataset, path=path)
    labels, files, column_values = [],[],[]

    for index, row in df.iterrows():
        for f in range(row['mfccs'].shape[1]):
            labels.append(row['Label'])
            files.append(index)
            column_values.append(row['mfccs'][:,f])

    df = pd.DataFrame({'File_id': files, 'Label': labels, 'column_values': column_values })

    #Here we make the lists inside the target column into independent columns, while keeping the file_id and label
    features_df = pd.concat([df['column_values'].apply(pd.Series), df['File_id'], df['Label']], axis = 1)
    features_df = features_df.set_index('File_id')

    return features_df
