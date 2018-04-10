import librosa
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import glob, audioread

def load_and_read_data(files_path, sr=16000):
    audio, sample_rates, channels=[],[],[]

    # get all wav files in folder
    sound_file_paths = glob.glob(files_path + '*.wav')

    #iterate over files and extract features.
    for file in sound_file_paths:
        ts, sr = librosa.load(file,sr=sr) #librosa returns a time series and sample rate
        audio.append(ts)
        sample_rates.append(sr)
        with audioread.audio_open(file) as input_file:
            channels.append(input_file.channels)

    return audio, sample_rates, channels, sound_file_paths

def get_more_audio_features(audio, sr=16000):
    frequencies, mel_deltas, mfccs, mfcc_deltas = [],[],[],[]

    for a in audio:
        # Get and store frequencies and their deltas
        fr = librosa.feature.melspectrogram(y=a,sr=sr)
        frequencies.append(fr)
        mel_deltas.append(librosa.feature.delta(fr))

        # Get and store mfccs and their deltas
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(fr),sr=sr)
        mfccs.append(mfcc)
        mfcc_deltas.append(librosa.feature.delta(mfcc))

    return frequencies, mel_deltas, mfccs, mfcc_deltas

def capture_class(file_id):
    if 'cat' in file_id:
        class_name = 0
    elif 'dog' in file_id:
        class_name = 1
    else:
        class_name = -1
    return class_name

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

def main():

    #path for audio files folder:
    raw_files_path = 'data/cats_dogs/'

    #call the function that will process the data.
    audio, sr, channels, file_names = load_and_read_data(raw_files_path)

    #get additional features from audio
    frequencies, mel_deltas, mfccs, mfcc_deltas = get_more_audio_features(audio)

    #Combining the lists into a single dataframe
    #The result will be a row per file with several attributes.
    features_df = pd.DataFrame({'audio': audio,
                                'sample_rates': sr,
                                'channels': channels,
                                'file_name': file_names,
                                'Mel': frequencies,
                                'Mel_deltas': mel_deltas,
                                'mfccs': mfccs,
                                'mfcc_deltas': mfcc_deltas,
                                'File_id': [f.replace('data/cats_dogs/', '').replace('.wav', '') for f in file_names]
                               })

    #Adding the class label to the dataframe
    features_df['Label'] = features_df.apply(lambda row: capture_class(row['File_id']), axis=1)

    #We'll shuffle our dataframe for each class and split into training and test set
    training_df, test_df = random_shuffle(features_df, seed=1)

    #save as pickles
    training_df.to_pickle('data_processed/'+ 'training_set.pkl')
    test_df.to_pickle('data_processed/'+ 'test_set.pkl')

if __name__ == '__main__':
    main()
