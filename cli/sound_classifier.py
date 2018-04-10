from __future__ import print_function
import argparse, sys, os, warnings
import librosa
import numpy as np
from numpy import linalg as LA
import keras
from keras.models import model_from_json

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS=None

def decide_class(prediction):
    if prediction <= 0.5:
        file_prediction = '😺  cat!!!'
    else:
        file_prediction = '🐶  dog!!!'
    return file_prediction

def get_final_prediction(scores):
    scores = [np.argmax(s) for s in scores]
    # print(np.mean(scores))
    return decide_class(np.mean(scores))

def pre_process_file(file, model):

    if FLAGS.file_path == 'data/cats_dogs.wav':
        file = os.getcwd() + '/' + file

    ts, sr = librosa.load(file)

    if model == 'mel':
        frequency = librosa.feature.melspectrogram(y=ts,sr=sr)
        mel_delta = librosa.feature.delta(frequency)
        return frequency, mel_delta
    else:
        frequency = librosa.feature.melspectrogram(y=ts,sr=sr)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(frequency),sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)
        return mfcc, mfcc_delta

def process_file(feature, feature_delta):

    if FLAGS.model_type.lower() == 'mel':
        height = 128
    else:
        height = 20

    window_size = 28
    combined_features = np.stack((feature, feature_delta))
    windows = int(combined_features.shape[2] / window_size)
    combined_features = np.reshape(combined_features[:,:,0:windows*window_size], (2, height, windows*window_size))

    data = []
    for w in range(windows):
        data.append(combined_features[:,:,w*window_size:(w+1)*window_size])

    return np.array(data, dtype=np.float32)

def reshape_input(windows):

    input_d = windows.shape[1] #Depth
    input_h = windows.shape[2] #Height
    input_w = windows.shape[3] #Width

    if FLAGS.model_type.lower() == 'mel':
        return windows.reshape(windows.shape[0], input_h, input_w, input_d)
    else:
        windows = windows - windows.mean()
        windows = windows/LA.norm(windows)
        return windows.reshape(windows.shape[0], input_h*input_w*input_d)

def predict(windows, model_type='mel'):

    if model_type == 'mel':
        model_path='keras_model_mel/saved_models/sound_classifier.json'
    else:
        model_path='keras_model_mfcc/saved_models/sound_classifier.json'

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #adapting our input to the model that we'll use
    windows = reshape_input(windows)
    #generate predictions
    scores = loaded_model.predict(windows)

    print('We think this is a....')
    print(get_final_prediction(scores))

def main(_):

    if FLAGS.model_type.lower() not in ['mel', 'mfcc']:
        print('Sorry this model doesn''t exist, choose from mel or mfcc')
        sys.exit()

    if FLAGS.file_path == 'data/cats_dogs.wav':
        print('We will classify the audio in the file under data/cats_dogs/cat_1.wav')
    elif '.wav' not in FLAGS.file_path:
        print('Please submit an audio file in WAV format')
        sys.exit()
    elif os.path.exists(FLAGS.file_path) == False:
        print('Cannot find the file, please resubmit')
        sys.exit()
    else:
        print('Let''s classify this file: ' + FLAGS.file_path)

    feature, feature_delta = pre_process_file(FLAGS.file_path, FLAGS.model_type)
    audio_windows = process_file(feature, feature_delta)
    predict(audio_windows, FLAGS.model_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='mel', help='Choose from mel or mfcc model to classify.')
    parser.add_argument('--file_path', type=str, default='data/cats_dogs/cat_1.wav', help='File you want to analyse.')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
