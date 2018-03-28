# Cats and Dogs classification challenge

This project was proposed as part of a group coursework for the Data Analytics modules at [QMUL](https://www.qmul.ac.uk/). The goal is to be able to classify correctly cat and dog audio files.

The dataset comes from [Kaggle](https://www.kaggle.com/mmoreaux/audio-cats-and-dogs).

The repository contains a number of folders and files described below:

* **data**: raw files.
* **data_processed**: dataframes divided into training and test with features extracted and ready for analysis.
* **keras_model_mel**: A [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) exploration using the mel and mel-deltas spectrograms from the audio files, complete with graphs, results and summary. It initially explores several learning and dropout rates to identify potentially effective ones, then it further fine tunes this into a model (saved under saved_models).
* **keras_model_mfcc**: A [Perceptron Neural Network](https://en.wikipedia.org/wiki/Perceptron) exploration using the MFCCs and MFCC deltas from the audio files, complete with graphs, results and summary. It initially explores several learning and dropout rates to identify potentially effective ones, then it further fine tunes this into a model (saved under saved_models).
* **tensorflow_model**: A similar architecture to the one in the keras_model_mel folder which uses both high and low level APIs from [Google's Tensorflow](https://www.tensorflow.org/) library. It was important to try lower level libraries to understand the programming differences involved.

In addition to this, there're multiple Jupyter Notebooks on the root folder, as well as inside the model folders numbered in order of progress. In them there's commentary and interesting outputs we encourage exploring as the feature selection process led to the model creation and fine-tuning.

Results are encouraging and a high accuracy (~89-90%) is achieved despite the small sample size.

.... to be continued ...
