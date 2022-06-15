# A Deep Learning Model that detects  Emotions in Human Voice

For my final project as a data science student I choose the topic of emotion recognition in human voice. I developed a Convolutional Neural Network architecture that classifies emotions in Spectograms produces from labeled audio recordings. I also developed a streamlit based web application that records live audio from the user, predicts the underline emotion and displays an emotion-specific response back to the user.

For the project the Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D) was used (https://www.kaggle.com/datasets/ejlok1/cremad)- a balanced dataset which include 7442 audio recordings, labeled with emotions belonging to six categories: fear, disgust, happiness, sadness, anger or neutral.

Emotions have been known to affect human speech resulting with differentiated patterns in the sound wave. An audio can be transformed into a Spectrogram image which is a visual representation of the spectrum of frequencies of a signal as it varies with time. This enables for the use of image based deep learning models to classify emotions from voice recordings.

The repository includes the following:
* data_exploration.ipynb - exploration of audio processing and data extraction techniques
* data_processing.ipynb - processing of the CREMA-D dataset into spectograms
* final_model.ipynb - model development, training and evaluation
* web_app - a folder that contains the web_app script using a saved model (cnn_final.h5) to predict emotion from a live audio recording
