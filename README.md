# A Deep Learning Model that detects  Emotions in Human Voice

For my final project as a data science student I choose the topic of emotion recognition in human voice and developed a Convolutional Neural Network architecture that classifies emotions from Spectograms produces from audio recordings labeled with emotions belonging to six categories: fear, disgust, happiness, sadness, anger or neutral. I also developed a streamlit-based web application that records live audio and makes a prediction, as well as desplays an emotion-adapted response back to the user.

The Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D) was used (https://www.kaggle.com/datasets/ejlok1/cremad)- a balanced dataset which include 7442 labeled audio recordings.

The repository includes the following files:

image_classifier_object.ipynb - shows the process of image processing and model training and evaluation
model_imageclassifier.h5 - saved trained model
live_image_classifier.py - script that uses the trained model to make prediction on images taken from a live webcam
utils.py - utils script
