"""
UTILS 
- Helper functions that extract audio, process it, and create images and predictions
- "Global" Variables:
    - ydl_opts, CHUNK_SIZE: audio related formats
    - dict_emotions: binds emotion names with numeric value representing them
- Model:
    - Deep_Learning_model: trained to take a Spectogram produced from Audio, and Predict emotion from speech
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa, librosa.display
# import youtube_dl
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# # GLOBAL VARIABLES
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'ffmpeg-location': './',
    'outtmpl': "./output/%(id)s.%(ext)s",
}
CHUNK_SIZE = 5242880
dict_emotions = {0: 'ANGER', 1: 'DISGUST', 2: 'FEAR', 3: 'SADNESS', 4: 'HAPPINESS', 5: 'NEUTRAL'}

# Import trained model
model = load_model('files/cnn_final.h5')

@st.cache
def create_wave(file_name):
    """
    function that recieves a .wav file name, creates audio_wave image, and returns audio_parameters

    Parameters
    ----------
    file_name  - recieved from former steps. Name of Audio file.wav saved locally
    image_name - file_name without .wav ending. used to name the saved wave_image
    x, sr      - audio_array, sample_rate extracted from audio
    """
    image_name = file_name.split('.')[0] + '.png'
    x , sr = librosa.load(f'output/{file_name}', sr=16000)
    librosa.display.waveshow(x , sr)
    plt.savefig(f'output/{image_name}')
    plt.close()
    return image_name, x, sr

@st.cache
def create_spectogram(x, sr, image_name):
    """
    function that recieves audio_parameters and creates a Spectogram

    Parameters:
    image_name   - name of wave_image created in create_wave function
    spectog_name - image_name without ending + new ending. used to name the Spectograme_image
    X, Xdb       - Transformations of x_array in order to create the Spectogram
    """
    spectog_name = image_name.split('.')[0] + '_spectog' + '.png'
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log', cmap='jet')
    plt.colorbar()
    plt.savefig(f'output/{spectog_name}')
    plt.close()
    return spectog_name

@st.cache
def make_prediction(spectog_name):
    """
    function that takes the Spectogram produced from Audio, and uses a pre-trained model to predict the emotion
    """
    image = tf.keras.preprocessing.image.load_img(f'output/{spectog_name}', color_mode='rgb', target_size= (224,224))
    image_array = np.array(image)
    image_array_scaled = image_array.astype("float32")/255
    prediction_ohe = model.predict(image_array_scaled.reshape(-1, 224, 224, 3))
    prediction = np.argmax(prediction_ohe)
    emotion = dict_emotions[prediction]
    return emotion
