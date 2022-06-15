import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa, librosa.display
import youtube_dl
import sounddevice as sd
import soundfile as sf

from utils import create_wave, create_spectogram, make_prediction
from utils import ydl_opts, CHUNK_SIZE

# set initial values of some variables to enable passing them between runs)
if 'mydata' not in st.session_state:
	st.session_state.mydata = None
if 'x' not in st.session_state:
	st.session_state.x = 0
if 'sr' not in st.session_state:
	st.session_state.sr = 1600
if 'image_name' not in st.session_state:
	st.session_state.image_name = None
if 'spectog_name' not in st.session_state:
	st.session_state.spectog_name = None
if 'emotion' not in st.session_state:
	st.session_state.emotion = None

choices = ['Yes', 'No']

# Webpage Visuals
st.title('Emotion Recognition app')
st.write('A helper app that detects emotion from voice and makes suggestions accordingly')
st.image('files/feelings.png')

# Get Audio from User
FILENAME  = 'audio.wav'
DURATION = 3  #seconds
SAMPLERATE = 16000
if  st.button('Record'):
    with st.spinner(f'Recording for {DURATION} seconds ....'):
        st.session_state.mydata = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1, blocking=True)
        sd.wait()
        sf.write(f'output/{FILENAME}', st.session_state.mydata, SAMPLERATE)
    st.success("Recording completed")

if st.session_state.mydata is not None:
    audio_file = open(f'output/{FILENAME}', "rb")
    st.audio(audio_file.read())

    # "Process Audio" - audio from video, display audio, creates Wave_image, update session_state variables
    if st.checkbox('Process Audio'):
        st.session_state.image_name, st.session_state.x, st.session_state.sr = create_wave(FILENAME)
        
        # Display created Wave_Image
        if st.session_state.image_name is not None:
            st.image(f'output/{st.session_state.image_name}')

        # "Create Spectogram" Button
        if st.checkbox('Create Spectogram'):
            st.session_state.spectog_name = create_spectogram(st.session_state.x, st.session_state.sr, st.session_state.image_name)
            
            # Display created Spectogram
            if st.session_state.spectog_name is not None:
                st.image(f'output/{st.session_state.spectog_name}')

            # Predict emotion from Spectogram
            if st.checkbox('Can you detect any emotions?'):
                st.session_state.emotion = make_prediction(st.session_state.spectog_name)

                # Display predicted emotion
                if st.session_state.emotion is not None:
                    st.title(f'There might be some: {st.session_state.emotion}')

                if st.button('What now?'):
                    if st.session_state.emotion == 'ANGER':
                        st.video('https://www.youtube.com/watch?v=6tqmXTYa3Xw&ab_channel=StephaniePittman')
                    if st.session_state.emotion == 'DISGUST':
                        st.image('files/puppy.jpg')
                    if st.session_state.emotion == 'FEAR':
                        st.video('https://www.youtube.com/watch?v=iaQed_Xdyvw&ab_channel=Calm')
                    if st.session_state.emotion == 'SADNESS':
                        st.video('https://www.youtube.com/watch?v=0Hkn-LSh7es&ab_channel=WaltDisneyStudiosMalaysia')
                    if st.session_state.emotion == 'HAPPINESS':
                        st.image('files/abundance.jpg')
                    if st.session_state.emotion == 'NEUTRAL':
                        st.image('files/swiss.jpg')
