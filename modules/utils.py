import scipy.io.wavfile
import numpy as np
import streamlit as st
import os



def upload_file(docs_path, audio_path):
    """
    Function to upload audio files and save them to specified directories.

    Parameters:
    - docs_path (str): Path to the directory for storing documents.
    - audio_path (str): Path to the directory for storing audio files.
    """
    with st.sidebar:
        DOCS_DIR = os.path.abspath(docs_path)
        AUDIO_DIR = os.path.abspath(audio_path)

        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)
        if not os.path.exists(AUDIO_DIR):
            os.makedirs(AUDIO_DIR)
        st.subheader("Add to the Knowledge Base")

        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Upload a file to the Knowledge Base:",
                type=["mp3", "wav"],
                accept_multiple_files=True)
            submitted = st.form_submit_button("Upload!")

        if uploaded_files and submitted:
            for uploaded_file in uploaded_files:
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                write_audio(os.path.join(AUDIO_DIR, uploaded_file.name), samplerate=44100,
                            audio=uploaded_file.read())
                # with open(os.path.join(AUDIO_DIR, uploaded_file.name), "wb") as f:
                #     f.write(uploaded_file.read())

        return uploaded_files, submitted


def write_audio(filename, samplerate, audio):
    """
    Function to write audio data to a WAV file.

    Parameters:
    - filename (str): Name of the WAV file to write.
    - samplerate (int): Sampling rate of the audio.
    - audio (bytes): Raw audio data as bytes.
    """
    # Converting raw audio bytes to numpy array
    audio_data = np.frombuffer(audio, dtype=np.int16)
    # Writing audio data to WAV file
    scipy.io.wavfile.write(filename, samplerate, audio_data)
