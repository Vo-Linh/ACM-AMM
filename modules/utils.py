import scipy.io.wavfile
import numpy as np
import streamlit as st
import pickle
import os

from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

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
        st.subheader("Add audio meeting to the Database Base")

        with st.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Upload a file to the Database Base:",
                type=["wav"],
                accept_multiple_files=True)
            submitted = st.form_submit_button("Upload!")

        if uploaded_files and submitted:
            for uploaded_file in uploaded_files:
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                write_audio(os.path.join(AUDIO_DIR, uploaded_file.name), samplerate=44100,
                            audio=uploaded_file.read())

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

def store_vector(vector_store_path, raw_documents, use_existing_vector_store, document_embedder):
    """
    Stores document vectors in a vector store.

    Parameters:
    - vector_store_path (str): Path to the vector store file.
    - raw_documents (list): List of raw document texts.
    - use_existing_vector_store (str): Whether to use an existing vector store.
    - document_embedder: Embedding model to convert documents to vectors.
    """
    vector_store_exists = os.path.exists(vector_store_path)
    vectorstore = None
    if use_existing_vector_store == "Yes" and vector_store_exists:
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
        with st.sidebar:
            st.success("Existing vector store loaded successfully.")
    else:
        with st.sidebar:
            if raw_documents:
                with st.spinner("Splitting documents into chunks..."):
                    text_splitter = CharacterTextSplitter(
                        chunk_size=2000, chunk_overlap=200)
                    documents = text_splitter.split_documents(raw_documents)

                with st.spinner("Adding document chunks to vector database..."):
                    vectorstore = FAISS.from_documents(
                        documents, document_embedder)

                with st.spinner("Saving vector store"):
                    with open(vector_store_path, "wb") as f:
                        pickle.dump(vectorstore, f)
                st.success("Vector store created and saved.")
            else:
                st.warning("No documents available to process!", icon="⚠️")
    return vectorstore
