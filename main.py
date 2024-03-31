#!/usr/bin/env python3
# -*- coding: utf-8
# ----------------------------------------------------------------------------
# Created By  : Linh Vo Van
# Organization: ACM Lab
# Created Date: 2024/03/15
# version ='1.0'
# Description:
# Automatic Meeting Assistant
# This application leverages language models and a knowledge base to provide
# comprehensive responses to user queries within a chat interface.

# Key Features:
# - Knowledge Base Creation: Users can upload documents to a knowledge base.
# - Document Retrieval: The application retrieves relevant documents from the
#   knowledge base based on user queries.
# - LLM Response Generation: It uses a large language model to generate
#   informative responses, considering both the user's query and retrieved
#   context.

# External Libraries:
# - Streamlit: Provides a user-friendly web interface.
# - langchain: Facilitates working with language models and vector stores.
# - NVIDIA AI Endpoints: Connects to NVIDIA's AI models for embedding and response generation.

# Environment Variables:
# - NVIDIA_API_KEY: Required for accessing NVIDIA AI services.

# Component Structure:
# 1. Knowledge Base Management: Handles file uploads and storage.
# 2. Embedding Model and LLM: Loads and initializes language models and embedders.
# 3. Vector Database Store: Creates and manages a FAISS vector store for document retrieval.
# 4. LLM Response Generation and Chat: Manages user interaction, retrieves relevant documents,
#    generates responses using the LLM, and presents conversations in a chat interface.
# ---------------------------------------------------------------------------
# ========================================
# Document Loader
# ========================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pickle
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import streamlit as st
import os

from modules.audio2dia import Audio2Dia
from modules.utils import write_audio

st.set_page_config(layout="wide")

with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    AUDIO_DIR = os.path.abspath("./uploaded_audios")

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

# ========================================
#  Audio to Dialogue Model
# ========================================
if uploaded_files and submitted:
    model_audio = Audio2Dia(name_model='large-v2',
                            batch_size=16,
                            compute_type = "float16",
                            device="cuda")
    model_audio.generate(os.path.join(AUDIO_DIR, uploaded_file.name),
                os.path.join(DOCS_DIR, uploaded_file.name))

# ========================================
#  Embedding Model and LLM
# ========================================


llm = ChatNVIDIA(model="mixtral_8x7b", device="cuda:0")
document_embedder = NVIDIAEmbeddings(
    model="nvolveqa_40k", model_type="passage", device="cuda:0")
query_embedder = NVIDIAEmbeddings(
    model="nvolveqa_40k", model_type="query", device="cuda:0")

# ========================================
# Vector Database Store
# ========================================


with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio("Use existing vector store if available", [
                                         "Yes", "No"], horizontal=True)

# Path to the vector store file
vector_store_path = "vectorstore.pkl"

# Load raw documents from the directory
raw_documents = DirectoryLoader(DOCS_DIR).load()


# Check for existing vector store file
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

# ========================================
# LLM Response Generation and Chat
# ========================================

st.subheader("Automatic Meeting Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ========================================
# Design Prompt

prompt_indetify_topic = ChatPromptTemplate.from_template(
    "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is identify what is topic discussion in this {augmented_user_input}.Return the name of the topic and nothing else:"
)
prompt_indetify_context = ChatPromptTemplate.from_template(
    "You are a highly skilled AI trained in language comprehension and summarization. I would like you generate context of {augmented_user_input}"
)
prompt_summarize = ChatPromptTemplate.from_messages(
    [("system", "You are a highly skilled AI trained in language comprehension and summarization."),
     ("user", "Based on {context_dialogue} and {topic} I would you summarize {input} into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points.")]

)
prompt_rephase = ChatPromptTemplate.from_messages(
    [("system", "You are a highly skilled AI trained in language comprehension and summarization."),
     ("user",  "Dialogue:{context}. \nUser query: {input}\nBased on your knowledge, assess whether the queries provided by the user contain sufficient and clear information, and provide guidance to the user regarding the specificity required for these queries.")]
)
# ========================================
user_input = st.chat_input("Can you tell me what NVIDIA is known for?")
llm = ChatNVIDIA(model="mixtral_8x7b")

chain = llm | StrOutputParser()
chain_summarize = prompt_summarize | llm | StrOutputParser()
chain_rephase = prompt_rephase | llm | StrOutputParser()

if user_input and vectorstore != None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + \
        "\n\nQuestion: " + user_input + "\n"
    topic_generate = prompt_indetify_topic | chain
    out_topic = topic_generate.invoke(
        {"augmented_user_input": augmented_user_input})
    context_generate = prompt_indetify_context | chain
    out_context = context_generate.invoke(
        {"augmented_user_input": augmented_user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # for response in chain_summarize.stream({"context_dialogue": out_context, "topic": out_topic, "input": augmented_user_input}):
        for response in chain_rephase.stream({"input": user_input, "context":{context}}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
