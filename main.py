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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pickle
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.chat_models.huggingface import ChatHuggingFace

import streamlit as st
import os


from modules.utils import upload_file
from modules.audio2dia import Audio2Dia

# ========================================
# Document Loader
# ========================================
st.set_page_config(layout="wide")

DOCS_DIR_PATH = "database/uploaded_docs"
AUDIO_DIR_PATH = "database/uploaded_audios"

uploaded_files, submitted = upload_file(DOCS_DIR_PATH, AUDIO_DIR_PATH)

#  Audio to Dialogue Model
if uploaded_files and submitted:
    for uploaded_file in uploaded_files:
        model_audio = Audio2Dia(name_model='large-v2',
                                batch_size=16,
                                compute_type="int8",
                                device="cpu")
        model_audio.generate(os.path.join(AUDIO_DIR_PATH, uploaded_file.name),
                                os.path.join(AUDIO_DIR_PATH, uploaded_file.name))
# ========================================
#  Embedding Model and LLM
# ========================================

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)
document_embedder = HuggingFaceHubEmbeddings()

# ========================================
# Vector Database Store
# ========================================


with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio("Use existing vector store if available", [
                                         "Yes", "No"], horizontal=True)

# Path to the vector store file
vector_store_path = "database/vectorstore/vectorstore.pkl"

# Load raw documents from the directory
raw_documents = DirectoryLoader(os.path.abspath(DOCS_DIR_PATH)).load()


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
user_input = st.chat_input("Can you tell me what is known for?")

# ===== Prompt pipeline =====
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

    # generate specific prompt
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
