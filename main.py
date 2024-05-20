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
# - OPENAI_AI_KEY Endpoints: Connects to NVIDIA's AI models for embedding and response generation.

# Environment Variables:
# - OPENAI_AI_KEY: Required for accessing OPENAI_AI_KEY services.

# Component Structure:
# 1. Knowledge Base Management: Handles file uploads and storage.
# 2. Embedding Model and LLM: Loads and initializes language models and embedders.
# 3. Vector Database Store: Creates and manages a FAISS vector store for document retrieval.
# 4. LLM Response Generation and Chat: Manages user interaction, retrieves relevant documents,
#    generates responses using the LLM, and presents conversations in a chat interface.
# ---------------------------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from langchain.chains import LLMChain, create_history_aware_retriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langserve import RemoteRunnable

import os
import torch
import streamlit as st

from modules.utils import upload_file, store_vector
from modules.audio2dia import Audio2Dia
from modules.prompt import *

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
                                compute_type="float16",
                                device="cuda",
                                device_index=0)
        model_audio.generate(os.path.join(AUDIO_DIR_PATH, uploaded_file.name),
                             os.path.join(DOCS_DIR_PATH, f"{uploaded_file.name[:-3]}txt"))
# ========================================
#  Embedding Model and LLM
# ========================================

llm = RemoteRunnable("http://localhost:8000/phi/")

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
# vectorstore = None
vectorstore = store_vector(vector_store_path, raw_documents,
                           use_existing_vector_store, document_embedder)

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
# Ininital Greeting Message
# ========================================
user_input = st.chat_input("Can you tell me what is known for?")
# Check if greeting has been shown before (using session state)
if 'shown_greeting' not in st.session_state:
    st.session_state['shown_greeting'] = False

if not st.session_state['shown_greeting']:
    with st.chat_message("assistant"):
        f = open("database/greeting_message/total.txt", "r")
        st.write(f.read())
        f.close()
        st.session_state['shown_greeting'] = True  # Mark as shown

# ===== Prompt pipeline =====
# parser actually doesn’t block the streaming output from the model, and instead processes each chunk individuall
chain_basic = prompt_basic | llm | StrOutputParser()

chain_seeking = prompt_seeking | llm | StrOutputParser()
chain_indetify = prompt_indetify | llm | StrOutputParser()
chain_finding = prompt_finding | llm | StrOutputParser()
chain_suggestion = prompt_suggestion | llm | StrOutputParser()

chain_rephase = prompt_rephase | llm | StrOutputParser()

if user_input and vectorstore != None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(user_input)

    ### Chain with chat history ###
    chat_history = []
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # =============================
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    # generate specific prompt
    augmented_user_input = "Context: " + context + \
        "\n\nQuestion: " + user_input + "\n"

    # the message will have a default bot icon with name "assistant"
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Due to each resposne is generate one words so it need to stored in one
        respond_rag = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        message_placeholder.markdown(respond_rag['answer'])
        chat_history.extend([HumanMessage(content=user_input), respond_rag['answer']])
    # store session_state of each interact
    st.session_state.messages.append(
        {"role": "assistant", "content": respond_rag['answer']})