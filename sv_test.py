#!/usr/bin/env python
import torch
import streamlit as st
from fastapi import FastAPI
from transformers import AutoModelForSeq2SeqLM
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub
from langserve import add_routes

app = FastAPI(
    title="Server",
    version="1.0",
    description="Api server using Langchain's Runnable interfaces",
)

@st.cache_resource
def load_model():
    """Loads the large language model for text generation.

    This function utilizes the HuggingFace Pipeline to load the specified model ID
    from the HuggingFace model hub. It caches the loaded model using `st.cache_resource`
    to improve performance by avoiding redundant loads on every execution.

    Args:
        model_id (str): The ID of the model to load from the HuggingFace model hub
            (e.g., "microsoft/Phi-3-mini-4k-instruct").
        task (str, optional): The task for which the model is being loaded. Defaults to "text-generation".
        device (int, optional): The device on which to load the model (e.g., GPU index). Defaults to 1.
        model_kwargs (dict, optional): Keyword arguments passed to the HuggingFace Pipeline for model loading.
            Defaults to a dictionary containing parameters for beam search, sampling, and data type.

    Returns:
        transformers.pipeline.Pipeline: The loaded HuggingFace Pipeline object representing the LLM.
    """
    return HuggingFacePipeline.from_model_id(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        device=1,
        model_kwargs={
            "top_k": 30,
            "temperature": 0.01,
            "repetition_penalty": 1.03,
            'trust_remote_code': True,
            'do_sample': True,
            'torch_dtype': torch.bfloat16,
        },
        pipeline_kwargs={
            "max_new_tokens": 1024,
        },
    )

model = load_model()

add_routes(
    app,
    model,
    path="/phi",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
