#!/usr/bin/env python3
# -*- coding: utf-8
"""
----------------------------------------------------------------------------
Created By  : Van-Linh Vo
Organization: ACM Lab
Created Date: 2024/04/25
version ='1.0'
Description:
Automatic Meeting Assistant
This application leverages language models and a knowledge base to provide
comprehensive responses to user queries within a chat interface.

langchain: Facilitates working with language models and vector stores.

Component Structure:
1. Prompt Templates: Defines various chat prompt templates for different tasks.
2. Example Prompts: Provides example prompts for few-shot learning.
3. Prompt Optimization Class: Contains the `PromptOptimze` class for optimizing prompts using an LLM.

Environment Variables:
- None

----------------------------------------------------------------------------
"""
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

# ========================================
# Design Prompt
# ========================================

contextualize_q_system_prompt = """Given a chat history and the latest user question or instruction \
which might reference context in the chat history, formulate a standalone instruction \
If you don't have enough information provided by the user, \
Just say that you don't have enough information. \
which can be understood without the chat history. Do NOT answer the instruction, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """You are an assistant for writing meeting minutes tasks. \
Use the following pieces of retrieved context to perform the require. \
If you don't have enough information provided by the user, \
Just say that you don't have enough information. \
Keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

prompt_basic = ChatPromptTemplate.from_messages(
    [("system", "You are a highly skilled AI trained in language."),
     ("user",  "{input}. Base on context, rephrase the query if necessary, and respond")]
)

prompt_rephase = ChatPromptTemplate.from_messages(
    [("system", "You are a highly skilled AI trained in language comprehension and summarization."),
     ("user",  "Dialogue:{context}. \nUser query: {input}\nBased on your knowledge, assess whether the queries provided by the user contain sufficient and clear information, \
     and provide guidance to the user regarding the specificity required for these queries.")]
)

prompt_seeking = ChatPromptTemplate.from_messages(
    [("system", "You're an expert at understanding what users are looking for. I'll provide the prompt typed by the user, and you'll determine what the user is intent"),
     ("user", "{prompt}")]
)

prompt_indetify = ChatPromptTemplate.from_messages(
    [("system", "You're proficient in domain language. Given the user's prompt and their intent, assist me in determining whether the provided information is sufficient or not.The respone only return yes or no, nothing else"),
     ("user", "user's prompt: {user_prompt}\n\nuser's intent: {user_intent}. The respone only return one word yes or no, nothing else, and remove anything else")]
)

prompt_finding = ChatPromptTemplate.from_messages(
    [("system", "You're an expert in domain language. Based on the user's prompt and their intention, help me identify what information the user might be missing from the prompt and return it under list format."),
     ("user", "user's prompt: {user_prompt}\n\nuser's intent: {user_intent}")]
)

examples = [
    {"input": '''potential pieces of information they might be missing:
    Specific Topic or Subject: It's unclear what specific topic or subject the user needs more detail on. They might need to specify which aspect they want more information about.''',
     "output": '''Certainly! Let's tailor the prompt format for LLMs:

    Task Description: Guide the LLM (Large Language Model) to effectively generate relevant information or responses based on user input.

    Input Format:

    Specify the general area or domain you need information about.
    Optionally, provide any specific prompts, questions, or keywords to guide the LLM's response.
    Output Format:

    Detailed information or responses relevant to the specified domain or prompts.
    Clear explanations, summaries, or answers to any specific questions provided.
    Constraints (if any): None'''},
]
# This is a prompt template used to format each individual example.
example_prompt_suggestion = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt_suggestion = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt_suggestion,
    examples=examples,
)

prompt_suggestion = ChatPromptTemplate.from_messages(
    [("system", "You're a prompt engineer."),
     few_shot_prompt_suggestion,
     ("user", "List missing information: {missing_information}")]
)


class PromptOptimze:
    def __init__(self, template_prompt, llm) -> None:
        self.template_prompt = prompt = ChatPromptTemplate.from_template(
            template_prompt)
        self.llm = llm
        self.chain = llm | self.template_prompt

    def optimize(self, prompt):
        prompt_opt = self.chain.invoke({"prompt": "prompt"})
        return prompt_opt
