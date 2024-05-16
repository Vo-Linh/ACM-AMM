from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# ========================================
# Design Prompt
# ========================================
prompt_basic = ChatPromptTemplate.from_messages(
    [("system", "You are a highly skilled AI trained in language comprehension and summarization."),
     ("user",  "{input}")]
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
