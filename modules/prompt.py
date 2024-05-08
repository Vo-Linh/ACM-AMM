from langchain_core.prompts import ChatPromptTemplate

class PromptOptimze:
    def __init__(self, template_prompt, llm) -> None:
        self.template_prompt = prompt = ChatPromptTemplate.from_template(template_prompt)
        self.llm = llm
        self.chain = llm | self.template_prompt
    def optimize(self, prompt):
        prompt_opt = self.chain.invoke({"prompt": "prompt"})
        return prompt_opt