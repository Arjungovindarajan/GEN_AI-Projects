from models.chat import generate_answer, llama_answer, rag_answer

class AIAgent:
    def __init__(self):
        pass

    def process_query(self, question: str, context: str = None, method: str = "gpt-j"):
        """
        Process a query by deciding which model or approach to use.
        - Default to using GPT-J or LLaMA (open-source models).
        """
        if context:
            # Use RAG or LLaMA if context is available
            if method == "rag":
                return rag_answer(question, context)
            elif method == "llama":
                return llama_answer(question, context)
            else:
                return generate_answer(question, context)
        else:
            # Use GPT-J for general queries
            return generate_answer(question)

# Instantiate AI Agent
ai_agent = AIAgent()
