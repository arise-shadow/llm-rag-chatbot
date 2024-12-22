from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

load_dotenv()


def get_llm(
    model: str = "llama3.1:70b",
    temperature: float = 0.2,
    request_timeout: int = 600,
    **kwargs
):
    if "gpt" in model:
        return OpenAI(
            model=model,
            temperature=temperature,
            request_timeout=request_timeout,
            **kwargs,
        )
    return Ollama(
        model=model,
        temperature=temperature,
        request_timeout=request_timeout,
        **kwargs,
    )
