import pandas as pd
import numpy as np
import os, platform
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate


def translate_text(source_text: str, 
                   source_lang: str = "Korean", 
                   target_lang: str = "English", 
                   llm_model: str = "llama3.1", 
                   temperature: float = 0.2, 
                   request_timeout: int = 600, 
                   max_token_length: int = 200) -> str:
    """
    Translates text from one language to another using an LLM.

    Parameters:
        source_text (str): The text to translate.
        source_lang (str): The language of the source text. 
        target_lang (str): The target language for translation. 
        llm_model (str): Name of the LLM model to use. Default: "llama3.1".
        temperature (float): Sampling temperature for the LLM output. Default: 0.2.
        request_timeout (int): Maximum time to wait for a response from the LLM. Default: 600 seconds.
        max_token_length (int): Maximum token length for the LLM response. Default: 200.

    Returns:
        str: Translated text in the target language.
    """
    # Initialize the LLM with given parameters
    llm = Ollama(
        model=llm_model, 
        temperature=temperature, 
        request_timeout=request_timeout, 
        max_tokens=max_token_length
    )
    
    # Define a translation prompt template
    prompt = PromptTemplate(
        template="""This is an {source_lang} to {target_lang} translation. Please provide the {target_lang} translation for this text in as polite a tone as possible. \
Do not provide any explanations or text apart from the translation. The translation result must be written in {target_lang}.

{source_lang}: {source_text}

{target_lang}:"""
    )
    
    # Format the prompt with provided source and target languages and text
    full_prompt = prompt.format(
        source_lang=source_lang, 
        target_lang=target_lang, 
        source_text=source_text
    )
    
    # Generate the translation using the LLM
    result = llm.complete(full_prompt)
    return result.text


def main():
    """
    Main function to execute translation using translate_text function.
    """
    # Define default test input
    default_text = "안녕하세요, 번역 테스트를 위해 이 문장을 사용합니다."
    print("Original Text:", default_text)
    # Translate the default text
    try:
        translated_text = translate_text(
            source_text=default_text
        )
        print("Translated Text:", translated_text)
    except Exception as e:
        print("An error occurred during translation:", str(e))


if __name__ == "__main__":
    main()