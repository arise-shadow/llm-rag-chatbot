import os
from furiosa_llm import LLM, SamplingParams

# Global instances for LLM and Sampling Parameters
furiosa_llm = None
sampling_params = None


def initialize_furiosa_llm(model_path,
                           devices: str = "npu:1:*", 
                           temperature: float = 0.2, 
                           max_tokens: int = 200):
    """
    Initializes the Furiosa LLM and sampling parameters globally.

    Parameters:
        model_path (str): Path to the LLM artifacts. 
        devices (str): Device string for Furiosa NPU. Default: "npu:1:*".
        temperature (float): Sampling temperature. Default: 0.
        max_tokens (int): Maximum number of tokens for generation. Default: 200.
    """
    global furiosa_llm, sampling_params

    if furiosa_llm is None:
        os.environ["RUST_BACKTRACE"] = "full"
        furiosa_llm = LLM.from_artifacts(model_path, devices=devices)

    if sampling_params is None:
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)


def apply_translation_template(source_lang: str, target_lang: str, source_text: str) -> str:
    """
    Creates a translation prompt using a predefined format.

    Parameters:
        source_lang (str): Source language.
        target_lang (str): Target language.
        source_text (str): The text to translate.

    Returns:
        str: The formatted translation prompt.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Translate from {source_lang} to {target_lang}: {source_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def translate_with_furiosa(source_text: str, 
                           source_lang: str = "Korean", 
                           target_lang: str = "English") -> str:
    """
    Translates text using the Furiosa-based LLM.

    Parameters:
        source_text (str): The text to translate.
        source_lang (str): Source language. Default: "Korean".
        target_lang (str): Target language. Default: "English".

    Returns:
        str: Translated text.
    """
    if furiosa_llm is None or sampling_params is None:
        initialize_furiosa_llm()

    # Create the translation prompt
    prompt = apply_translation_template(source_lang, target_lang, source_text)

    # Generate and return the translation
    output_txt = furiosa_llm.generate(prompt, sampling_params)
    return output_txt.outputs[0].text[2:]


def main():
    """
    Main function to execute translation using translate_with_furiosa.
    """
    test_text = "안녕하세요, 우주는 얼마나 넓나요?"
    print(f"Original Text: {test_text}")

    try:
        # Perform translation
        translated_text = translate_with_furiosa(test_text)
        print(f"Translated Text: {translated_text}")
    except Exception as e:
        print(f"An error occurred during translation: {e}")


if __name__ == "__main__":
    main()