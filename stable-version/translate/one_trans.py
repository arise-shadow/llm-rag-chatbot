# translate/common.py
from environment import detect_environment
from translate.gpu import translate_with_gpu
from translate.furiosa import translate_with_furiosa

def translate_text(input_text, source_language="ko", target_language="en"):
    """
    Translates text based on the detected environment.

    Args:
        input_text (str): Text to be translated.
        source_language (str): Source language (default: "ko").
        target_language (str): Target language (default: "en").

    Returns:
        str: Translated text.
    """
    environment = detect_environment()

    if environment == "gpu":
        return translate_with_gpu(input_text, source_language, target_language)
    elif environment == "furiosa":
        return translate_with_furiosa(input_text, source_language, target_language)
    else:
        raise EnvironmentError(
            "No suitable environment detected. Please ensure GPU or Furiosa RNGD is available."
        )