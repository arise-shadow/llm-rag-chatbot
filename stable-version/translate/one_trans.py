import logging
from environment import detect_environment

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Store the detected environment and the corresponding translation function
_environment = None
_translation_function = None

def initialize_translation_environment():
    """
    Detects the translation environment and sets the corresponding translation function.

    Raises:
        EnvironmentError: If no suitable environment is detected.
    """
    global _environment, _translation_function
    _environment = detect_environment()
    
    if _environment == "gpu":
        from translate.gpu import translate_with_gpu
        _translation_function = translate_with_gpu
        logging.info("Translation environment detected: GPU. Using GPU for translations.")
    elif _environment == "furiosa":
        from translate.furiosa import translate_with_furiosa
        _translation_function = translate_with_furiosa
        logging.info("Translation environment detected: Furiosa NPU. Using NPU for translations.")
    else:
        error_message = (
            "No suitable environment detected. "
            "Ensure that either a CUDA-enabled GPU or Furiosa NPU is properly configured."
        )
        logging.error(error_message)
        raise EnvironmentError(error_message)
        

def translate_text(input_text, source_language="Korean", target_language="English"):
    """
    Translates text using the pre-determined environment and translation function.

    Args:
        input_text (str): Text to be translated.
        source_language (str): Source language (default: "Korean").
        target_language (str): Target language (default: "English").

    Returns:
        str: Translated text.

    Raises:
        RuntimeError: If the translation environment is not initialized.
    """
    if _translation_function is None:
        raise RuntimeError("Translation environment not initialized. Call initialize_translation_environment first.")
    
    return _translation_function(input_text, source_language, target_language)