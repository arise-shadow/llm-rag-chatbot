from translate.one_trans import initialize_translation_environment, translate_text

# Initialize the translation environment
try:
    initialize_translation_environment()
except EnvironmentError as e:
    print(f"Failed to initialize translation environment: {e}")
    exit(1)

# Perform translation
test_input = "안녕하세요, 번역 테스트입니다."
try:
    result = translate_text(test_input)
    print("Translated Text:", result)
except RuntimeError as e:
    print(f"Translation failed: {e}")