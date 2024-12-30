import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from nltk.tokenize import word_tokenize
from janome.tokenizer import Tokenizer
from kiwipiepy import Kiwi
from nltk.translate.meteor_score import meteor_score


def tokenize(text, lang):
    if lang == "English":
        # 영어: NLTK word_tokenize
        return word_tokenize(text)
    elif lang == "Korean":
        # 한국어: Kiwi 형태소 분석기
        kiwi = Kiwi()
        tokens = kiwi.tokenize(text)
        return [token.form for token in tokens]  # 형태소만 추출
    elif lang == "Japanese":
        # 일본어: Janome 형태소 분석기
        tokenizer = Tokenizer()
        return [token.surface for token in tokenizer.tokenize(text)]
    else:
        raise ValueError(f"Unsupported language: {lang}")
    
    
def calculate_bleu(reference: str, candidate: str, tgt_lang: str = "English") -> float:
    """
    Calculates BLEU score between a reference and a candidate translation.

    Parameters:
        reference (str): The reference (ground truth) text.
        candidate (str): The candidate (generated) translation.

    Returns:
        float: BLEU score.
    """
    # 텍스트를 토큰화
    reference_tokens = tokenize(reference, tgt_lang)
    candidate_tokens = tokenize(candidate, tgt_lang)
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)


def calculate_bert_score(reference: str, candidate: str, lang: str = "en", device: str = "cuda") -> float:
    """
    Calculates BERTScore between a reference and a candidate translation.

    Parameters:
        reference (str): The reference (ground truth) text.
        candidate (str): The candidate (generated) translation.
        lang (str): Language code (default is "en" for English).
        device (str): Device for computation ("cuda" for GPU, "cpu" for CPU).

    Returns:
        float: BERTScore (F1 score).
    """
    P, R, F1 = score([candidate], [reference], lang=lang, device=device)
    return float(F1.mean())