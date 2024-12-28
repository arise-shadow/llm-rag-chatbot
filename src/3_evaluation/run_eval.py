import os
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score
from nltk.tokenize import word_tokenize
from janome.tokenizer import Tokenizer
from kiwipiepy import Kiwi
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate

# Define LLM
llm = Ollama(model="llama3.1:70b", request_timeout=600, temperature=0)

prompt = PromptTemplate(
    template="""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text in as polite a tone as possible. \
Do not provide any explanations or text apart from the translation.
The translation result must be written in {target_lang}.

{source_lang}: {source_text}

{target_lang}:"""
)

# 번역 함수
def translate(source_text, source_lang, target_lang, prompt):
    full_prompt = prompt.format(source_lang=source_lang, target_lang=target_lang, source_text=source_text)
    result = llm.complete(full_prompt)
    return result.text

# 언어별 토크나이저
def tokenize(text, lang):
    if lang == "eng":
        return word_tokenize(text)  # 영어: NLTK word_tokenize
    elif lang == "kor":
        kiwi = Kiwi()  # 한국어: Kiwi 형태소 분석기
        tokens = kiwi.tokenize(text)
        return [token.form for token in tokens]
    elif lang == "jpn":
        tokenizer = Tokenizer()  # 일본어: Janome 형태소 분석기
        return [token.surface for token in tokenizer.tokenize(text)]
    else:
        raise ValueError(f"Unsupported language: {lang}")

# 평가 함수
def evaluate(reference, candidate, tgt_lang, bert_model="microsoft/deberta-xlarge-mnli"):
    # 텍스트를 토큰화
    reference_tokens = tokenize(reference, tgt_lang)
    candidate_tokens = tokenize(candidate, tgt_lang)

    # BLEU 점수 계산
    bleu = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=SmoothingFunction().method1,
    )

    # METEOR 점수 계산
    meteor = meteor_score([reference_tokens], candidate_tokens)

    # BERTScore 계산
    P, R, F1 = score(
        [candidate],
        [reference],
        lang=tgt_lang,
        model_type=bert_model,
        device="cuda",  # GPU 사용
    )

    return {
        "BLEU": round(bleu, 4),
        "METEOR": round(meteor, 4),
        "BERT": round(F1.item(), 4),
    }

# 텍스트 파일 로더
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# 데이터 위치
data_dir = "/home/dudaji/joonwon/llm-rag-chatbot/data/flores/"

data_eng = load_text_file(f"{data_dir}/devtest.eng_Latn")
data_kor = load_text_file(f"{data_dir}/devtest.kor_Hang")
data_jpn = load_text_file(f"{data_dir}/devtest.jpn_Jpan")

# 번역 및 평가
pairs = [
    ("eng", "kor", data_eng, data_kor),
    ("eng", "jpn", data_eng, data_jpn),
    ("kor", "eng", data_kor, data_eng),
    ("kor", "jpn", data_kor, data_jpn),
    ("jpn", "eng", data_jpn, data_eng),
    ("jpn", "kor", data_jpn, data_kor),
]

results = []

print('evaluation start!')
for source_lang, target_lang, sources, references in pairs:
    for source, reference in zip(sources, references):


        # 번역 및 시간 측정
        start_time = time.time()
        candidate = translate(source, source_lang, target_lang, prompt)
        elapsed_time = time.time() - start_time  # 번역 소요 시간

        # 평가 실행
        evaluation = evaluate(reference, candidate, target_lang)

        # 결과 저장
        results.append({
            "Source Language": source_lang,
            "Target Language": target_lang,
            "Source": source,
            "Reference": reference,
            "Candidate": candidate,
            "Translation Time (s)": round(elapsed_time, 4),
            **evaluation,
        })

    # 결과 저장
    df = pd.DataFrame(results)
    save_dir = '/home/dudaji/joonwon/llm-rag-chatbot/data/translate'
    df.to_csv(f"{save_dir}/Eval_results_flores.csv", index=False, encoding="utf-8")
    print("Intermediate result saved :)")

# 결과 저장
df = pd.DataFrame(results)
save_dir = '/home/dudaji/joonwon/llm-rag-chatbot/data/translate'
df.to_csv(f"{save_dir}/Eval_results_flores.csv", index=False, encoding="utf-8")
print("Results saved: Eval_results_flores.csv")