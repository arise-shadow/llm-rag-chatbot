from llama_index.core import PromptTemplate
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform

hyde_prompt = PromptTemplate(
    """Instructions
You are an expert financial writer with in-depth knowledge of Vietnam Shinhan Bank's products and services. Your task is to create a detail passage answering the question: {context_str}.

The passage should be highly specific, detailed, and structured as if it were extracted from a comprehensive report or article. Ensure the response focuses solely on Shinhan Bank Vietnam and falls under one of the following categories: Card Services, Loan Services, or Deposit Products. Provide accurate, professional information that reflects the bank’s offerings and policies. 

Requirements
- The passage should be informative and provide clear insights into the relevant banking service.
- Always mention specific features, benefits, and terms related to Vietnam Shinhan Bank's Card, Loan, or Deposit products.
- Format the passage in a way that makes it easily integrated into larger banking documents or articles.
- Maintain a formal, authoritative tone suitable for financial and banking contexts.

Passage:"""
)


def get_hyde_transformer(llm) -> HyDEQueryTransform:
    return HyDEQueryTransform(llm=llm, hyde_prompt=hyde_prompt, include_original=True)
