from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor

cutoff = SimilarityPostprocessor(similarity_cutoff=0.3)


def get_reranker(llm, top_n=2):
    return LLMRerank(llm=llm, top_n=top_n)
