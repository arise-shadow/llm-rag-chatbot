from typing import List

import chromadb
import numpy as np
import pandas as pd
import Stemmer
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

vectordb_save_path = "./chroma"
bm25_save_path = "./bm25"

# embed_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_1.5B_v5", device=3)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def get_vector_store_index(collection_name: str = "stella") -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=vectordb_save_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Load data
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


def get_chroma_retriever(collection_name: str = "stella", top_k=3):
    index = get_vector_store_index(collection_name)
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)


def get_bm25_retriever(top_k: int = 3):
    bm25_retriever = BM25Retriever.from_persist_dir(bm25_save_path)
    bm25_retriever.similarity_top_k = top_k
    bm25_retriever.stemmer = Stemmer.Stemmer("english")
    bm25_retriever.language = "en"
    return bm25_retriever


def normalize_dbsf(scores: List[str]):
    arr = np.array(scores)
    mean_value = np.mean(arr)
    std_value = np.std(arr)
    min_value = mean_value - 3 * std_value
    max_value = mean_value + 3 * std_value
    norm_score = (arr - min_value) / (max_value - min_value)
    return norm_score


def hybrid_cc(lexical_results, semantic_results, top_k=5, alpha=0.5):
    """
    Perform hybrid search using convex combination of BM25 and semantic scores.

    :param query: Search query (string)
    :param alpha: Weight for BM25 scores (0 <= alpha <= 1). 1-alpha is weight for semantic scores.
    """
    # Step 1: Perform BM25 Search
    bm25_ids = np.array([result.id_ for result in lexical_results])
    bm25_scores = np.array([result.score for result in lexical_results])

    # Step 2: Perform Semantic Search using ChromaRetriever
    chroma_ids = np.array([result.id_ for result in semantic_results])
    chroma_scores = np.array([result.score for result in semantic_results])

    # Step 3: Normalize the Scores
    bm25_scores_norm = normalize_dbsf(bm25_scores)
    chroma_scores_norm = normalize_dbsf(chroma_scores)

    ids = [bm25_ids, chroma_ids]
    scores = [bm25_scores_norm, chroma_scores_norm]

    df = pd.concat(
        [pd.Series(dict(zip(_id, score))) for _id, score in zip(ids, scores)], axis=1
    )
    df.columns = ["semantic", "lexical"]
    df = df.fillna(0)
    df["weighted_sum"] = df.mul((alpha, 1.0 - alpha)).sum(axis=1)
    df = df.sort_values(by="weighted_sum", ascending=False)

    retrieved_ids, retrieved_scores = (
        df.index.tolist()[:top_k],
        df["weighted_sum"][:top_k].tolist(),
    )
    retrieved_contents = []
    for idx, id in enumerate(retrieved_ids):
        content = next((node for node in lexical_results if node.id_ == id), None)
        if content is not None:
            content.score = retrieved_scores[idx]
            retrieved_contents.append(content)
            continue
        content = next((node for node in semantic_results if node.id_ == id), None)
        if content is not None:
            content.score = retrieved_scores[idx]
            retrieved_contents.append(content)

    return retrieved_contents


def retrieve_chroma(query: QueryBundle):
    chroma_retriever = get_chroma_retriever()
    return chroma_retriever.retrieve(query)


def retrieve_hybrid(query: QueryBundle):
    chroma_retriever = get_chroma_retriever()
    bm25_retriever = get_bm25_retriever()
    semantic_results = chroma_retriever.retrieve(query)
    lexical_results = bm25_retriever.retrieve(query)
    return hybrid_cc(lexical_results, semantic_results)
