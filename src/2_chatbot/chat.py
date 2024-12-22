from typing import List

from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import BaseTool

from llm import get_llm
from prompt import chat_system_prompt, react_chat_formatter
from retriever import get_vector_store_index


def get_chat_engine():
    index = get_vector_store_index(collection_name="card-product")
    llm = get_llm(model="gpt-4o-mini", temperature=0)
    return index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        llm=llm,
        system_prompt=chat_system_prompt,
        verbose=True,
    )


def get_tools() -> List[BaseTool]:
    index = get_vector_store_index(collection_name="card-product")

    def retrieve_nodes(query) -> List[NodeWithScore]:
        """Retrieve a list of nodes that are highly relevant to query"""
        chroma_retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
        return chroma_retriever.retrieve(query)

    shinhan_retriever_tool = FunctionTool.from_defaults(fn=retrieve_nodes)
    return [shinhan_retriever_tool]


def get_agent():
    tools = get_tools()
    llm = get_llm(model="gpt-4o-mini", temperature=0)
    # llm = get_llm(model="qwen2.5:72b", temperature=0.1)
    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        react_chat_formatter=react_chat_formatter,
        max_iterations=10,
        verbose=True,
    )
    return agent
