import sys
sys.path.append('/Users/jwlee-pro/anaconda3/envs/llm-quantize/lib/python3.11/site-packages')

from typing import List

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from llm import get_llm
from postprocess import cutoff, get_reranker
from prompt import prompt
from query_expansion import get_hyde_transformer
from retriever import retrieve_chroma


class QueryExpansionEvent(Event):
    query_bundle: QueryBundle


class RetrieverEvent(Event):
    retrieved_nodes: List[NodeWithScore]


class CutoffEvent(Event):
    retrieved_nodes_with_score: List[NodeWithScore]


class RerankEvent(Event):
    retrieved_nodes_with_score: List[NodeWithScore]


class PostprocessEvent(Event):
    retrieved_contents: List[dict]


class PromptEvent(Event):
    prompt: str
    retrieved_contents: List[str]


class ShinhanWorkflow(Workflow):

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        request_timeout: float = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(request_timeout, verbose=verbose)
        self.llm = get_llm(
            model=model, temperature=temperature, request_timeout=request_timeout
        )
        self.hyde = get_hyde_transformer(self.llm)
        self.reranker = get_reranker(self.llm)

    # @step
    # async def query_expansion(self, ctx: Context, ev: StartEvent) -> QueryExpansionEvent:
    #     ctx.data["query"] = ev.query
    #     ctx.data["chat_history"] = ev.chat_history
    #     query_bundle = self.hyde.run(ev.query)
    #     ctx.data["query_bundle"] = query_bundle
    #     return QueryExpansionEvent(query_bundle=query_bundle)

    @step
    # async def retrieve(self, ctx: Context, ev: QueryExpansionEvent) -> RetrieverEvent:
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent:
        ctx.data["query"] = ev.query
        ctx.data["chat_history"] = ev.chat_history
        # return RetrieverEvent(retrieved_nodes=retrieve_hybrid(ev.query))
        # return RetrieverEvent(retrieved_nodes=description_bm25_retriever.retrieve(query))
        query = ev.query
        return RetrieverEvent(retrieved_nodes=retrieve_chroma(query))

    @step
    async def cutoff(self, ctx: Context, ev: RetrieverEvent) -> CutoffEvent:
        retrieved_nodes = ev.retrieved_nodes
        # query_bundle = ctx.data["query_bundle"]
        # query = ctx.data["query"]
        return CutoffEvent(
            retrieved_nodes_with_score=cutoff.postprocess_nodes(retrieved_nodes)
        )

    # @step
    # async def rerank(self, ctx: Context, ev: CutoffEvent) -> RerankEvent:
    #     nodes = ev.retrieved_nodes_with_score
    #     query_bundle = ctx.data["query_bundle"]
    #     return RerankEvent(retrieved_nodes_with_score=self.reranker.postprocess_nodes(nodes, query_bundle=query_bundle))

    @step
    async def prompt(self, ctx: Context, ev: CutoffEvent) -> PromptEvent:
        query = ctx.data["query"]
        chat_history = ctx.data["chat_history"]
        retrieved_contents = ev.retrieved_nodes_with_score
        return PromptEvent(
            prompt=prompt.format(
                chat_history=chat_history,
                query=query,
                retrieved_contents="\n\n".join(
                    [content.text for content in retrieved_contents]
                ),
            ),
            retrieved_contents=[
                content.metadata["source"] for content in retrieved_contents
            ],
        )

    @step
    async def generate(self, ev: PromptEvent) -> StopEvent:
        prompt = ev.prompt
        response = await self.llm.acomplete(prompt)
        result = {
            "response": str(response),
            "retrieved_contents": ev.retrieved_contents,
        }
        return StopEvent(result=result)


def get_workflow(
    model: str = "llama3.1:70b", temperature: float = 0.2, request_timeout: int = 600
) -> ShinhanWorkflow:
    return ShinhanWorkflow(
        model=model,
        temperature=temperature,
        request_timeout=request_timeout,
        verbose=True,
    )
