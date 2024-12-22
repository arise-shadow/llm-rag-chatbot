from typing import Generator

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import StreamingAgentChatResponse


class MockStreamingAgentChatResponse(StreamingAgentChatResponse):

    def __init__(self, sentence: str, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = 3
        self.data = sentence
        self.generator = self.string_to_generator()

    @property
    def response_gen(self) -> Generator[str, None, None]:
        for chunk in self.generator:
            yield chunk

    def string_to_generator(self) -> Generator[str, None, None]:
        for i in range(0, len(self.data), self.chunk_size):
            # Yield a chunk of the string asynchronously
            yield self.data[i : i + self.chunk_size]


def convert_to_chat_message(message: dict) -> ChatMessage:
    if message["role"] == "user":
        return ChatMessage(role=MessageRole.USER, content=message["content"])
    return ChatMessage(role=MessageRole.ASSISTANT, content=message["content"])
