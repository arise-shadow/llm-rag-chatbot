import streamlit as st
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

from chat import get_agent
from utils import MockStreamingAgentChatResponse, convert_to_chat_message

# App title
st.set_page_config(page_title="💬 FuriosaAI Chatbot")

with st.sidebar:
    st.title("💬 FuriosaAI Chatbot")
    st.write("This QA chatbot for FuriosaAI Product.")

    st.subheader("Question Examples")
    questions = [
        "What are the specific example models available on HuggingFace Hub for the 'LlamaForCausalLM' architecture supported by Furiosa SDK?",
        "How does the 'furiosa-smi status' subcommand help in monitoring the utilization of NPU cores, and what specific details does it provide?",
        "What are the quantization methods supported by the Furiosa Quantizer?",
        "What are the minimum system requirements and initial setup steps needed to install Furiosa LLM?",
    ]
    for question in questions:
        st.code(question, language=None)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_response(query) -> StreamingAgentChatResponse:
    agent = get_agent()
    try:
        chat_history = [
            convert_to_chat_message(message) for message in st.session_state.messages
        ]
        return agent.stream_chat(query, chat_history)
    except ValueError:
        error_log = """I'm sorry, I did my best, but cannot find proper information for your question.
The more specific questions you ask, the more likely you are to get accurate answers."""
        return MockStreamingAgentChatResponse(error_log)


# User-provided prompt
if prompt := st.chat_input("Ask question anything about FuriosaAI"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = generate_response(prompt)
            placeholder = st.empty()
            full_response = ""
            for chunk in result.response_gen:
                full_response += chunk
                placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
