import streamlit as st
from dotenv import load_dotenv
import os
import json
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

HISTORY_FILE = "chat_history.json"

def load_all_sessions():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_all_sessions(data):
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# load api key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# --- Sidebar ---
with st.sidebar:
    st.title("Gemini RAG Bot")
    st.markdown("""
    ## Chat with your data
    - [Gemini](https://gemini.google.com/)
    - [LlamaIndex](https://www.llamaindex.ai/)
    - [Streamlit](https://streamlit.io/)
    """)



@st.cache_resource
def build_index(_documents):
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(_documents)

    return VectorStoreIndex(nodes)

def main():
    st.header("🫦 Chat with you data")

    # --------- Session State Initialization ---------
    # load all chat sessions
    if "sessions" not in st.session_state:
        st.session_state.sessions = load_all_sessions()

    # current chat id
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "default"

    # ensure session exists
    if st.session_state.current_chat not in st.session_state.sessions:
        st.session_state.sessions[st.session_state.current_chat] = []

    # LLM memory
    if "memory" not in st.session_state:
        st.session_state.memories = {}

    if st.session_state.current_chat not in st.session_state.memories:
        st.session_state.memories[st.session_state.current_chat] = ChatMemoryBuffer.from_defaults()

    # --------- Sidebar for Chat Management ---------
    with st.sidebar:
        st.title("💬 Chats")

        if st.button("➕ New Chat"):
            new_id = f"chat_{len(st.session_state.sessions)}"
            st.session_state.sessions[new_id] = []
            st.session_state.current_chat = new_id
            st.rerun()

        for chat_id in list(st.session_state.sessions.keys()):

            col1, col2 = st.columns([0.8, 0.2])

            with col1:
                if st.button(chat_id, key=f"select_{chat_id}"):
                    st.session_state.current_chat = chat_id
                    st.rerun()

            with col2:
                if st.button("🗑", key=f"del_{chat_id}"):
                    del st.session_state.sessions[chat_id]

                    # fallback
                    if len(st.session_state.sessions) > 0:
                        st.session_state.current_chat = list(st.session_state.sessions.keys())[0]
                    else:
                        st.session_state.current_chat = "default"
                        st.session_state.sessions["default"] = []

                    save_all_sessions(st.session_state.sessions)
                    st.rerun()

    # --------- RAG ---------
    documents = SimpleDirectoryReader(
        input_dir="./data",
        recursive=False
    ).load_data()

    # Gemini LLM
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        api_key=gemini_api_key,
        temperature=0.2
    )
    # Gemini Embedding
    Settings.embed_model = GoogleGenAIEmbedding(
        model="gemini-embedding-001",
        embed_batch_size=1,
        max_retries=5
    )
    # build index
    index = build_index(documents)

    # --------- Display chat history ---------
    messages = st.session_state.sessions[st.session_state.current_chat]
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --------- Chat interface ---------
    query = st.chat_input("Please enter your question...")

    # --------- Streaming Chat Logic ---------
    if query:
        # save user msg
        messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        chat_engine = index.as_chat_engine(
            chat_mode="condense_question",
            memory=st.session_state.memories[st.session_state.current_chat],
            streaming=True
        )

        response = chat_engine.stream_chat(query)

        # assistant container
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            for token in response.response_gen:
                full_response += token
                placeholder.markdown(full_response)

        # save assistant msg
        messages.append({"role": "assistant", "content": full_response})

        # persist ALL sessions
        save_all_sessions(st.session_state.sessions)

if __name__ == "__main__":
    main()