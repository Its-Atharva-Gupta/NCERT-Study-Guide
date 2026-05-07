import streamlit as st

from ncert_rag.config import Settings
from ncert_rag.rag import answer_question, create_qa_chain, load_vector_db

st.set_page_config(page_title="NCERT RAG Q&A", page_icon="📚")

# Use @st.cache_resource so the DB and LLM only load once, saving memory/time
@st.cache_resource
def initialize_rag():
    settings = Settings()
    vector_db = load_vector_db(settings)
    return create_qa_chain(vector_db, settings)

# --- UI Layout ---
st.title("🎓 NCERT Teacher Assistant")
st.markdown("Chat with your NCERT corpus (RAG).")

try:
    qa_chain = initialize_rag()
    st.success("Database Loaded!")
except Exception as e:
    st.error(f"Failed to load database: {e}")
    st.stop()

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        sources = msg.get("sources")
        if sources:
            with st.expander("📚 Sources", expanded=False):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:** {s}")

# Build (user, assistant) history tuples for the RAG chain.
def _build_chat_history(messages: list[dict]) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            history.append((pending_user, content))
            pending_user = None
    return history

# Input box pinned at the bottom
user_prompt = st.chat_input("Ask a question about your NCERT content...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                chat_history = _build_chat_history(st.session_state.messages[:-1])
                result = answer_question(qa_chain, user_prompt, chat_history=chat_history)
                answer = result.get("result", "")
                st.markdown(answer)

                source_lines: list[str] = []
                for doc in result.get("source_documents", []) or []:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", None)
                    if page is None or page == "Unknown":
                        source_lines.append(f"{source}")
                    else:
                        source_lines.append(f"{source} (page {page})")

                if source_lines:
                    with st.expander("📚 Sources", expanded=False):
                        for i, s in enumerate(source_lines, 1):
                            st.markdown(f"**Source {i}:** {s}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": source_lines}
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
