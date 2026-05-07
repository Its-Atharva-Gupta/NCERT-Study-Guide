from __future__ import annotations

import os
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

from .config import Settings


DEFAULT_SYSTEM_PROMPT = """You are a helpful teacher assistant for NCERT textbooks.
Use the provided context to answer the student's question.
If you don't know the answer based on the context, say so honestly - don't make things up.
Explain concepts clearly and simply, as if teaching a student."""

DEFAULT_HUMAN_PROMPT = """Context from NCERT:
{context}

Student's Question: {question}"""


def load_vector_db(settings: Settings) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.resolved_device()},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(settings.vector_db_dir),
        embedding_function=embeddings,
    )


def _create_llm(settings: Settings):
    if settings.llm_provider != "groq":
        raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")

    try:
        from langchain_groq import ChatGroq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Groq support requires `langchain-groq`. Install deps with `uv sync`."
        ) from e

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.temperature,
        groq_api_key=groq_api_key,
    )


def create_qa_chain(vector_db: Chroma, settings: Settings) -> RetrievalQA:
    llm = _create_llm(settings)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DEFAULT_SYSTEM_PROMPT),
            ("human", DEFAULT_HUMAN_PROMPT),
        ]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": settings.top_k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def answer_question(
    chain: Any,
    question: str,
    *,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict:
    if chat_history is None:
        chat_history = []

    if chat_history:
        history_lines: list[str] = []
        for user_msg, assistant_msg in chat_history:
            history_lines.append(f"User: {user_msg}")
            history_lines.append(f"Assistant: {assistant_msg}")
        question = "Previous conversation:\n" + "\n".join(history_lines) + "\n\nCurrent question:\n" + question

    return chain.invoke({"query": question})
