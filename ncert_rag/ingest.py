from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


def load_documents(corpus_path: Path) -> list:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

    if corpus_path.is_file():
        if corpus_path.suffix.lower() == ".txt":
            return TextLoader(str(corpus_path), encoding="utf-8").load()
        if corpus_path.suffix.lower() == ".pdf":
            return PyPDFLoader(str(corpus_path)).load()
        raise ValueError(f"Unsupported corpus file type: {corpus_path}")

    pdfs = list(corpus_path.rglob("*.pdf"))
    txts = list(corpus_path.rglob("*.txt"))
    if pdfs and txts:
        raise ValueError(f"Mixed corpus types in {corpus_path}: contains both .pdf and .txt")
    if not pdfs and not txts:
        raise FileNotFoundError(f"No .pdf or .txt files found under: {corpus_path}")

    if txts:
        loader = DirectoryLoader(
            str(corpus_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        return loader.load()

    loader = DirectoryLoader(
        str(corpus_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    return loader.load()


def load_pdfs(pdf_dir: Path) -> list:
    return load_documents(pdf_dir)


def split_documents(
    documents: Sequence,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_db(chunks: Iterable, settings: Settings) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.resolved_device()},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        persist_directory=str(settings.vector_db_dir),
    )
