from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .ingest import build_vector_db, load_pdfs, split_documents
from .rag import answer_question, create_qa_chain, load_vector_db


def _settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        pdf_dir=Path(args.pdf_dir),
        vector_db_dir=Path(args.vector_db_dir),
        embedding_model=args.embedding_model,
        groq_model=args.groq_model,
        device=args.device,
        top_k=args.top_k,
        temperature=args.temperature,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    documents = load_pdfs(settings.pdf_dir)
    if not documents:
        raise RuntimeError(f"No documents loaded from: {settings.pdf_dir}")
    chunks = split_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    build_vector_db(chunks, settings)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    vector_db = load_vector_db(settings)
    chain = create_qa_chain(vector_db, settings)

    question = args.question.strip()
    if not question:
        raise ValueError("Question must be non-empty.")

    result = answer_question(chain, question)
    print(result["result"])
    if args.show_sources:
        for i, doc in enumerate(result.get("source_documents", []), 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            print(f"\n[Source {i}] {source} (page {page})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ncert-rag", description="NCERT RAG utilities")
    p.add_argument(
        "--pdf-dir",
        default=str(Settings().pdf_dir),
        help="Corpus path (a .txt file, a directory of .txt files, a .pdf file, or a directory of PDFs).",
    )
    p.add_argument("--vector-db-dir", default="vector_db")
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--groq-model", default=Settings().groq_model)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--top-k", type=int, default=Settings().top_k)
    p.add_argument("--temperature", type=float, default=0.3)

    sub = p.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Build or rebuild the vector DB from the corpus")
    ingest.add_argument("--chunk-size", type=int, default=1000)
    ingest.add_argument("--chunk-overlap", type=int, default=200)
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a question using an existing vector DB")
    ask.add_argument("question")
    ask.add_argument("--show-sources", action="store_true")
    ask.set_defaults(func=cmd_ask)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
