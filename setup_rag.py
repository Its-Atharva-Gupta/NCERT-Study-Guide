"""
NCERT RAG Setup Script (compat wrapper).

Prefer using the package CLI:
  python -m ncert_rag.cli ingest
"""

from ncert_rag.config import Settings
from ncert_rag.ingest import build_vector_db, load_pdfs, split_documents

def main():
    """
    Main function to set up the RAG system
    """
    print("=" * 60)
    print("🚀 NCERT RAG System Setup")
    print("=" * 60)

    settings = Settings()
    
    # Step 1: Load corpus documents
    try:
        documents = load_pdfs(settings.pdf_dir)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    if not documents:
        print(f"\n❌ No documents loaded from '{settings.pdf_dir}'. Add data and retry.")
        return
    
    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Create vector database
    build_vector_db(chunks, settings)
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print(f"\nYour vector database is ready at: {settings.vector_db_dir}")
    print("\nNext steps:")
    print("1. Run 'python query_rag.py' to ask questions")
    print("2. The system will search NCERT and answer using Groq")
    print("\nHappy learning! 📖")

if __name__ == "__main__":
    main()
