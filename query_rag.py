"""
NCERT RAG Query Script (compat wrapper).

Prefer using the package CLI:
  python -m ncert_rag.cli ask "your question"
"""

from ncert_rag.config import Settings
from ncert_rag.rag import answer_question, create_qa_chain, load_vector_db

def main():
    """
    Main interactive Q&A loop
    """
    print("=" * 60)
    print("🎓 NCERT RAG Question-Answering System")
    print("=" * 60)
    
    settings = Settings()

    # Load vector database
    try:
        print("📂 Loading vector database...")
        vector_db = load_vector_db(settings)
    except Exception as e:
        print(f"\n❌ Error loading vector database: {e}")
        print("\nPlease run 'python setup_rag.py' first to create the database.")
        return
    
    # Create QA chain
    qa_chain = create_qa_chain(vector_db, settings)
    
    print("\n" + "=" * 60)
    print("Ready to answer your questions!")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60)
    
    # Interactive loop
    while True:
        question = input("\n💭 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using NCERT RAG. Happy learning!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        try:
            print(f"\n❓ Question: {question}")
            print("\n🔍 Searching NCERT textbooks...")

            result = answer_question(qa_chain, question)

            print("\n" + "=" * 60)
            print("📝 Answer:")
            print("=" * 60)
            print(result["result"])

            print("\n" + "=" * 60)
            print("📚 Sources from NCERT:")
            print("=" * 60)
            for i, doc in enumerate(result.get("source_documents", []), 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                print(f"\n[Source {i}]")
                print(f"File: {source}")
                print(f"Page: {page}")
                print(f"Content preview: {doc.page_content[:200]}...")

            print("\n" + "=" * 60)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
