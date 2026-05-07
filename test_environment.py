"""
Test Script to Verify Your Environment Setup
Run this before setup_rag.py to check if everything is configured correctly
"""

import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.10 or higher"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Good!")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.10+")
        return False

def check_groq():
    """Check if Groq API key is configured"""
    print("\n🔑 Checking Groq API key...")
    import os

    if os.getenv("GROQ_API_KEY"):
        print("   ✅ GROQ_API_KEY is set")
        return True

    print("   ❌ GROQ_API_KEY is not set")
    print("   Set it, e.g.: export GROQ_API_KEY='your_key_here'")
    return False

def check_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking Python packages...")
    
    required = {
        'langchain': 'langchain',
        'langchain_community': 'langchain-community',
        'langchain_groq': 'langchain-groq',
        'langchain_text_splitters': 'langchain-text-splitters',
        'langchain_classic': 'langchain-classic',
        'chromadb': 'chromadb',
        'pypdf': 'pypdf',
        'sentence_transformers': 'sentence-transformers',
    }
    
    all_installed = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n   Run: pip install -r requirements.txt")
    
    return all_installed

def check_cuda():
    """Check if CUDA/GPU is available"""
    print("\n🎮 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available - {gpu_name}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("   ⚠️  CUDA not available - will use CPU (slower)")
            return True  # Not critical, system can work on CPU
    except ImportError:
        print("   ⚠️  PyTorch not installed - can't check GPU")
        print("   System will use CPU for embeddings (slower)")
        return True

def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")
    import os
    
    corpus_file = os.path.join("Books", "text.txt")
    if os.path.exists(corpus_file):
        size = os.path.getsize(corpus_file)
        if size > 0:
            print(f"   ✅ Found corpus file: {corpus_file} ({size} bytes)")
        else:
            print(f"   ⚠️  Found corpus file but it's empty: {corpus_file}")
    else:
        print(f"   ⚠️  Corpus file not found: {corpus_file}")
        print("   Put your NCERT text/data into Books/text.txt")
    
    return True

def main():
    print("=" * 60)
    print("🔧 NCERT RAG Environment Check")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Groq API Key", check_groq()),
        ("Python Packages", check_packages()),
        ("GPU/CUDA", check_cuda()),
        ("Directories", check_directories())
    ]
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    all_passed = all(result for _, result in checks)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All checks passed! You're ready to run setup_rag.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
    
    print("\nNext steps:")
    print("1. Make sure Books/text.txt exists and is non-empty")
    print("2. Run: python setup_rag.py")
    print("3. Then run: python query_rag.py")

if __name__ == "__main__":
    main()
